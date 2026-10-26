"""Ketahanan medan (rencana 24 Okt, CATATAN §8): seberapa turun model AKTIF bila kondisi kamera/cahaya/belt berubah?
Tidak melatih apa pun; hanya menguji model yang ada.

A. Gangguan buatan pada foto belt 22 Okt (`1_foto_mentah/2026-10-22_water_v1`, 65 foto, 5 botol; label tutup & isi dari MANIFEST).
   Foto ini TIDAK ikut melatih cap_belt_v2 (21 Okt) maupun cairan_RH -> patokan = medan sama, botol baru.
   Gangguan: cahaya (gelap/terang), warna cahaya (hangat/dingin/pink), blur, noise, JPEG, kamera lebih tinggi (botol mengecil),
   warna belt diganti (piksel hijau belt -> abu/hitam/biru; ikut terlihat menembus botol bening).
B. Medan NYATA lain: foto studio Sep (shoot_v1: kain cokelat + 2 lampu; shoot_v2: alas putih, lampu kamar), label tutup dari nama file;
   semua botol kosong -> isi seharusnya 'kosong'.
C. Merek: model fold CV (runs/dasar_F*), hanya potongan yang TIDAK ikut dilatih fold itu, dengan gangguan yang sama pada potongannya.

  venv/Scripts/python.exe scripts/pipeline_ketahanan/k01_uji_ketahanan.py
  -> datasets/4_eksperimen/ketahanan/{hasil.jsonl, laporan.txt, contoh_gangguan.jpg}"""
import csv, glob, json, os, re, sys, time
from collections import defaultdict
import cv2, numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts", "pipeline_merek"))
D = os.path.join(REPO, "datasets"); OUT = os.path.join(D, "4_eksperimen", "ketahanan"); os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------- gangguan (frame BGR uint8 -> frame BGR uint8)
def _gain(im, b=1.0, g=1.0, r=1.0, k=1.0):
    return np.clip(im.astype(np.float32) * np.array([b * k, g * k, r * k], np.float32), 0, 255).astype(np.uint8)


def _kecil(im, s):                                  # kamera lebih tinggi: isi mengecil, tepi diisi pantulan latar
    H, W = im.shape[:2]; h, w = round(H * s), round(W * s)
    sm = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)
    t, l = (H - h) // 2, (W - w) // 2
    return cv2.copyMakeBorder(sm, t, H - h - t, l, W - w - l, cv2.BORDER_REFLECT)


def _belt(im, mode):                                # ganti warna belt hijau
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    m = (hsv[..., 0] >= 35) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 35) & (hsv[..., 2] >= 25)
    h = hsv.copy()
    if mode == "abu":
        h[..., 1][m] = 0
    elif mode == "hitam":
        h[..., 1][m] = 0; h[..., 2][m] = (h[..., 2][m] * 0.3).astype(np.uint8)
    elif mode == "biru":
        h[..., 0][m] = 110
    return cv2.cvtColor(h, cv2.COLOR_HSV2BGR)


def _jpeg(im, q):
    return cv2.imdecode(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, q])[1], cv2.IMREAD_COLOR)


def _noise(im, s):
    rng = np.random.default_rng(0)
    return np.clip(im.astype(np.float32) + rng.normal(0, s, im.shape), 0, 255).astype(np.uint8)


G = {  # nama: (kelompok, fungsi)
    "asli": ("patokan", lambda im: im),
    "gelap_50%": ("cahaya", lambda im: _gain(im, k=0.5)),
    "gelap_70%": ("cahaya", lambda im: _gain(im, k=0.7)),
    "terang_140%": ("cahaya", lambda im: _gain(im, k=1.4)),
    "terang_180%": ("cahaya", lambda im: _gain(im, k=1.8)),
    "hangat": ("warna cahaya", lambda im: _gain(im, b=0.8, r=1.15)),
    "dingin": ("warna cahaya", lambda im: _gain(im, b=1.2, r=0.85)),
    "pink": ("warna cahaya", lambda im: _gain(im, b=1.05, g=0.8, r=1.3)),
    "blur_s2": ("kamera", lambda im: cv2.GaussianBlur(im, (0, 0), 2)),
    "blur_s5": ("kamera", lambda im: cv2.GaussianBlur(im, (0, 0), 5)),
    "noise_15": ("kamera", lambda im: _noise(im, 15)),
    "jpeg_q20": ("kamera", lambda im: _jpeg(im, 20)),
    "kamera_tinggi_80%": ("jarak", lambda im: _kecil(im, 0.8)),
    "kamera_tinggi_60%": ("jarak", lambda im: _kecil(im, 0.6)),
    "belt_abu": ("latar", lambda im: _belt(im, "abu")),
    "belt_hitam": ("latar", lambda im: _belt(im, "hitam")),
    "belt_biru": ("latar", lambda im: _belt(im, "biru")),
}
G_MEREK = [g for g in G if not g.startswith("kamera_tinggi")] + ["resolusi_50%"]


def g_merek(name, im):
    if name == "resolusi_50%":
        return cv2.resize(cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA), im.shape[1::-1])
    return G[name][1](im)


def main():
    from rvm_kp import RVMAnalyzer, config as rc
    an = RVMAnalyzer("pytorch", profile="nano")
    res_path = os.path.join(OUT, "hasil.jsonl")
    done = {json.loads(l)["uji"] for l in open(res_path)} if os.path.exists(res_path) else set()   # lanjutkan bagian yang belum
    fo = open(res_path, "a")

    def judge(im):
        rr = an.analyze(im)
        if not rr:
            return dict(det=False)
        r = max(rr, key=lambda z: (z.box[2] - z.box[0]) * (z.box[3] - z.box[1]))
        return dict(det=True, cap=r.cap, p=round(r.p_liquid, 4), size=r.size, area=round(r.area_1920))

    # ---- A. foto belt 22 Okt + gangguan
    S = os.path.join(D, "1_foto_mentah", "2026-10-22_water_v1")
    M = [r for r in csv.DictReader(open(os.path.join(S, "MANIFEST.csv"), encoding="utf-8")) if os.path.exists(os.path.join(S, r["file"]))]
    t0 = time.time()
    for r in (M if "A" not in done else []):
        im = cv2.imread(os.path.join(S, r["file"]))
        lab = dict(cap_lab=r["cap"], isi="kosong" if r["level"] == "kosong" else ("bening" if r["liquid"].startswith("bening") else "berwarna"))
        for g, (kel, fn) in G.items():
            fo.write(json.dumps(dict(uji="A", file=r["file"], botol=r["bottle"], g=g, kel=kel, **lab, **judge(fn(im)))) + "\n")
    print("A selesai", len(M), "foto", round(time.time() - t0), "s", flush=True)

    # ---- B. studio Sep
    st = sorted(glob.glob(os.path.join(D, "2_dataset_cap", "cap_shoot_v1", "images", "*.jpg"))) + \
        sorted(glob.glob(os.path.join(D, "2_dataset_cap", "shoot_v2", "*", "*.jpg")))
    for f in (st if "B" not in done else []):
        n = os.path.basename(f); im = cv2.imread(f)
        cap = "off" if "_nocap_" in n else "on"
        set_ = "studio_v1_cokelat" if "cap_shoot_v1" in f else "studio_v2_putih"
        if "cap_shoot_v1" in f:                                          # urutan slot per kondisi: 3 topdown, 5 lift45, 2 side45
            k = int(re.search(r"_(?:no)?cap_(\d+)", n).group(1)); view = "atas" if k <= 3 else "miring"
        else:
            view = "atas" if "topdown" in n else "miring"
        fo.write(json.dumps(dict(uji="B", file=n, set=set_, view=view, cap_lab=cap, isi="kosong", g="asli", **judge(im))) + "\n")
    print("B selesai", len(st), "foto", flush=True)

    # ---- C. merek, model fold CV skema F, potongan yang tidak dilatih
    from m02_cv_dasar import rows, folds, CR, S as SM
    from ultralytics import YOLO
    R = [r for r in rows() if os.path.exists(f"{CR}/{r['id']}.jpg")]
    F = folds(R, "grup_F")
    for k in (range(5) if "C" not in done else []):
        m = YOLO(os.path.join(D, "4_eksperimen", "merek", "runs", f"dasar_F{k}", "weights", "last.pt"))
        ci = [i for i, v in m.names.items() if v == "mitra"][0]
        te = [r for r in R if F[r["grup_F"]] == k]
        ims = [cv2.imread(f"{CR}/{r['id']}.jpg") for r in te]
        for g in G_MEREK:
            ps = [float(p.probs.data[ci]) for p in m.predict([g_merek(g, im) for im in ims], imgsz=SM, verbose=False)]
            for r, p in zip(te, ps):
                fo.write(json.dumps(dict(uji="C", id=r["id"], produk=r["produk"], kelas=r["kelas"], fold=k, g=g, p=round(p, 4))) + "\n")
    print("C selesai", flush=True)
    fo.close()

    # ---- contoh gangguan (gambar untuk laporan)
    im = cv2.imread(os.path.join(S, M[len(M) // 3]["file"])); tiles = []
    for g in G:
        t = cv2.resize(G[g][1](im), (480, 270)); cv2.rectangle(t, (0, 0), (480, 30), (0, 0, 0), -1)
        cv2.putText(t, g, (8, 22), 0, 0.7, (255, 255, 255), 2); tiles.append(t)
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(OUT, "contoh_gangguan.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]),
                [cv2.IMWRITE_JPEG_QUALITY, 85])
    report(rc.LIQUID_THR)


def report(thr=0.5):
    X = [json.loads(l) for l in open(os.path.join(OUT, "hasil.jsonl"))]
    L = ["Uji ketahanan medan — model AKTIF (botol_ver9c, cap_belt_v2, cairan_RH, ukuran luas), profil nano. Ambang cairan %.2f." % thr,
         "Tutup benar = 'tutup' untuk on / 'tanpa_tutup' untuk off ('?' = salah). Cairan RH dinilai hanya berwarna vs kosong",
         "(air bening bukan tugas RH). Ukuran stabil = kelas ukuran sama dengan foto asli.", ""]
    A = [x for x in X if x["uji"] == "A"]
    base = {x["file"]: x for x in A if x["g"] == "asli"}
    L.append("== A. Foto belt 22 Okt (65 foto, 5 botol, tidak ikut dilatih) + gangguan buatan")
    L.append(f"{'gangguan':20s} {'botol terdeteksi':>16s} {'tutup benar':>12s} {'berwarna terdeteksi':>20s} {'kosong benar':>13s} {'ukuran stabil':>14s}")
    for g, (kel, _) in G.items():
        rr = [x for x in A if x["g"] == g]
        L.append(_row_A(g, rr, base, thr))
    L.append("")
    B = [x for x in X if x["uji"] == "B"]
    L.append("== B. Medan NYATA lain: foto studio Sep (bukan belt, kamera dekat, lampu lain), semua botol kosong")
    L.append(f"{'set / sudut':28s} {'n':>4s} {'botol terdeteksi':>16s} {'tutup benar':>12s} {'bertutup dikenali':>18s} {'tanpa tutup dikenali':>21s} {'kosong benar':>13s}")
    for key in sorted({(x["set"], x["view"]) for x in B}):
        rr = [x for x in B if (x["set"], x["view"]) == key]
        L.append(_row_B(" / ".join(key), rr, thr))
    L.append(_row_B("SEMUA studio", B, thr))
    L.append("")
    C = [x for x in X if x["uji"] == "C"]
    L.append("== C. Merek (model fold CV skema F; potongan tak dilatih) + gangguan pada potongan")
    L.append(f"{'gangguan':20s} {'Coca-Cola terdeteksi':>21s} {'non benar':>11s} {'salah terima':>13s}")
    for g in G_MEREK:
        rr = [x for x in C if x["g"] == g]; mt = [x for x in rr if x["kelas"] == "mitra"]; bk = [x for x in rr if x["kelas"] == "bukan"]
        L.append(f"{g:20s} {sum(x['p'] >= .5 for x in mt):>13d}/{len(mt):<7d} {sum(x['p'] < .5 for x in bk):>5d}/{len(bk):<5d} {sum(x['p'] >= .5 for x in bk):>8d}")
    open(os.path.join(OUT, "laporan.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


def _row_A(g, rr, base, thr):
    dd = [x for x in rr if x["det"]]
    cap_ok = sum((x["cap"] == "tutup") if x["cap_lab"] == "on" else (x["cap"] == "tanpa_tutup") for x in dd)
    wr = [x for x in dd if x["isi"] == "berwarna"]; ks = [x for x in dd if x["isi"] == "kosong"]
    nw = sum(x["isi"] == "berwarna" for x in rr); nk = sum(x["isi"] == "kosong" for x in rr)
    st = [x for x in dd if base.get(x["file"], {}).get("det")]
    stab = sum(x["size"] == base[x["file"]]["size"] for x in st)
    return (f"{g:20s} {len(dd):>9d}/{len(rr):<6d} {cap_ok:>6d}/{len(rr):<5d} {sum(x['p'] >= thr for x in wr):>11d}/{nw:<8d} "
            f"{sum(x['p'] < thr for x in ks):>6d}/{nk:<6d} {stab:>7d}/{len(st):<6d}")


def _row_B(name, rr, thr):
    dd = [x for x in rr if x["det"]]
    on = [x for x in rr if x["cap_lab"] == "on"]; off = [x for x in rr if x["cap_lab"] == "off"]
    on_ok = sum(x.get("cap") == "tutup" for x in on if x["det"]); off_ok = sum(x.get("cap") == "tanpa_tutup" for x in off if x["det"])
    return (f"{name:28s} {len(rr):>4d} {len(dd):>9d}/{len(rr):<6d} {on_ok + off_ok:>6d}/{len(rr):<5d} {on_ok:>10d}/{len(on):<7d} "
            f"{off_ok:>12d}/{len(off):<8d} {sum(x['p'] < thr for x in dd):>6d}/{len(rr):<6d}")


if __name__ == "__main__":
    main()
