"""ver10 = resep ver9c + foto NYATA di belt 22 Okt: non-botol (plastik kresek, tangan, HP, gunting, dll.) sebagai gambar TANPA botol,
dan botol PET hari yang sama (sesi capture + frame diam aplikasi) agar 'foto hari ini' bukan tanda non-botol.
Tidak dipakai: hand-sanitizer & Listerine (botol PET non-minuman: menunggu keputusan mitra).
Kotak botol positif = prediksi botol_ver9c (conf >= 0,5, kotak terbesar) -> label otomatis, dicek lewat lembar contoh.
CV 5 fold per BENDA (plastik-bening-remas-2 dibagi 6 kantong menurut urutan foto, pengamatan visual 22 Okt).
Pembanding = botol_ver9c apa adanya (tidak pernah melihat foto ini). Foto nyata diulang x5 di daftar latih (bobot).
  python scripts/pipeline_botol/ver10_nyata.py          CV -> datasets/4_eksperimen/botol/ver10/cv.jsonl + laporan
  python scripts/pipeline_botol/ver10_nyata.py final    latih semua -> datasets/3_model_final/botol_ver10_kandidat.pt"""
import csv, glob, json, os, re, shutil, sys
from collections import defaultdict

DS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = f"{DS}/4_eksperimen/botol/ver10"; NY = f"{OUT}/nyata22"
NB = f"{DS}/1_foto_mentah/2026-10-22_nonbottle_v1"; WV = f"{DS}/1_foto_mentah/2026-10-22_water_v1"; SC = f"{DS}/1_foto_mentah/2026-10-22_scan_app"
V9C = f"{DS}/3_model_final/botol_ver9c.pt"; V7 = os.path.join(DS, "..", "export_model", "bottle_best_ver7.pt")
K, REP_NEG, REP_POS = 5, 5, 2
SKIP = ("hand-sanitizer", "listerin")
REMAS2 = [(1, 6), (7, 13), (14, 16), (17, 21), (22, 28), (29, 33)]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline_cairan"))


def group_of_neg(n):
    base, num = n.rsplit("_", 1); num = int(num[:-4])
    if base == "plastik-bening-remas-2":
        return f"remas2_{[i for i, (a, b) in enumerate(REMAS2) if a <= num <= b][0] + 1}"
    return base


def build():
    """Salin foto nyata ke nyata22/ (images + labels). Kembalikan daftar item (nama, kelompok, jenis)."""
    if os.path.exists(f"{NY}/items.json"):
        return json.load(open(f"{NY}/items.json"))
    import cv2
    from ultralytics import YOLO
    from s31_data_22okt import GROUP_V22, GROUP_SCAN
    os.makedirs(f"{NY}/images", exist_ok=True); os.makedirs(f"{NY}/labels", exist_ok=True)
    det = YOLO(V9C); items = []
    for f in sorted(glob.glob(f"{NB}/*.png")):
        n = os.path.basename(f)
        if n.startswith(SKIP):
            continue
        name = "nb_" + n; os.link(f, f"{NY}/images/{name}") if not os.path.exists(f"{NY}/images/{name}") else None
        open(f"{NY}/labels/{name[:-4]}.txt", "w").write("")
        items.append(dict(name=name, kel=group_of_neg(n), jenis="non_botol", kat=n.rsplit("_", 1)[0].rstrip("-0123456789")))
    pos = [(f, GROUP_V22[os.path.basename(f)[1:3]]) for f in sorted(glob.glob(f"{WV}/b*.png"))]
    for i, r in enumerate(csv.DictReader(open(f"{SC}/SCAN.csv")), 1):
        for k in (1, 2, 3):
            f = f"{SC}/s{r['sesi']}_b{r['botol']}_{r['waktu']}_{k}.png"
            if os.path.exists(f):
                pos.append((f, GROUP_SCAN[i]))
    for f, g in pos:
        im = cv2.imread(f); H, W = im.shape[:2]
        bb = [(float(b.conf), b.xyxy[0].tolist()) for b in det.predict(im, imgsz=640, conf=0.5, verbose=False)[0].boxes if int(b.cls) == 0]
        if not bb:
            print("botol tanpa kotak, dilewati:", f); continue
        x1, y1, x2, y2 = max(bb, key=lambda z: (z[1][2] - z[1][0]) * (z[1][3] - z[1][1]))[1]
        name = "pos_" + os.path.basename(os.path.dirname(f))[11:] + "_" + os.path.basename(f)
        if not os.path.exists(f"{NY}/images/{name}"):
            os.link(f, f"{NY}/images/{name}")
        open(f"{NY}/labels/{name[:-4]}.txt", "w").write(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        items.append(dict(name=name, kel=g, jenis="botol", kat="botol"))
    json.dump(items, open(f"{NY}/items.json", "w"), indent=0)
    return items


def folds(items):
    cnt = defaultdict(int)
    for it in items:
        cnt[it["kel"]] += 1
    F, load = {}, [0] * K
    for g in sorted(cnt, key=lambda g: -cnt[g]):
        k = min(range(K), key=lambda i: load[i]); F[g] = k; load[k] += cnt[g]
    return F, load


def train(extra, name):
    from ultralytics import YOLO
    base = [l.strip() for l in open(f"{DS}/4_eksperimen/botol/ver9c_train.txt") if l.strip()]
    tr = base + [f"{NY}/images/{it['name']}" for it in extra for _ in range(REP_NEG if it["jenis"] == "non_botol" else REP_POS)]
    open(f"{OUT}/{name}_train.txt", "w").write("\n".join(tr) + "\n")
    open(f"{OUT}/{name}.yaml", "w").write(f"train: {OUT}/{name}_train.txt\nval: {DS}/4_eksperimen/botol/old_test.txt\nnames:\n  0: bottle\n  1: non_bottle\n")
    YOLO(V7).train(data=f"{OUT}/{name}.yaml", epochs=10, imgsz=640, batch=16, workers=0, seed=0, deterministic=True, patience=0, plots=False,
                   verbose=False, project=OUT, name=name, exist_ok=True, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, degrees=0, fliplr=0.5,
                   flipud=0.0, mosaic=1.0, close_mosaic=3)
    return f"{OUT}/{name}/weights/last.pt"


def predict(pt, its):
    from ultralytics import YOLO
    m = YOLO(pt); out = []
    for it in its:
        r = m.predict(f"{NY}/images/{it['name']}", imgsz=640, conf=0.35, verbose=False)[0]
        out.append(any(int(b.cls) == 0 for b in r.boxes))
    return out


def report():
    R = [json.loads(l) for l in open(f"{OUT}/cv.jsonl") if l.strip()]
    L = ["ver10 CV 5 fold per benda, foto NYATA 22 Okt. Diterima = ada kotak bottle conf>=0,35. Pembanding ver9c (tak pernah melihat foto ini).", ""]
    kats = sorted({r["kat"] for r in R if r["jenis"] == "non_botol"})
    for m in ("ver9c", "ver10"):
        rr = [r for r in R if r["model"] == m]
        nb = [r for r in rr if r["jenis"] == "non_botol"]; bt = [r for r in rr if r["jenis"] == "botol"]
        L.append(f"{m:6s} non-botol diterima {sum(r['terima'] for r in nb)}/{len(nb)} | botol terdeteksi {sum(r['terima'] for r in bt)}/{len(bt)}")
        L.append("        " + ", ".join(f"{k} {sum(r['terima'] for r in nb if r['kat'] == k)}/{sum(1 for r in nb if r['kat'] == k)}" for k in kats))
    open(f"{OUT}/cv_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    os.makedirs(OUT, exist_ok=True)
    items = build(); F, load = folds(items)
    print(len(items), "foto nyata,", len({i["kel"] for i in items}), "kelompok; beban fold", load, flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        pt = train(items, "final"); shutil.copy2(pt, f"{DS}/3_model_final/botol_ver10_kandidat.pt")
        print("FINAL ->", f"{DS}/3_model_final/botol_ver10_kandidat.pt"); return
    res = f"{OUT}/cv.jsonl"
    done = {(json.loads(l)["fold"], json.loads(l)["model"]) for l in open(res)} if os.path.exists(res) else set()
    if (-1, "ver9c") not in done:
        with open(res, "a") as fh:
            for it, t in zip(items, predict(V9C, items)):
                fh.write(json.dumps(dict(fold=-1, model="ver9c", **it, terima=bool(t))) + "\n")
    for k in range(K):
        if (k, "ver10") in done:
            continue
        te = [i for i in items if F[i["kel"]] == k]
        pt = train([i for i in items if F[i["kel"]] != k], f"cv_f{k}")
        with open(res, "a") as fh:
            for it, t in zip(te, predict(pt, te)):
                fh.write(json.dumps(dict(fold=k, model="ver10", **it, terima=bool(t))) + "\n")
        print(f"fold {k}: uji {len(te)} selesai", flush=True)
    report()


if __name__ == "__main__":
    main()
