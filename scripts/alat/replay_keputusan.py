"""Simulasi ulang KEPUTUSAN MESIN (detektor -> tutup -> isi -> ukuran -> keputusan, sama dengan app.tracker.decide) pada frame
diam tersimpan sesi belt 27 & 29 Okt, untuk membandingkan set model AKTIF vs set model BARU secara jujur:
model baru untuk menilai suatu hari = model yang dilatih TANPA hari itu (hasil validasi per hari s37 / cap_v3 / m09).
Profil nano (seperti app), ambang isi 0,5, ukuran aturan 'panjang'. Keputusan dibandingkan dengan 'harapan' label kelompok.
  python scripts/alat/replay_keputusan.py  -> datasets/4_eksperimen/replay_keputusan/{hasil.jsonl, ringkas.txt}"""
import csv, glob, json, os, sys
from collections import Counter

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
DS = f"{REPO}/datasets"; M = f"{DS}/3_model_final"; OUT = f"{DS}/4_eksperimen/replay_keputusan"
AKTIF = dict(bottle=f"{M}/botol_ver9c.pt", cap=f"{M}/cap_belt_v2.pt", cap_roi=f"{M}/cap_belt_v2.pt", liquid=f"{M}/cairan_RH.pt")
BARU = {  # model per hari uji (tidak pernah melihat hari itu)
    "29": dict(bottle=f"{M}/botol_ver9c.pt", cap=f"{DS}/4_eksperimen/cap/_cap_v3/runs/cv_A29/weights/last.pt",
               cap_roi=f"{DS}/4_eksperimen/cap/_cap_v3/runs/cv_A29/weights/last.pt", liquid=f"{M}/cairan_berisi_v2_kandidat.pt"),
    "27": dict(bottle=f"{M}/botol_ver9c.pt", cap=f"{DS}/4_eksperimen/cap/_cap_v3/runs/cv_B27/weights/last.pt",
               cap_roi=f"{DS}/4_eksperimen/cap/_cap_v3/runs/cv_B27/weights/last.pt", liquid=f"{DS}/4_eksperimen/cairan/cv29/b27/weights/last.pt"),
}
VARIAN = {  # gabungan per komponen, agar terlihat sumber perbaikan
    "aktif": lambda h: AKTIF,
    "tutup_baru": lambda h: dict(AKTIF, cap=BARU[h]["cap"], cap_roi=BARU[h]["cap_roi"]),
    "isi_baru": lambda h: dict(AKTIF, liquid=BARU[h]["liquid"]),
    "semua_baru": lambda h: BARU[h],
    # 30 Okt: set AKTIF di aplikasi (rvm_kp MODEL_SETS, model FINAL yang sudah melihat 27 & 29 Okt) -> hanya untuk mencocokkan
    # aplikasi utuh (uji_app_putar) dengan simulasi ini, BUKAN ukuran akurasi
    "aktif_final": lambda h: __import__("rvm_kp.config", fromlist=["x"]).MODEL_SETS["pytorch"],
    # 30 Okt: model tutup v4w (latih + salinan warna tutup buatan yang diaudit user), isi v3 per hari
    "tutup_warna": lambda h: dict(BARU[h], cap=f"{DS}/4_eksperimen/cap/_cap_v4w/runs/cv_{'A29' if h == '29' else 'B27'}/weights/last.pt",
                                  cap_roi=f"{DS}/4_eksperimen/cap/_cap_v4w/runs/cv_{'A29' if h == '29' else 'B27'}/weights/last.pt"),
}


def sesi_rows():
    out = []
    for hari, pola in (("27", f"{DS}/1_foto_mentah/2026-10-27_scan_app/sesi[1-5]_*"), ("29", f"{DS}/1_foto_mentah/2026-10-29_scan_app/sesi_*")):
        for d in sorted(glob.glob(pola)):
            if not os.path.exists(d + "/SCAN.csv"):
                continue
            for r in csv.DictReader(open(d + "/SCAN.csv")):
                if not r.get("harapan") or r["alasan"] == "bukan_botol" or r.get("k_tutup") == "buang_audit":
                    continue
                fr = [f"{d}/s{r['sesi']}_b{r['botol']}_{r['waktu']}_{k}.png" for k in (1, 2, 3)]
                fr = [f for f in fr if os.path.exists(f)]
                if fr:
                    out.append(dict(hari=hari, sesi=r["sesi"], botol=r["botol"], isi=r["k_isi"], tutup=r["k_tutup"], harapan=r["harapan"], frames=fr))
    return out


def main():
    import cv2
    from rvm_kp import RVMAnalyzer
    from app.tracker import decide
    from app import config as ac
    os.makedirs(OUT, exist_ok=True); rows = sesi_rows(); res = f"{OUT}/hasil.jsonl"
    done = {(json.loads(l)["varian"], json.loads(l)["hari"]) for l in open(res)} if os.path.exists(res) else set()
    imgs = {}
    pilih = sys.argv[1:] or list(VARIAN)          # mis. `replay_keputusan.py aktif isi_baru`
    for var, fn in VARIAN.items():
        if var not in pilih:
            continue
        for hari in os.environ.get("REPLAY_HARI", "27,29").split(","):
            if (var, hari) in done:
                continue
            an = RVMAnalyzer("pytorch", models=fn(hari), profile="nano", size_rule="panjang")
            with open(res, "a") as fh:
                for r in [x for x in rows if x["hari"] == hari]:
                    votes = []
                    for f in r["frames"]:
                        im = imgs.get(f)
                        if im is None:
                            im = imgs[f] = cv2.imread(f)
                        rr = an.analyze(im)
                        if rr:
                            v = max(rr, key=lambda z: (z.box[2] - z.box[0]) * (z.box[3] - z.box[1]))
                            votes.append((v.cap, v.cap_conf, v.p_liquid, v.size, v.area_1920))

                    class _T:
                        pass
                    t = _T(); t.id = r["botol"]; t.votes = votes
                    k = decide(t, ac.LIQUID_THR if hasattr(ac, "LIQUID_THR") else 0.5) if votes else dict(decision="tidak_terdeteksi", reason="", p_liquid=0)
                    fh.write(json.dumps(dict(varian=var, hari=hari, sesi=r["sesi"], botol=r["botol"], isi=r["isi"], tutup=r["tutup"],
                                             harapan=r["harapan"], keputusan=k["decision"], alasan=k.get("reason") or "",
                                             p=k.get("p_liquid", 0))) + "\n")
            print("selesai", var, hari, flush=True)
            if len(imgs) > 2500:
                imgs.clear()
    R = [json.loads(l) for l in open(res)]
    L = ["Simulasi ulang keputusan mesin (frame diam tersimpan). Benar = keputusan == harapan label kelompok.", ""]
    for hari in ("27", "29"):
        L.append(f"== {hari} Okt")
        grp = sorted({(r["isi"], r["tutup"]) for r in R if r["hari"] == hari})
        for var in VARIAN:
            rr = [r for r in R if r["hari"] == hari and r["varian"] == var]
            if not rr:
                continue
            per = " | ".join(f"{i}/{t} {sum(r['keputusan'] == r['harapan'] for r in rr if (r['isi'], r['tutup']) == (i, t))}/"
                             f"{sum((r['isi'], r['tutup']) == (i, t) for r in rr)}" for i, t in grp)
            L.append(f"  {var:11s} benar {sum(r['keputusan'] == r['harapan'] for r in rr)}/{len(rr)} | salah terima "
                     f"{sum(r['keputusan'] == 'terima' and r['harapan'] == 'tolak' for r in rr)} | salah tolak "
                     f"{sum(r['keputusan'] != 'terima' and r['harapan'] == 'terima' for r in rr)}")
            L.append(f"              {per}")
        L.append("")
    open(f"{OUT}/ringkas.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
