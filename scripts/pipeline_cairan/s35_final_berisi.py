"""Langkah 35: model cairan KANDIDAT definisi baru (user 23 Okt): 'berisi apa pun' (berwarna ATAU air bening) vs 'kosong'.
Dasar keputusan: s34 (CV per kelompok): air bening belt sebagai berisi -> air bening 26/26 terdeteksi, cairan berwarna tidak turun
(122/124), kosong tidak memburuk; data luar TIDAK membantu pada definisi ini -> tidak dipakai.
Resep = s32 RN (W16 + V20 + V22 + S22, crop lipat standar + 3 salinan geser-hue) + air bening belt (V22 Mizone penuh/setengah,
S22 'berisi air bening') sebagai kelas 'berwarna' (artinya kini: berisi). Seimbang (ceil), 30 epoch, seed 0.
Keluaran: out/crops22b.csv (air bening, crop + hue), out/final5/cls5_berisi_all -> datasets/3_model_final/cairan_berisi_kandidat.pt
Belum dipakai rvm_kp: diuji dulu di sesi pembuktian (botol air bening TANPA tutup, merek lain)."""
import csv, glob, os, random, shutil, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop
from s29_hueaug import shifted, K
from s31_data_22okt import GROUP_SCAN
from s32_cv_rn import rows_all, items_of, train, OUT, ROOT

HD = f"{OUT}/crops_fold_hue"; DV = f"{ROOT}/1_foto_mentah/2026-10-22_water_v1"; DS_ = f"{ROOT}/1_foto_mentah/2026-10-22_scan_app"


def bening_crops():
    from ultralytics import YOLO
    det = YOLO(f"{ROOT}/3_model_final/botol_ver9c.pt"); src = []
    for f in sorted(glob.glob(f"{DV}/b*.png")):
        t = os.path.basename(f).split("_")
        if "bening" in t[2] and t[3] in ("penuh", "setengah"):
            src.append((f, "V22_mizone"))
    for i, r in enumerate(csv.DictReader(open(f"{DS_}/SCAN.csv")), 1):
        if "bening" in (r["label_manual"] or ""):
            for k in (1, 2, 3):
                f = f"{DS_}/s{r['sesi']}_b{r['botol']}_{r['waktu']}_{k}.png"
                if os.path.exists(f):
                    src.append((f, GROUP_SCAN[i]))
    rows = []
    for j, (f, g) in enumerate(src):
        im = cv2.imread(f)
        bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
        if not bb:
            continue
        box = max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])); cid = f"B22_{j:04d}"
        cv2.imwrite(f"{OUT}/crops_fold/{cid}.jpg", fold_crop(im, *box), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rng = random.Random(cid)
        for k, dh in enumerate(rng.sample([30, 60, 90, 120, 150], K)):
            cv2.imwrite(f"{HD}/{cid}_h{k}.jpg", fold_crop(shifted(im, box, dh), *box), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(dict(id=cid, src="B22", group=g, product="air bening", liquid="berwarna", path=os.path.relpath(f, ROOT), crop=f"crops/{cid}.jpg"))
    with open(f"{OUT}/crops22b.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


if __name__ == "__main__":
    rows = rows_all() + bening_crops()
    print(len(rows), "foto; air bening (berisi):", sum(r["src"] == "B22" for r in rows), flush=True)
    pt = train(items_of(rows), f"{OUT}/final5/data_berisi", f"{OUT}/final5", "cls5_berisi_all")
    shutil.copy2(pt, f"{ROOT}/3_model_final/cairan_berisi_kandidat.pt")
    print("FINAL ->", f"{ROOT}/3_model_final/cairan_berisi_kandidat.pt")
