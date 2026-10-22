"""Langkah 31: data cairan baru 22 Okt -> crop lipat + 3 salinan geser-hue (resep RH), siap digabung ke latih ulang.
Sumber:
  V22  : sesi capture 2026-10-22_water_v1 (5 botol berlabel penuh; b01 = b03 Mizone yang sama). Label dari nama file:
         penuh/setengah -> berwarna, kosong -> kosong. Cairan BENING (Mizone) DIBUANG dari latih (di luar lingkup, ambigu).
  S22  : frame DIAM aplikasi kiosk (stop-and-scan) 2026-10-22_scan_app/, label user di SCAN.csv kolom label_manual.
         'berisi air bening' DIBUANG; 'berisi ...' -> berwarna; 'kosong' -> kosong. 720p (resolusi kerja mesin).
Kelompok (untuk CV, botol fisik sama = satu kelompok; pengelompokan S22 dari pengamatan visual lembar cek):
  lihat GROUP_SCAN di bawah.
Kotak botol: botol_ver9c (detektor aktif). Keluaran: out/crops22.csv, out/crops_fold/{V22,S22}_*.jpg, out/crops_fold_hue/..."""
import csv, glob, os, random, re, sys
import cv2
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop
from s29_hueaug import shifted, K

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); HD = f"{OUT}/crops_fold_hue"
DV = f"{ROOT}/1_foto_mentah/2026-10-22_water_v1"; DS_ = f"{ROOT}/1_foto_mentah/2026-10-22_scan_app"
GROUP_V22 = {"01": "V22_mizone", "03": "V22_mizone", "02": "V22_minutemaid", "04": "V22_isoplus", "05": "V22_nutriboost"}
# nomor lembar cek (urutan baris SCAN.csv, 1..30) -> kelompok botol fisik
GROUP_SCAN = {1: "S22_fanta", 2: "S22_fanta", 3: "S22_fanta", 4: "S22_fanta", 5: "S22_fanta", 18: "S22_fanta",
              6: "S22_pocari", 10: "S22_pocari", 7: "S22_aqua_cola", 30: "S22_aqua_cola", 8: "S22_crystalin", 28: "S22_crystalin",
              9: "S22_leminerale", 12: "S22_leminerale", 13: "S22_leminerale", 14: "S22_leminerale2", 11: "S22_cokezero", 15: "S22_cokezero",
              16: "S22_cheers", 22: "S22_cheers", 23: "S22_cheers2", 17: "S22_botol17", 19: "S22_sprite", 25: "S22_sprite", 21: "S22_sprite_kecil",
              26: "S22_sprite_kecil", 20: "S22_fitri", 24: "S22_coke", 27: "S22_frestea", 29: "S22_aqua"}


def main():
    det = YOLO(f"{ROOT}/3_model_final/botol_ver9c.pt")
    items = []                                              # (path, liquid, group, src, note)
    for f in sorted(glob.glob(f"{DV}/b*.png")):
        t = os.path.basename(f).split("_")
        if "bening" in t[2] and t[3] != "kosong":              # air bening berisi dibuang; botol Mizone KOSONG tetap (negatif)
            continue
        items.append((f, "kosong" if t[3] == "kosong" else "berwarna", GROUP_V22[t[0][1:]], "V22", t[2]))
    for i, r in enumerate(csv.DictReader(open(f"{DS_}/SCAN.csv")), 1):
        lab = r["label_manual"]
        if not lab or "bening" in lab:
            continue
        liq = "kosong" if lab.startswith("kosong") else "berwarna"
        for k in (1, 2, 3):
            f = f"{DS_}/s{r['sesi']}_b{r['botol']}_{r['waktu']}_{k}.png"
            if os.path.exists(f):
                items.append((f, liq, GROUP_SCAN[i], "S22", lab))
    rows = []
    for path, liq, grp, src, note in items:
        im = cv2.imread(path); res = det.predict(im, imgsz=640, conf=0.35, verbose=False)[0]
        bb = [b.xyxy[0].tolist() for b in res.boxes if int(b.cls) == 0]
        if not bb:
            print("tanpa kotak:", os.path.basename(path)); continue
        box = max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        cid = f"{src}_{len(rows):04d}"
        cv2.imwrite(f"{OUT}/crops_fold/{cid}.jpg", fold_crop(im, *box), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rng = random.Random(cid)
        for k, dh in enumerate(rng.sample([30, 60, 90, 120, 150], K)):
            cv2.imwrite(f"{HD}/{cid}_h{k}.jpg", fold_crop(shifted(im, box, dh), *box), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(dict(id=cid, src=src, group=grp, product=note, liquid=liq, path=os.path.relpath(path, ROOT),
                         crop=f"crops/{cid}.jpg", x1=round(box[0]), y1=round(box[1]), x2=round(box[2]), y2=round(box[3])))
    with open(f"{OUT}/crops22.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    from collections import Counter
    print(len(rows), "crop;", Counter((r["src"], r["liquid"]) for r in rows), "; kelompok", len({r["group"] for r in rows}))


if __name__ == "__main__":
    main()
