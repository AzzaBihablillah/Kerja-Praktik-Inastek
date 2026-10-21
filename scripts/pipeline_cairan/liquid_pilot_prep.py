"""Pilot deteksi isi cairan — langkah 1: salin data, indeks, potong botol.

Sumber (TIDAK diubah): C:/Users/ASUS/Downloads/foto_botol_update/foto_botol/{warna,bening,fanta,kosong}  (76 PNG 640x360)
Output: datasets/4_eksperimen/cairan_pilot/{raw,crops,index.csv,roi_check.jpg}

Kondisi (1 botol fisik diputar ~8-11 pose per kondisi; merek dari pengamatan visual, label isi dari user):
  C1 Coke/cola(keruh)   warna 6-15        C5 Fanta/air putih    fanta 94-103
  C2 Coke/air putih     bening 83-93      C6 Frestea/kuning     warna 21-31
  C3 Coke/kosong        kosong 1-9        C7 LeMinerale/air     bening 32-40
  C4 Fanta/merah(keruh) warna 0-5,16-20   C8 Prima/air          bening 78-82
Botol dipotong dgn model bottle_best_ver7 (leher kiri, dasar kanan). ROI dihitung relatif thd box botol.
"""
import csv, glob, os, re, shutil
import cv2, numpy as np
from ultralytics import YOLO

DS = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets"
SRC = "C:/Users/ASUS/Downloads/foto_botol_update/foto_botol"
OUT = f"{DS}/4_eksperimen/cairan_pilot"
BOTTLE = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/export_model/bottle_best_ver7.pt"


def cond_of(folder, n):
    if folder == "warna":
        if 6 <= n <= 15: return "C1", "coke", "cola_keruh"
        if n <= 5 or 16 <= n <= 20: return "C4", "fanta", "merah_keruh"
        return "C6", "frestea", "kuning_pucat"
    if folder == "bening":
        if 83 <= n <= 93: return "C2", "coke", "air_putih"
        if 78 <= n <= 82: return "C8", "prima", "air_putih"
        return "C7", "leminerale", "air_putih"
    if folder == "fanta": return "C5", "fanta", "air_putih"
    return "C3", "coke", "kosong"


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)
    os.makedirs(f"{OUT}/raw"); os.makedirs(f"{OUT}/crops")
    det = YOLO(BOTTLE)
    rows, ov = [], []
    for folder in ("warna", "bening", "fanta", "kosong"):
        os.makedirs(f"{OUT}/raw/{folder}")
        for p in sorted(glob.glob(f"{SRC}/{folder}/*.png")):
            name = os.path.basename(p)[:-4]
            n = int(re.sub(r"\D", "", name))
            shutil.copy2(p, f"{OUT}/raw/{folder}/{name}.png")
            cond, brand, fill = cond_of(folder, n)
            im = cv2.imread(p)
            r = det.predict(im, imgsz=640, conf=0.25, verbose=False)[0]
            bb = [b for b in r.boxes if int(b.cls) == 0]
            if bb:
                b = max(bb, key=lambda z: float(z.conf)); x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]; cf = float(b.conf)
            else:
                x1 = y1 = x2 = y2 = -1; cf = 0.0
            rows.append(dict(file=f"{folder}/{name}", folder=folder, n=n, cond=cond, brand=brand, fill=fill,
                             det=int(bool(bb)), conf=f"{cf:.3f}", x1=x1, y1=y1, x2=x2, y2=y2))
            if bb:
                cv2.imwrite(f"{OUT}/crops/{folder}__{name}.png", im[max(0, y1):y2, max(0, x1):x2])
                if len(ov) < 12 and (len(rows) % 6 == 1):
                    o = im.copy(); w, h = x2 - x1, y2 - y1
                    cv2.rectangle(o, (x1, y1), (x2, y2), (0, 255, 255), 1)
                    cv2.rectangle(o, (x1 + int(.64 * w), y1 + int(.22 * h)), (x1 + int(.97 * w), y1 + int(.78 * h)), (0, 0, 255), 1)   # ROI dasar
                    cv2.rectangle(o, (x1 + int(.30 * w), y1 + int(.25 * h)), (x1 + int(.62 * w), y1 + int(.75 * h)), (255, 128, 0), 1)  # ROI badan
                    ov.append(cv2.resize(o[60:300, :], (640, 240)))
    with open(f"{OUT}/index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    if ov:
        while len(ov) % 2: ov.append(np.zeros_like(ov[0]))
        cv2.imwrite(f"{OUT}/roi_check.jpg", cv2.vconcat([cv2.hconcat(ov[i:i + 2]) for i in range(0, len(ov), 2)]))
    det_ok = sum(r["det"] for r in rows)
    print(f"{len(rows)} foto | botol terdeteksi {det_ok}/{len(rows)}")
    from collections import Counter
    c = Counter((r["cond"], r["brand"], r["fill"]) for r in rows)
    for k, v in sorted(c.items()):
        print("  ", k, v)
    miss = [r["file"] for r in rows if not r["det"]]
    print("tanpa deteksi botol:", miss)


if __name__ == "__main__":
    main()
