"""Model merek langkah 3: potong setiap kotak dataset warisan brand2 (Downloads/brand2, format YOLO dari Label Studio)
dengan cara potong yang SAMA dengan model dasar (m02.brand_crop: sumbu panjang mendatar, letterbox 256). Belum dipakai latih:
semua potongan harus diaudit user dulu (halaman validasi) karena label lama diketahui keliru (~12% produk Coca-Cola = unknown).
  venv/Scripts/python.exe -I scripts/pipeline_merek/m03_warisan_crop.py C:/Users/ASUS/Downloads/brand2
-> datasets/4_eksperimen/merek/warisan/crops/W#####.jpg + warisan.csv (id, file, kelas_lama, w, h, md5 gambar)"""
import csv, hashlib, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m02_cv_dasar import ROOT, brand_crop

SRC = sys.argv[1]; OUT = f"{ROOT}/4_eksperimen/merek/warisan"; os.makedirs(f"{OUT}/crops", exist_ok=True)
CLS = [l.strip() for l in open(f"{SRC}/classes.txt") if l.strip()]
rows, n = [], 0
for fn in sorted(os.listdir(f"{SRC}/images")):
    lb = f"{SRC}/labels/{os.path.splitext(fn)[0]}.txt"
    if not os.path.exists(lb):
        continue
    raw = open(f"{SRC}/images/{fn}", "rb").read(); im = cv2.imdecode(__import__("numpy").frombuffer(raw, "uint8"), cv2.IMREAD_COLOR)
    if im is None:
        continue
    H, W = im.shape[:2]; md5 = hashlib.md5(raw).hexdigest()
    for line in open(lb):
        p = line.split()
        if len(p) != 5:
            continue
        c, xc, yc, w, h = int(p[0]), *map(float, p[1:])
        x1, y1, x2, y2 = (xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H
        if (x2 - x1) < 24 or (y2 - y1) < 24:
            continue
        i = f"W{n:05d}"; n += 1
        cv2.imwrite(f"{OUT}/crops/{i}.jpg", brand_crop(im, x1, y1, x2, y2), [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows.append(dict(id=i, file=fn, kelas_lama=CLS[c], w=round(x2 - x1), h=round(y2 - y1), md5=md5))
with open(f"{OUT}/warisan.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from collections import Counter
print(len(rows), "kotak dari", len({r["file"] for r in rows}), "gambar;", Counter(r["kelas_lama"] for r in rows),
      "; gambar duplikat (md5):", len(rows and {r['file'] for r in rows}) - len({r['md5'] for r in rows}))
