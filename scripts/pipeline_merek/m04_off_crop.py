"""Model merek langkah 4: data luar Open Food Facts (datasets/5_data_luar/_dl_ext/off, sudah diunduh untuk eksperimen cairan).
Ambil merek Coca-Cola Company + merek PESAING YANG MIRIP (Pepsi vs Coca-Cola, 7up vs Sprite, Mirinda vs Fanta, Nestea/Lipton vs
Frestea/Fuze Tea, Dr Pepper). Kotak botol dicari dengan botol_ver9c (kaleng/kemasan tanpa botol terbuang otomatis), dipotong cara m02.
Label usulan dari kolom merek OFF = BELUM dipercaya; wajib diaudit user (halaman validasi) sebelum dipakai latih.
  venv/Scripts/python.exe -I scripts/pipeline_merek/m04_off_crop.py -> datasets/4_eksperimen/merek/off/crops + off.csv"""
import csv, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m02_cv_dasar import ROOT, brand_crop

SRC = f"{ROOT}/5_data_luar/_dl_ext/off"; OUT = f"{ROOT}/4_eksperimen/merek/off"; os.makedirs(f"{OUT}/crops", exist_ok=True)
KO = ("coca-cola", "coca cola", "fanta", "sprite", "fuze", "minute maid", "powerade", "the coca-cola company", "smartwater", "dasani")
MIRIP = ("pepsi", "7up", "7 up", "dr pepper", "dr. pepper", "mirinda", "nestea", "lipton", "ice tea")


def usul(brands):
    b = brands.lower()
    if any(k in b for k in KO):
        return "coca_cola"
    if any(k in b for k in MIRIP):
        return "non_coca_cola"
    return None


from ultralytics import YOLO
det = YOLO(f"{ROOT}/3_model_final/botol_ver9c.pt"); rows = []
for r in csv.DictReader(open(f"{SRC}/candidates.csv", encoding="utf-8", errors="replace")):
    u = usul(r["brands"])
    f = f"{SRC}/images/{r['code']}.jpg"
    if u is None or not os.path.exists(f):
        continue
    im = cv2.imread(f)
    if im is None:
        continue
    bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
    if not bb:
        continue
    i = f"O{len(rows):05d}"
    cv2.imwrite(f"{OUT}/crops/{i}.jpg", brand_crop(im, *max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))), [cv2.IMWRITE_JPEG_QUALITY, 92])
    rows.append(dict(id=i, code=r["code"], brands=r["brands"], name=r["name"], usul=u))
with open(f"{OUT}/off.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
from collections import Counter
print(len(rows), Counter(r["usul"] for r in rows))
