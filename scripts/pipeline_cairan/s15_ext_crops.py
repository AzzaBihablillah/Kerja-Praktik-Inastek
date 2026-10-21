"""Langkah 15: saring foto Open Food Facts jadi crop botol tunggal utuh, lalu ubah ke domain belt:
  - saring dgn ver7: tepat 1 botol (conf>=0.35), kelas teratas 'bottle' conf>=0.5, kotak tinggi (h/w>=1.7) & mengisi >=55% tinggi gambar
    -> membuang kaleng, karton, kemasan multi-botol, botol kaca (ver7 menandainya non_bottle)
  - crop kotak, putar agar botol rebah (leher ke kanan), 'lipat' 224x224 (sama dgn data belt)
Label = jenis minuman (berwarna / tidak berwarna) -> HARUS divalidasi manusia (lembar periksa dibuat).
Keluaran: out/crops_fold/EXT_<code>.jpg, out/crops_ext.csv, out/ext_sheets/*.jpg"""
import csv, os, sys
import cv2, numpy as np
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); CD = f"{OUT}/crops_fold"; SD = f"{OUT}/ext_sheets"; os.makedirs(SD, exist_ok=True)
D = f"{ROOT}/5_data_luar/_dl_ext/off"
v7 = YOLO(os.path.join(ROOT, "..", "export_model", "bottle_best_ver7.pt"))
rows, keep, rej = [], [], {"tidak_1_botol": 0, "bukan_bottle": 0, "bentuk": 0}
for r in csv.DictReader(open(f"{D}/candidates.csv", encoding="utf-8")):
    f = f"{D}/images/{r['code']}.jpg"
    im = cv2.imread(f)
    if im is None:
        continue
    H, W = im.shape[:2]
    res = v7.predict(im, imgsz=640, conf=0.35, verbose=False)[0]
    bb = [b for b in res.boxes if int(b.cls) == 0]
    if len(res.boxes) != 1 or len(bb) != 1:
        rej["tidak_1_botol" if len(res.boxes) != 1 else "bukan_bottle"] += 1; continue
    b = bb[0]; x1, y1, x2, y2 = b.xyxy[0].tolist(); w, h = x2 - x1, y2 - y1
    if float(b.conf) < 0.5 or h / max(w, 1) < 1.7 or h < 0.55 * H:
        rej["bentuk"] += 1; continue
    cid = f"EXT_{r['code']}"
    cv2.imwrite(f"{CD}/{cid}.jpg", fold_crop(im, x1, y1, x2, y2, pad=0.03), [cv2.IMWRITE_JPEG_QUALITY, 95])
    rows.append(dict(id=cid, src="EXT", group=f"EXT_{r['code']}", product=(r["brands"] + " " + r["name"])[:60], liquid="berwarna" if r["label"] == "berwarna" else "kosong",
                     level="", cap="", path=os.path.relpath(f, ROOT), crop=f"crops/{cid}.jpg"))
with open(f"{OUT}/crops_ext.csv", "w", newline="", encoding="utf-8") as fh:
    w_ = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w_.writeheader(); w_.writerows(rows)
from collections import Counter
print("ditolak:", rej, "| dipakai:", Counter(r["liquid"] for r in rows))
for lab in ("berwarna", "kosong"):
    sel = [r for r in rows if r["liquid"] == lab]
    for s in range(0, min(len(sel), 480), 96):
        t = []
        for r in sel[s:s + 96]:
            c = cv2.resize(cv2.imread(f"{CD}/{r['id']}.jpg"), (160, 160)); cv2.putText(c, r["product"][:22], (2, 12), 0, 0.33, (0, 255, 255), 1); t.append(c)
        while len(t) % 12: t.append(np.zeros_like(t[0]))
        cv2.imwrite(f"{SD}/ext_{lab}_{s // 96:02d}.jpg", cv2.vconcat([cv2.hconcat(t[i:i + 12]) for i in range(0, len(t), 12)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
