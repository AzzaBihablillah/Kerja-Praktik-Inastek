"""Langkah 13: varian pra-pemrosesan citra BELT (W16, C13, P15) sebelum crop:
  P1 = normalisasi warna belt: tiap kanal dikalikan (referensi global / median belt di atas+bawah kotak botol)
  P2 = CLAHE pada kanal L (clip 2.0, tile 8x8) di seluruh frame
Keluaran: out/crops_fold_P1/, out/crops_fold_P2/ (nama sama dgn crops_fold). Dipakai s03 dgn --crops crops_fold_P1 dst."""
import csv, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan")
rows = list(csv.DictReader(open(f"{OUT}/belt_index.csv")))
for d in ("crops_fold_P1", "crops_fold_P2"):
    os.makedirs(f"{OUT}/{d}", exist_ok=True)


def belt_ref(im, b):
    x1, y1, x2, y2 = [int(float(v)) for v in b]; h = y2 - y1
    top = im[max(0, y1 - int(h * .5)):max(1, y1 - 8), x1:x2].reshape(-1, 3); bot = im[min(im.shape[0] - 1, y2 + 8):y2 + int(h * .5), x1:x2].reshape(-1, 3)
    a = np.concatenate([top, bot]); return np.median(a, axis=0) if len(a) > 50 else None


refs = {}
for i, r in enumerate(rows):
    im = cv2.imread(os.path.join(ROOT, r["path"])); refs[i] = belt_ref(im, [r[k] for k in ("x1", "y1", "x2", "y2")])
glob_ref = np.median(np.array([v for v in refs.values() if v is not None]), axis=0); print("referensi belt global (BGR):", glob_ref.round(1))
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
for i, r in enumerate(rows):
    im = cv2.imread(os.path.join(ROOT, r["path"])); b = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
    cid = f"{r['src']}_{i:04d}"
    ref = refs[i] if refs[i] is not None else glob_ref
    p1 = np.clip(im.astype(np.float32) * (glob_ref / ref)[None, None, :], 0, 255).astype(np.uint8)
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB); lab[..., 0] = clahe.apply(lab[..., 0]); p2 = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    cv2.imwrite(f"{OUT}/crops_fold_P1/{cid}.jpg", fold_crop(p1, *b), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(f"{OUT}/crops_fold_P2/{cid}.jpg", fold_crop(p2, *b), [cv2.IMWRITE_JPEG_QUALITY, 95])
print(len(rows), "crop per varian")
