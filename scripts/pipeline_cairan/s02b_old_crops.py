"""Langkah 2b: crop 'fold' dari data warisan bottle_detection sesuai pemilahan visual Claude (berwarna/bukan; 'ragu' dibuang).
Dipakai varian classifier --old (menambah keragaman warna cairan: Calpico, teh, Mirinda, cola, dll.). Label BELUM diverifikasi manusia."""
import csv, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); BD = os.path.join(ROOT, "5_data_luar/_warisan", "bottle_detection_1280")
from common import fold_crop  # noqa: E402

CD = os.path.join(OUT, "crops_fold"); rows = []
for r in csv.DictReader(open(f"{OUT}/old_relabel_manual.csv")):
    if r["label"] == "ragu":
        continue
    stem = r["stem"]; line = int(r["line"])
    im = cv2.imread(f"{BD}/images/{stem}.jpg"); H, W = im.shape[:2]
    _, cx, cy, bw, bh = map(float, open(f"{BD}/labels/{stem}.txt").read().splitlines()[line].split()[:5])
    cid = f"OLD_{int(r['idx']):04d}"
    cv2.imwrite(f"{CD}/{cid}.jpg", fold_crop(im, (cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H), [cv2.IMWRITE_JPEG_QUALITY, 95])
    rows.append(dict(id=cid, src="OLD", group=f"OLD_{stem}", product="", liquid="berwarna" if r["label"] == "berwarna" else "kosong",
                     level="", cap="", path=f"5_data_luar/_warisan/bottle_detection_1280/images/{stem}.jpg", crop=f"crops/{cid}.jpg"))
with open(f"{OUT}/crops_old.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
from collections import Counter
print(Counter(r["liquid"] for r in rows))
