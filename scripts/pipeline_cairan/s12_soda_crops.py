"""Langkah 12: crop botol dari dataset luar RF100 'soda-bottles' (Roboflow, CC BY 4.0; HF Francesco/soda-bottles).
Isi: foto lemari pendingin (kamera fisheye, botol penuh Coca-Cola/Fanta/Sprite, bertumpuk). SEMUA positif (cairan berwarna terlihat).
Pilih kotak yang cukup besar & tidak terlalu tumpang tindih (IoU<0.3 dgn kotak lain), sampel seimbang per kelas, seed tetap.
Keluaran: out/crops_fold/SODA_*.jpg + out/crops_soda.csv (src=SODA, liquid=berwarna)."""
import csv, json, os, random
import cv2
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); CD = os.path.join(OUT, "crops_fold")
P = os.path.join(ROOT, "5_data_luar/_dl_soda_bottles", "raw", "home", "zuppif", "Documents", "Work", "RoboFlow", "ODinW-RF100-challenge", "rf100", "soda-bottles")
N_PER_CLASS = 200
rng = random.Random(0)
cands = {1: [], 2: [], 3: []}


def iou(a, b):
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])); iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    i = ix * iy; return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


for split in ("train", "valid", "test"):
    d = json.load(open(f"{P}/{split}/_annotations.coco.json")); img = {i["id"]: i for i in d["images"]}
    by = {}
    for a in d["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    for iid, A in by.items():
        for a in A:
            x, y, w, h = a["bbox"]
            if min(w, h) < 70 or max(w, h) < 110 or a["category_id"] not in cands:
                continue
            if any(o is not a and iou(a["bbox"], o["bbox"]) > 0.3 for o in A):
                continue
            cands[a["category_id"]].append((f"{P}/{split}/{img[iid]['file_name']}", a["bbox"], iid))
rows = []
for c, lst in cands.items():
    print("kelas", c, "kandidat", len(lst))
    rng.shuffle(lst)
    for k, (f, b, iid) in enumerate(lst[:N_PER_CLASS]):
        im = cv2.imread(f); x, y, w, h = b
        cid = f"SODA_{c}_{k:03d}"
        cv2.imwrite(f"{CD}/{cid}.jpg", fold_crop(im, x, y, x + w, y + h), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(dict(id=cid, src="SODA", group=f"SODA_img{iid}", product={1: "coca-cola", 2: "fanta", 3: "sprite"}[c], liquid="berwarna",
                         level="penuh", cap="on", path=os.path.relpath(f, ROOT), crop=f"crops/{cid}.jpg"))
with open(f"{OUT}/crops_soda.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(len(rows), "crop SODA")
