"""Baseline warna sederhana: botol berisi cairan berwarna vs kosong, evaluasi leave-one-bottle-out.

Pakai LEVEL dari MANIFEST (penuh/setengah/seperempat = berisi, kosong = kosong). Cap dan arah leher
TIDAK dipakai (labelnya terbukti kacau di set 16 Okt).
Fitur: jarak warna (Lab) badan botol terhadap belt di sekitarnya + selisih saturasi. Stiker dilewati
dengan median per kolom. Tanpa deep learning; tujuannya mengukur apakah fitur sederhana cukup.

  ../venv/Scripts/python.exe liquid_baseline.py [--dir primary/2026-10-16_water_v1]
"""
import argparse, csv, json, os
import cv2, numpy as np
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="1_foto_mentah/2026-10-16_water_v1")
ap.add_argument("--bottle", default="../export_model/bottle_best_ver7.pt")
a = ap.parse_args()
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"))

rows = [r for r in csv.DictReader(open(os.path.join(a.dir, "MANIFEST.csv"))) if r["file"] != "SKIP"]
model = YOLO(a.bottle)
cache = os.path.join(a.dir, "_boxes.json")
boxes = json.load(open(cache)) if os.path.exists(cache) else {}
feats, miss = [], []
for r in rows:
    f = r["file"]
    im = cv2.imread(os.path.join(a.dir, f))
    if f not in boxes:
        res = model.predict(im, imgsz=640, conf=0.05, verbose=False)[0]
        bb = [b for b in res.boxes if int(b.cls) == 0]
        boxes[f] = [float(v) for v in max(bb, key=lambda b: float(b.conf)).xyxy[0]] if bb else None
    b = boxes[f]
    if b is None:
        miss.append(f); continue
    x1, y1, x2, y2 = [int(v) for v in b]
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    h = y2 - y1
    body = (slice(y1 + int(h * .30), y2 - int(h * .30)), slice(x1, x2))
    # belt di atas dan bawah botol, selebar badan
    bg = np.concatenate([lab[max(0, y1 - int(h * .5)):max(1, y1 - 8), x1:x2].reshape(-1, 3),
                         lab[min(im.shape[0] - 1, y2 + 8):y2 + int(h * .5), x1:x2].reshape(-1, 3)])
    bgm = np.median(bg, axis=0)
    bgs = np.median(np.concatenate([hsv[max(0, y1 - int(h * .5)):max(1, y1 - 8), x1:x2, 1].ravel(),
                                    hsv[min(im.shape[0] - 1, y2 + 8):y2 + int(h * .5), x1:x2, 1].ravel()]))
    colL = np.median(lab[body].reshape(lab[body].shape[0], -1, 3), axis=0)   # per kolom
    colS = np.median(hsv[body][..., 1], axis=0)
    # median per kolom lalu persentil antar kolom: stiker menempati minoritas kolom
    d = np.linalg.norm(colL - bgm, axis=1)
    feats.append(dict(file=f, bottle=int(r["bottle"]), level=r["level"], liquid=r["liquid"],
                      y=int(r["level"] != "kosong"),
                      dist_med=float(np.median(d)), dist_p25=float(np.percentile(d, 25)),
                      dsat_med=float(np.median(colS) - bgs), dsat_p25=float(np.percentile(colS, 25) - bgs)))
json.dump(boxes, open(cache, "w"))
print(f"foto {len(rows)}, bottle tidak terdeteksi: {len(miss)} {miss[:5]}")

keys = ["dist_med", "dist_p25", "dsat_med", "dsat_p25"]
X = np.array([[f[k] for k in keys] for f in feats]); y = np.array([f["y"] for f in feats])
g = np.array([f["bottle"] for f in feats])

def fit_threshold(Xtr, ytr, j):
    """ambang 1-fitur terbaik di train (akurasi seimbang), arah dicoba dua-duanya"""
    best = (-1, 0, 1)
    for t in np.unique(Xtr[:, j]):
        for s in (1, -1):
            p = (s * (Xtr[:, j] - t) > 0).astype(int)
            ba = (p[ytr == 1].mean() + (1 - p[ytr == 0]).mean()) / 2 if (ytr == 1).any() and (ytr == 0).any() else 0
            if ba > best[0]: best = (ba, t, s)
    return best[1], best[2]

print("\nLeave-one-bottle-out, 1 fitur + ambang (akurasi per foto):")
for j, k in enumerate(keys):
    ok = np.zeros(len(y), bool)
    for b in np.unique(g):
        tr, te = g != b, g == b
        t, s = fit_threshold(X[tr], y[tr], j)
        ok[te] = ((s * (X[te, j] - t) > 0).astype(int) == y[te])
    per = {int(b): f"{ok[g == b].mean():.2f}" for b in np.unique(g)}
    print(f"  {k:9s} {ok.mean():.3f}  per botol {per}")

print("\nRata-rata fitur per botol/level (untuk melihat pemisahan):")
for b in np.unique(g):
    s = []
    for lvl in ("penuh", "setengah", "seperempat", "kosong"):
        m = [f for f in feats if f["bottle"] == b and f["level"] == lvl]
        if m: s.append(f"{lvl[:6]} {np.mean([f['dist_med'] for f in m]):5.1f}/{np.mean([f['dsat_med'] for f in m]):5.1f}")
    print(f"  b{b:02d} {m[0]['liquid'][:12]:12s} (dist/dsat): " + "  ".join(s))
json.dump(feats, open("liquid_baseline_feats.json", "w"), indent=1)
