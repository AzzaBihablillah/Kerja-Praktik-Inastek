"""Langkah 2: potong kotak botol -> crop persegi (pad abu 114, 224 px) + fitur warna (baseline M0).

Masukan : out/belt_index.csv (langkah 1) + water_internet_fanta (Fanta penuh dari internet, label box YOLO)
Keluaran: out/crops/*.jpg, out/crops.csv (id, src, group, liquid, level, path_crop, fitur warna)
Fitur warna dihitung di gambar ASLI (butuh belt di atas/bawah botol sebagai pembanding); untuk foto internet = kosong.
"""
import csv, glob, os
import cv2, numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); CD = os.path.join(OUT, "crops"); os.makedirs(CD, exist_ok=True)
CDF = os.path.join(OUT, "crops_fold"); os.makedirs(CDF, exist_ok=True)
S = 224


def square_crop(im, x1, y1, x2, y2, pad=0.04, fold=False):
    H, W = im.shape[:2]; w, h = x2 - x1, y2 - y1
    x1, x2 = max(0, int(x1 - pad * w)), min(W, int(x2 + pad * w)); y1, y2 = max(0, int(y1 - pad * h)), min(H, int(y2 + pad * h))
    c = im[y1:y2, x1:x2]
    if c.shape[0] > c.shape[1]:                       # sumbu panjang botol dibuat horizontal
        c = cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)
    if fold:                                          # 'fold': potong dua & tumpuk -> rasio ~1.5:1, resolusi vertikal 2x
        hw = c.shape[1] // 2
        c = cv2.vconcat([c[:, :hw], c[:, hw:hw * 2]])
    ch, cw = c.shape[:2]; s = S / max(ch, cw)
    c = cv2.resize(c, (max(1, round(cw * s)), max(1, round(ch * s))), interpolation=cv2.INTER_AREA)
    out = np.full((S, S, 3), 114, np.uint8); oy, ox = (S - c.shape[0]) // 2, (S - c.shape[1]) // 2
    out[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return out


def color_feats(im, x1, y1, x2, y2):
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2); h = y2 - y1
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    top = (slice(max(0, y1 - int(h * .5)), max(1, y1 - 8)), slice(x1, x2)); bot = (slice(min(im.shape[0] - 1, y2 + 8), y2 + int(h * .5)), slice(x1, x2))
    bgL = np.concatenate([lab[top].reshape(-1, 3), lab[bot].reshape(-1, 3)])
    if len(bgL) < 50:
        return None
    bgm = np.median(bgL, axis=0)
    bgs = np.median(np.concatenate([hsv[top][..., 1].ravel(), hsv[bot][..., 1].ravel()]))
    body = (slice(y1 + int(h * .30), y2 - int(h * .30)), slice(x1, x2))
    colL = np.median(lab[body], axis=0); colS = np.median(hsv[body][..., 1], axis=0)
    diff = colL - bgm; d = np.linalg.norm(diff, axis=1)
    return [np.median(d), np.percentile(d, 25), np.median(colS) - bgs, np.percentile(colS, 25) - bgs, *np.median(diff, axis=0)]


FK = ["f_dist_med", "f_dist_p25", "f_dsat_med", "f_dsat_p25", "f_dL", "f_da", "f_db"]
rows = []
for i, r in enumerate(csv.DictReader(open(f"{OUT}/belt_index.csv"))):
    im = cv2.imread(os.path.join(ROOT, r["path"]))
    b = [float(r[k]) for k in ("x1", "y1", "x2", "y2")]
    cid = f"{r['src']}_{i:04d}"
    cv2.imwrite(f"{CD}/{cid}.jpg", square_crop(im, *b), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(f"{CDF}/{cid}.jpg", square_crop(im, *b, fold=True), [cv2.IMWRITE_JPEG_QUALITY, 95])
    f = color_feats(im, *b)
    rows.append(dict(id=cid, src=r["src"], group=r["group"], product=r["product"], liquid=r["liquid"], level=r["level"], cap=r["cap"],
                     path=r["path"], crop=f"crops/{cid}.jpg", **{k: (f"{v:.3f}" if f else "") for k, v in zip(FK, f or [0] * 7)}))

# Fanta penuh dari internet (Roboflow, CC BY 4.0): group = foto sumber (3 augmentasi per foto)
for j, f in enumerate(sorted(glob.glob(f"{ROOT}/5_data_luar/water_internet_fanta/images/*.jpg"))):
    im = cv2.imread(f); H, W = im.shape[:2]
    lab = open(f.replace("images", "labels").rsplit(".", 1)[0] + ".txt").read().split()
    if len(lab) < 5:
        continue
    _, cx, cy, bw, bh = map(float, lab[:5])
    b = [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]
    cid = f"WEB_{j:04d}"
    cv2.imwrite(f"{CD}/{cid}.jpg", square_crop(im, *b), [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(f"{CDF}/{cid}.jpg", square_crop(im, *b, fold=True), [cv2.IMWRITE_JPEG_QUALITY, 95])
    base = os.path.basename(f).split("_JPG")[0]
    rows.append(dict(id=cid, src="WEB", group=f"WEB_{base}", product="fanta", liquid="berwarna", level="penuh", cap="on",
                     path=os.path.relpath(f, ROOT), crop=f"crops/{cid}.jpg", **{k: "" for k in FK}))

with open(f"{OUT}/crops.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
from collections import Counter
print(Counter((r["src"], r["liquid"]) for r in rows))
# lembar contoh crop
sel = [r for r in rows if r["src"] != "WEB"][::6][:60] + [r for r in rows if r["src"] == "WEB"][:12:3]
t = [cv2.resize(cv2.imread(f"{OUT}/{r['crop']}"), (112, 112)) for r in sel]
while len(t) % 16: t.append(np.zeros_like(t[0]))
t2 = [cv2.resize(cv2.imread(f"{OUT}/{r['crop'].replace('crops/', 'crops_fold/')}"), (112, 112)) for r in sel]
while len(t2) % 16: t2.append(np.zeros_like(t2[0]))
cv2.imwrite(f"{OUT}/crop_samples_fold.jpg", cv2.vconcat([cv2.hconcat(t2[i:i + 16]) for i in range(0, len(t2), 16)]))
cv2.imwrite(f"{OUT}/crop_samples.jpg", cv2.vconcat([cv2.hconcat(t[i:i + 16]) for i in range(0, len(t), 16)]))
