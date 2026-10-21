"""Kurva belajar: berapa botol latih yang dibutuhkan agar 'berisi vs kosong' bisa ditebak pada botol BARU?

Fitur warna sederhana (selisih Lab + saturasi badan botol vs belt di sekitarnya, median per kolom spy stiker minoritas),
regresi logistik (numpy). Untuk k = 1..4 botol latih: SEMUA kombinasi botol latih, uji pada tiap botol sisa.
Dilaporkan: akurasi seimbang rata-rata, terburuk, dan per botol-uji. Dua varian: semua level, dan tanpa 'seperempat'
(level itu tidak konsisten di set 16 Okt).

  ../venv/Scripts/python.exe liquid_learning_curve.py [--dir primary/2026-10-16_water_v1]
Ini kurva untuk FITUR SEDERHANA; model CNN akan butuh lebih banyak botol, jadi ini batas bawah kebutuhan, bukan perkiraan CNN.
"""
import argparse, csv, itertools, json, os
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="1_foto_mentah/2026-10-16_water_v1")
a = ap.parse_args()
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"))

rows = [r for r in csv.DictReader(open(os.path.join(a.dir, "MANIFEST.csv"))) if r["file"] != "SKIP" and r["level"] != "ref"]
cache = os.path.join(a.dir, "_boxes.json")
boxes = json.load(open(cache)) if os.path.exists(cache) else {}
need = [r for r in rows if r["file"] not in boxes]
if need:
    from ultralytics import YOLO
    m = YOLO("../export_model/bottle_best_ver7.pt")
    for r in need:
        res = m.predict(cv2.imread(os.path.join(a.dir, r["file"])), imgsz=640, conf=0.05, verbose=False)[0]
        bb = [b for b in res.boxes if int(b.cls) == 0]
        boxes[r["file"]] = [float(v) for v in max(bb, key=lambda b: float(b.conf)).xyxy[0]] if bb else None
    json.dump(boxes, open(cache, "w"))

X, y, g, lvl = [], [], [], []
for r in rows:
    b = boxes[r["file"]]
    if b is None:
        continue
    im = cv2.imread(os.path.join(a.dir, r["file"]))
    x1, y1, x2, y2 = [int(v) for v in b]; h = y2 - y1
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32); hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV).astype(np.float32)
    top = (slice(max(0, y1 - int(h * .5)), max(1, y1 - 8)), slice(x1, x2)); bot = (slice(min(im.shape[0] - 1, y2 + 8), y2 + int(h * .5)), slice(x1, x2))
    bg = np.concatenate([lab[top].reshape(-1, 3), lab[bot].reshape(-1, 3)]); bgm = np.median(bg, axis=0)
    bgs = np.median(np.concatenate([hsv[top][..., 1].ravel(), hsv[bot][..., 1].ravel()]))
    body = (slice(y1 + int(h * .30), y2 - int(h * .30)), slice(x1, x2))
    colL = np.median(lab[body], axis=0); colS = np.median(hsv[body][..., 1], axis=0)       # per kolom
    diff = colL - bgm; d = np.linalg.norm(diff, axis=1)
    X.append([np.median(d), np.percentile(d, 25), np.median(colS) - bgs, np.percentile(colS, 25) - bgs, *np.median(diff, axis=0)])
    y.append(int(r["level"] != "kosong")); g.append(int(r["bottle"])); lvl.append(r["level"])
X, y, g, lvl = np.array(X, float), np.array(y), np.array(g), np.array(lvl)
print(f"{len(y)} foto dipakai (botol tak terdeteksi dibuang), botol {sorted(set(g.tolist()))}, berisi={int(y.sum())} kosong={int((1-y).sum())}")


def fit(Xtr, ytr, lam=1.0, it=400, lr=0.3):
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Z = (Xtr - mu) / sd; w = np.zeros(Z.shape[1]); bias = 0.0
    sw = np.where(ytr == 1, 0.5 / max(1, (ytr == 1).sum()), 0.5 / max(1, (ytr == 0).sum())) * len(ytr)   # seimbangkan kelas
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Z @ w + bias)))
        gr = Z.T @ (sw * (p - ytr)) / len(ytr) + lam * w / len(ytr)
        w -= lr * gr; bias -= lr * (sw * (p - ytr)).mean()
    return mu, sd, w, bias


def bacc(model, Xte, yte):
    mu, sd, w, b = model
    p = ((Xte - mu) / sd @ w + b > 0).astype(int)
    tpr = (p[yte == 1] == 1).mean() if (yte == 1).any() else np.nan
    tnr = (p[yte == 0] == 0).mean() if (yte == 0).any() else np.nan
    return np.nanmean([tpr, tnr]), (p == yte).mean()


def curve(mask, name):
    Xm, ym, gm = X[mask], y[mask], g[mask]
    bottles = sorted(set(gm.tolist()))
    print(f"\n== {name}  ({len(ym)} foto) ==")
    print("k_latih  n_kombinasi  bacc_rata2  bacc_terburuk   akurasi_rata2")
    for k in range(1, len(bottles)):
        accs, plain, worst_by_test = [], [], {}
        for tr in itertools.combinations(bottles, k):
            mtr = np.isin(gm, tr)
            if len(set(ym[mtr].tolist())) < 2:
                continue
            mod = fit(Xm[mtr], ym[mtr])
            for tb in bottles:
                if tb in tr:
                    continue
                mt = gm == tb
                ba, ac = bacc(mod, Xm[mt], ym[mt])
                accs.append(ba); plain.append(ac); worst_by_test.setdefault(tb, []).append(ba)
        print(f"  {k}        {len(accs):4d}        {np.mean(accs):.3f}       {np.min(accs):.3f}          {np.mean(plain):.3f}")
    k = len(bottles) - 1
    per = {}
    for tb in bottles:
        tr = [b for b in bottles if b != tb]
        mod = fit(Xm[np.isin(gm, tr)], ym[np.isin(gm, tr)]); per[tb] = bacc(mod, Xm[gm == tb], ym[gm == tb])[0]
    print("  per botol-uji (latih di sisanya):", {f"b{b:02d}": round(v, 2) for b, v in per.items()})


curve(np.ones(len(y), bool), "semua level (penuh/setengah/seperempat vs kosong)")
curve(lvl != "seperempat", "tanpa seperempat (penuh/setengah vs kosong)")
