"""Langkah 6: data detektor untuk dua pendekatan, image set SAMA agar adil:
  A3 = 3 kelas: 0 botol (kosong/tidak berwarna), 1 botol_berwarna, 2 non_bottle
  B2 = 2 kelas (seperti ver7): 0 bottle, 1 non_bottle   -> 'ver8' = ver7 + foto belt
Sumber: data warisan bottle_detection (pemilahan visual Claude; gambar yg punya kotak 'ragu' DIBUANG dari kedua set)
        + foto belt W16 (tanpa seperempat) + C13. P15 tidak dilatih (uji lintas sesi).
Data warisan dibagi tetap: 85% latih / 15% 'old_test' (uji regresi: jangan sampai deteksi botol lama memburuk).
Gambar diperkecil (sisi terpanjang 960) agar cepat. Fold sama dengan classifier.
"""
import csv, hashlib, os, shutil
import cv2

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan")
DD = os.path.join(OUT, "det_data"); IMG = os.path.join(DD, "img960")
for d in (IMG, f"{DD}/A3/images", f"{DD}/A3/labels", f"{DD}/B2/images", f"{DD}/B2/labels", f"{DD}/lists"):
    os.makedirs(d, exist_ok=True)
BD = os.path.join(ROOT, "5_data_luar/_warisan", "bottle_detection_1280")


def save960(src, name):
    o = f"{IMG}/{name}.jpg"
    if not os.path.exists(o):
        im = cv2.imread(src); h, w = im.shape[:2]; s = 960 / max(h, w)
        if s < 1:
            im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(o, im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return o


relab = {(r["stem"], int(r["line"])): r["label"] for r in csv.DictReader(open(f"{OUT}/old_relabel_manual.csv"))}
recs = []                                                   # (name, A3 lines, B2 lines, split tag)
drop_ragu = 0
for f in sorted(os.listdir(f"{BD}/images")):
    stem = os.path.splitext(f)[0]
    L = [l.split() for l in open(f"{BD}/labels/{stem}.txt").read().splitlines() if l.strip()]
    a3, b2, ragu = [], [], False
    for k, l in enumerate(L):
        c = int(l[0]); box = " ".join(l[1:5])
        if c == 0:
            lab = relab[(stem, k)]
            ragu |= lab == "ragu"
            a3.append(f"{1 if lab == 'berwarna' else 0} {box}"); b2.append(f"0 {box}")
        else:
            a3.append(f"2 {box}"); b2.append(f"1 {box}")
    if ragu:
        drop_ragu += 1; continue
    tag = "old_test" if int(hashlib.md5(stem.encode()).hexdigest(), 16) % 100 < 15 else "old_train"
    recs.append((f"old_{stem}", f"{BD}/images/{stem}.jpg", a3, b2, tag))

for i, r in enumerate(csv.DictReader(open(f"{OUT}/belt_index.csv"))):
    W, H = float(r["W"]), float(r["H"]); x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
    box = f"{(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"
    liq = r["liquid"]
    a3c = 1 if liq == "berwarna" else 0
    if r["src"] == "W16":
        tag = f"W16_f{int(r['group'][-2:])}" + ("_q" if liq == "seperempat" else "")
    elif r["src"] == "C13":
        tag = f"C13_{r['group']}"
    else:
        tag = f"P15_{liq}"
    recs.append((f"{r['src']}_{i:04d}", os.path.join(ROOT, r["path"]), [f"{a3c} {box}"], [f"0 {box}"], tag))

for name, src, a3, b2, tag in recs:
    p = save960(src, name)
    for scheme, lines in (("A3", a3), ("B2", b2)):
        dst = f"{DD}/{scheme}/images/{name}.jpg"
        if not os.path.exists(dst):
            os.link(p, dst)
        open(f"{DD}/{scheme}/labels/{name}.txt", "w").write("\n".join(lines) + "\n")

c13_groups = sorted({t[1][4:] for t in [(0, r[4]) for r in recs] if t[1].startswith("C13_")}, key=lambda g: int(g.split("_b")[1]))
c13_fold = {g: (i % 5) + 1 for i, g in enumerate(c13_groups)}


def fold_of(tag):
    if tag.startswith("W16_f"):
        return int(tag[5])
    if tag.startswith("C13_"):
        return c13_fold[tag[4:]]
    return 0


for scheme in ("A3", "B2"):
    for k in range(1, 6):
        tr = [n for n, _, _, _, t in recs if (t == "old_train" or t.startswith("C13_") or (t.startswith("W16_f") and not t.endswith("_q")))
              and fold_of(t) != k]
        va = [n for n, _, _, _, t in recs if fold_of(t) == k and not t.endswith("_q")]
        for nm, lst in (("train", tr), ("val", va)):
            open(f"{DD}/lists/{scheme}_f{k}_{nm}.txt", "w").write("\n".join(f"{DD}/{scheme}/images/{n}.jpg" for n in lst) + "\n")
        names = ["botol", "botol_berwarna", "non_bottle"] if scheme == "A3" else ["bottle", "non_bottle"]
        open(f"{DD}/{scheme}_f{k}.yaml", "w").write(
            f"train: {DD}/lists/{scheme}_f{k}_train.txt\nval: {DD}/lists/{scheme}_f{k}_val.txt\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
with open(f"{DD}/det_index.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["name", "tag", "fold"]); [w.writerow([n, t, fold_of(t)]) for n, _, _, _, t in recs]
from collections import Counter
print("gambar warisan dibuang karena ada kotak ragu:", drop_ragu)
print(Counter(t.split("_")[0] if not t.startswith("old") else t for *_, t in recs))
print("contoh train f1:", sum(1 for _ in open(f"{DD}/lists/A3_f1_train.txt")), "val f1:", sum(1 for _ in open(f"{DD}/lists/A3_f1_val.txt")))
cnt = Counter()
for n, _, a3, _, t in recs:
    if t == "old_train":
        for l in a3:
            cnt[l.split()[0]] += 1
print("kelas A3 di old_train:", dict(cnt))
