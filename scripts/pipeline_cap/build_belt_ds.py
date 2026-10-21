"""Bangun dataset training dari foto belt (cap_belt_v1/images) + label HASIL KOREKSI Roboflow (_rf_export).

Split PER BOTOL (b01..b30), bukan per foto -> satu botol tidak pernah muncul di dua split.
Pemilihan val/test sengaja mencakup warna tutup berbeda (hasil ukur warna dari box koreksi):
  test: b21 (PUTIH), b10 (biru, botol 1500), b28 (kuning, botol minyak), b13 (merah)
  val : b04 (biru gelap), b14 (tosca), b16 (oranye), b26 (hijau)
  train: 22 botol sisanya (termasuk b17 = satu-satunya tutup putih lain).
Gambar = salinan asli cap_belt_v1 (JPEG q95, 1920x1080); label = hasil koreksi (nama hash di-strip).
Output: datasets/2_dataset_cap/cap_belt_ds/{images,labels}/{train,val,test} + data.yaml + SPLIT.csv
"""
import csv, glob, os, re, shutil

DS = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets"
OUT = f"{DS}/2_dataset_cap/cap_belt_ds"
TEST = {21, 10, 28, 13}
VAL = {4, 14, 16, 26}

if os.path.exists(OUT):
    shutil.rmtree(OUT)
for s in ("train", "val", "test"):
    os.makedirs(f"{OUT}/images/{s}")
    os.makedirs(f"{OUT}/labels/{s}")

lab = {}
for p in glob.glob(f"{DS}/2_dataset_cap/_rf_export/train/labels/*.txt"):
    lab[re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", os.path.basename(p)[:-4])] = p

rows, cnt = [], {}
for ip in sorted(glob.glob(f"{DS}/2_dataset_cap/cap_belt_v1/images/*.jpg")):
    name = os.path.basename(ip)[:-4]
    b = int(name[1:3])
    split = "test" if b in TEST else "val" if b in VAL else "train"
    shutil.copy2(ip, f"{OUT}/images/{split}/{name}.jpg")
    shutil.copy2(lab[name], f"{OUT}/labels/{split}/{name}.txt")
    c = 0 if "_cap_" in name else 1
    rows.append([name, b, split, c])
    cnt[(split, c)] = cnt.get((split, c), 0) + 1

assert len(rows) == 180 and len(lab) == 180
with open(f"{OUT}/SPLIT.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "bottle", "split", "cls"]); w.writerows(rows)
open(f"{OUT}/data.yaml", "w").write(
    f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: cap\n  1: no_cap\n")
for s in ("train", "val", "test"):
    bs = sorted({r[1] for r in rows if r[2] == s})
    print(f"{s:5s}: {sum(1 for r in rows if r[2]==s):3d} foto | botol {len(bs)} | cap {cnt.get((s,0),0)} no_cap {cnt.get((s,1),0)}")
    print("       ", ["b%02d" % x for x in bs])
