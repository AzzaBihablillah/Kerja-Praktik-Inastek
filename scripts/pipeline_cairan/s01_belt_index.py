"""Langkah 1: indeks semua foto BELT berlabel + kotak botol + ukur recall model bottle ver7 (baseline).

Sumber (semua dari belt yang sama):
  W16 = primary/2026-10-16_water_v1  (5 botol, level isi dari manifest yang sudah diperbaiki)
  P15 = _warisan/foto_botol_update   (pilot 15 Okt; folder = label; 'fanta' & 'bening' = diisi AIR PUTIH)
  C13 = primary/2026-10-13_conveyor_v1 (30 botol KOSONG, cap/nocap)
Label cairan: berwarna | kosong | air_bening (di luar lingkup, tidak dilatih) | seperempat (ambigu, tidak dilatih)
Kotak: ver7 (conf>=0.05, kotak 'bottle' terbaik, lalu kotak apa pun) -> fallback COCO yolov8n kelas 39 'bottle'.
Keluaran: 4_eksperimen/cairan/belt_index.csv + lembar periksa kotak.
"""
import csv, glob, os, re
import cv2
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")          # datasets/
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); os.makedirs(OUT, exist_ok=True)
v7 = YOLO(os.path.join(ROOT, "..", "export_model", "bottle_best_ver7.pt"))
coco = YOLO(os.path.join(ROOT, "3_model_final/dasar/yolov8n.pt"))

items = []
for r in csv.DictReader(open(f"{ROOT}/1_foto_mentah/2026-10-16_water_v1/MANIFEST.csv")):
    if r["file"] in ("SKIP",) or r["level"] == "ref":
        continue
    lab = {"penuh": "berwarna", "setengah": "berwarna", "kosong": "kosong", "seperempat": "seperempat"}[r["level"]]
    items.append(dict(src="W16", path=f"1_foto_mentah/2026-10-16_water_v1/{r['file']}", group=f"W16_b{int(r['bottle']):02d}",
                      product=r["brand"], liquid=lab, level=r["level"], cap=r["cap"]))
for d, lab in (("warna", "berwarna"), ("kosong", "kosong"), ("fanta", "air_bening"), ("bening", "air_bening")):
    for f in sorted(glob.glob(f"{ROOT}/5_data_luar/_warisan/foto_botol_update/foto_botol/{d}/*.png")):
        items.append(dict(src="P15", path=os.path.relpath(f, ROOT).replace("\\", "/"), group=f"P15_{d}",
                          product="", liquid=lab, level=d, cap=""))
for f in sorted(glob.glob(f"{ROOT}/1_foto_mentah/2026-10-13_conveyor_v1/*.png")):
    m = re.match(r"(b\d+)_([a-z0-9-]+?)_(cap|nocap)_\d+", os.path.basename(f))
    items.append(dict(src="C13", path=os.path.relpath(f, ROOT).replace("\\", "/"), group=f"C13_{m.group(1)}",
                      product=m.group(2), liquid="kosong", level="kosong", cap="on" if m.group(3) == "cap" else "off"))

for it in items:
    im = cv2.imread(os.path.join(ROOT, it["path"]))
    H, W = im.shape[:2]
    r = v7.predict(im, imgsz=640, conf=0.05, verbose=False)[0]
    bb = sorted(r.boxes, key=lambda b: -float(b.conf))
    bot = [b for b in bb if int(b.cls) == 0]
    it["v7_top_cls"] = (["bottle", "non_bottle"][int(bb[0].cls)] if bb else "none")
    it["v7_top_conf"] = f"{float(bb[0].conf):.3f}" if bb else "0"
    it["v7_bottle_conf"] = f"{float(bot[0].conf):.3f}" if bot else "0"
    it["v7_detect_035"] = int(bool(bot) and float(bot[0].conf) >= 0.35 and int(bb[0].cls) == 0)   # sama dgn server (conf 0.35)
    box, how = None, ""
    if bot:
        box, how = bot[0].xyxy[0].tolist(), "v7_bottle"
    elif bb:
        box, how = bb[0].xyxy[0].tolist(), "v7_nonbottle"
    else:
        rc = coco.predict(im, imgsz=640, conf=0.05, classes=[39], verbose=False)[0]
        if len(rc.boxes):
            box, how = max(rc.boxes, key=lambda b: float(b.conf)).xyxy[0].tolist(), "coco"
    it["box_src"] = how or "none"
    it["x1"], it["y1"], it["x2"], it["y2"] = ([f"{v:.1f}" for v in box] if box else ["", "", "", ""])
    it["W"], it["H"] = W, H

keys = list(items[0].keys())
with open(f"{OUT}/belt_index.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(items)

# ringkasan recall ver7 (ambang server 0.35, kelas teratas harus bottle)
from collections import defaultdict
agg = defaultdict(lambda: [0, 0])
for it in items:
    k = (it["src"], it["liquid"]); agg[k][0] += it["v7_detect_035"]; agg[k][1] += 1
print("Recall model bottle ver7 (conf 0.35, kelas teratas=bottle):")
for k, (a, n) in sorted(agg.items()):
    print(f"  {k[0]} {k[1]:11s} {a}/{n} = {a / n:.2f}")
per = defaultdict(lambda: [0, 0])
for it in items:
    if it["src"] == "W16":
        k = (it["group"], it["level"]); per[k][0] += it["v7_detect_035"]; per[k][1] += 1
print("W16 per botol x level:")
for k, (a, n) in sorted(per.items()):
    print(f"  {k[0]} {k[1]:10s} {a}/{n}")
print("sumber kotak:", {s: sum(1 for i in items if i["box_src"] == s) for s in set(i["box_src"] for i in items)})

# lembar periksa kotak (urut per sumber)
tiles = []
for it in items:
    im = cv2.imread(os.path.join(ROOT, it["path"]))
    if it["x1"]:
        x1, y1, x2, y2 = [int(float(it[k])) for k in ("x1", "y1", "x2", "y2")]
        cv2.rectangle(im, (x1, y1), (x2, y2), {"v7_bottle": (0, 255, 0), "v7_nonbottle": (0, 0, 255), "coco": (255, 0, 255)}[it["box_src"]], max(2, im.shape[1] // 300))
    t = cv2.resize(im, (320, 180))
    cv2.putText(t, f"{it['group']} {it['liquid'][:6]} {it['box_src']}", (3, 14), 0, 0.42, (0, 255, 255), 1)
    tiles.append(t)
import numpy as np
for s in range(0, len(tiles), 48):
    ch = tiles[s:s + 48]
    while len(ch) % 6:
        ch.append(np.zeros_like(ch[0]))
    cv2.imwrite(f"{OUT}/boxsheet_{s // 48:02d}.jpg", cv2.vconcat([cv2.hconcat(ch[i:i + 6]) for i in range(0, len(ch), 6)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
print("lembar:", (len(tiles) + 47) // 48)
