"""Model merek langkah 5: usulan label untuk audit (model-in-the-loop).
Latih model dasar pada SEMUA foto belt (label tervalidasi) -> skor p(coca_cola) tiap potongan warisan (dedup md5) & OFF.
Usulan: warisan  = coca_cola bila kelas_lama coca_cola ATAU p >= 0,5 ; OFF = dari kolom merek, ditandai 'beda' bila model tidak setuju.
Keluaran: datasets/4_eksperimen/merek/audit_items.csv (id, sumber, kelas_lama, usul, p, path) -> dipakai halaman validasi.
  venv/Scripts/python.exe scripts/pipeline_merek/m05_usul_audit.py"""
import csv, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m02_cv_dasar import OUT, CR, S, rows, train

pt = f"{OUT}/runs/dasar_semua/weights/last.pt"
if not os.path.exists(pt):
    R = [r for r in rows() if os.path.exists(f"{CR}/{r['id']}.jpg")]
    train([(f"{CR}/{r['id']}.jpg", r["kelas"], r["id"]) for r in R], f"{OUT}/tmp/semua", f"{OUT}/runs", "dasar_semua")
from ultralytics import YOLO
m = YOLO(pt); cm = [i for i, v in m.names.items() if v == "mitra"][0]

items = []
W = list(csv.DictReader(open(f"{OUT}/warisan/warisan.csv")))
first = {}
for r in W:
    first.setdefault(r["md5"], r["file"])
for r in W:
    if first[r["md5"]] == r["file"]:
        items.append(dict(id=r["id"], sumber="warisan", kelas_lama=r["kelas_lama"], path=f"{OUT}/warisan/crops/{r['id']}.jpg"))
for r in csv.DictReader(open(f"{OUT}/off/off.csv", encoding="utf-8")):
    items.append(dict(id=r["id"], sumber="off", kelas_lama=r["usul"] + " | " + r["brands"][:30], path=f"{OUT}/off/crops/{r['id']}.jpg"))
for i in range(0, len(items), 64):
    ch = items[i:i + 64]
    for it, p in zip(ch, m.predict([x["path"] for x in ch], imgsz=S, verbose=False)):
        it["p"] = round(float(p.probs.data[cm]), 3)
for it in items:
    if it["sumber"] == "warisan":
        it["usul"] = "coca_cola" if it["kelas_lama"] == "coca_cola" or it["p"] >= .5 else "non_coca_cola"
    else:
        it["usul"] = it["kelas_lama"].split(" | ")[0]
with open(f"{OUT}/audit_items.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["id", "sumber", "kelas_lama", "usul", "p", "path"]); w.writeheader(); w.writerows(items)
from collections import Counter
print(Counter((x["sumber"], x["usul"]) for x in items))
print("warisan non-coca lama -> diusulkan coca oleh model:", sum(x["sumber"] == "warisan" and x["kelas_lama"] != "coca_cola" and x["p"] >= .5 for x in items))
print("OFF model tidak setuju dgn merek:", sum(x["sumber"] == "off" and (x["p"] >= .5) != (x["usul"] == "coca_cola") for x in items))
