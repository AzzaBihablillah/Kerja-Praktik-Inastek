"""Audit satu folder set cairan: bandingkan label tutup/leher di MANIFEST dengan tebakan model.
Hanya MELAPORKAN, tidak mengubah file.

  ../venv/Scripts/python.exe qc_water_set.py [--dir primary/2026-10-16_water_v1]
Keluaran: ringkasan di layar + <dir>/QC-REPORT.csv
"""
import argparse, csv, os, collections
import cv2
import water_qc

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="1_foto_mentah/2026-10-16_water_v1")
a = ap.parse_args()
os.chdir(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"))

rows = [r for r in csv.DictReader(open(os.path.join(a.dir, "MANIFEST.csv"))) if r["file"] not in ("SKIP",) and r["level"] != "ref"]
rep, stat = [], collections.Counter()
per = collections.defaultdict(lambda: [0, 0])
for r in rows:
    res = water_qc.analyze(cv2.imread(os.path.join(a.dir, r["file"])))
    w, lvl = water_qc.warnings(res, r["cap"])
    stat["foto"] += 1
    stat["botol_tak_terdeteksi"] += res["bottle_box"] is None
    if res["cap_state"] is None:
        stat["cap_tak_terlihat"] += 1
    else:
        ok = res["cap_state"] == r["cap"]
        stat["cap_cocok" if ok else "cap_beda"] += 1
        per[(r["bottle"], r["level"], r["cap"])][0] += ok
        per[(r["bottle"], r["level"], r["cap"])][1] += 1
    rep.append([r["file"], r["bottle"], r["level"], r["cap"], res["cap_state"], f"{res['cap_conf']:.2f}", res["neck_auto"], lvl, " | ".join(w)])
with open(os.path.join(a.dir, "QC-REPORT.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "bottle", "level", "cap_label", "cap_model", "cap_conf", "neck_auto", "level_peringatan", "peringatan"]); w.writerows(rep)
print(dict(stat))
print("\nKelompok (botol, level, tutup-label) dengan kecocokan model < 100%:")
for k, (ok, n) in sorted(per.items()):
    if ok < n:
        print(f"  b{int(k[0]):02d} {k[1]:10s} cap{k[2]:3s}  model setuju {ok}/{n}")
print("\nlaporan:", os.path.join(a.dir, "QC-REPORT.csv"))
