"""Langkah 8: bandingkan pipeline secara ujung-ke-ujung pada foto yang TIDAK ikut latih (fold sama):
  P0  = ver7 (sekarang)                              -> hanya deteksi botol
  PA  = detektor 3 kelas (A3_fk)                     -> botol + cairan sekaligus
  PB  = ver8 2 kelas (B2_fk) + classifier crop (fk)  -> dua tahap
  PB7 = ver7 + classifier crop (fk)                  -> dua tahap tanpa melatih ulang detektor
Aturan keputusan meniru server: kotak teratas conf>=0.35; 'botol terdeteksi' = kelas kotak teratas adalah kelas botol.
Uji regresi data warisan (old_test): recall kotak botol (IoU>=0.5) & salah terima non-botol sebagai botol.

  python s08_det_eval.py --cls V2_fold_belt [--folds 1,2,3,4,5]
Keluaran: out/det_eval_<cls>.json + .txt
"""
import argparse, csv, json, os
from collections import defaultdict
import sys
import cv2, numpy as np
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--cls", default="V2_fold_belt")
ap.add_argument("--folds", default="1,2,3,4,5")
a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); DD = f"{OUT}/det_data"
S = 224


from common import fold_crop  # noqa: E402


def top_box(model, im, bottle_cls):
    r = model.predict(im, imgsz=640, conf=0.05, verbose=False)[0]
    if not len(r.boxes):
        return None
    b = max(r.boxes, key=lambda z: float(z.conf))
    return dict(cls=int(b.cls), conf=float(b.conf), xyxy=b.xyxy[0].tolist(), bottle=int(b.cls) in bottle_cls and float(b.conf) >= 0.35)


def iou(a_, b_):
    ix = max(0, min(a_[2], b_[2]) - max(a_[0], b_[0])); iy = max(0, min(a_[3], b_[3]) - max(a_[1], b_[1]))
    inter = ix * iy; u = (a_[2] - a_[0]) * (a_[3] - a_[1]) + (b_[2] - b_[0]) * (b_[3] - b_[1]) - inter
    return inter / u if u > 0 else 0


idx = list(csv.DictReader(open(f"{DD}/det_index.csv")))
liq = {}
for i, r in enumerate(csv.DictReader(open(f"{OUT}/belt_index.csv"))):
    liq[f"{r['src']}_{i:04d}"] = r["liquid"]
v7 = YOLO(os.path.join(ROOT, "..", "export_model", "bottle_best_ver7.pt"))
recs = []                                                      # per foto belt per pipeline
old = defaultdict(lambda: defaultdict(int))
for k in [int(x) for x in a.folds.split(",")]:
    pa, pb = f"{OUT}/runs_det/A3_f{k}/weights/last.pt", f"{OUT}/runs_det/B2_f{k}/weights/last.pt"
    pc = f"{OUT}/runs_cls/{a.cls}_f{k}/weights/last.pt"
    if not all(os.path.exists(p) for p in (pa, pb, pc)):
        print("lewati fold", k, "(model belum ada)"); continue
    A, B, C = YOLO(pa), YOLO(pb), YOLO(pc)
    cb = [n for n, v in C.names.items() if v == "berwarna"][0]
    test = [r for r in idx if (int(r["fold"]) == k and (r["tag"].startswith("W16") or r["tag"].startswith("C13"))) or r["tag"].startswith("P15")]
    for r in test:
        im = cv2.imread(f"{DD}/A3/images/{r['name']}.jpg")
        t0, ta, tb = top_box(v7, im, {0}), top_box(A, im, {0, 1}), top_box(B, im, {0})
        def clsp(t):
            if not t or not t["bottle"]:
                return None
            return float(C.predict(fold_crop(im, *t["xyxy"]), imgsz=S, verbose=False)[0].probs.data[cb])
        row = dict(fold=k, name=r["name"], src=r["name"][:3], liquid=liq[r["name"]],
                   P0_det=bool(t0 and t0["bottle"]),
                   PA_det=bool(ta and ta["bottle"]), PA_berwarna=bool(ta and ta["bottle"] and ta["cls"] == 1),
                   PB_det=bool(tb and tb["bottle"]), PB_p=clsp(tb), PB7_p=clsp(t0))
        row["PB_berwarna"] = bool(row["PB_p"] is not None and row["PB_p"] >= .5)
        row["PB7_det"] = row["P0_det"]; row["PB7_berwarna"] = bool(row["PB7_p"] is not None and row["PB7_p"] >= .5)
        recs.append(row)
    # regresi data warisan
    for r in [x for x in idx if x["tag"] == "old_test"]:
        im = cv2.imread(f"{DD}/B2/images/{r['name']}.jpg"); H, W = im.shape[:2]
        gt = []
        for l in open(f"{DD}/B2/labels/{r['name']}.txt").read().split("\n"):
            if l.strip():
                c, cx, cy, bw, bh = l.split(); cx, cy, bw, bh = map(float, (cx, cy, bw, bh))
                gt.append((int(c), [(cx - bw / 2) * W, (cy - bh / 2) * H, (cx + bw / 2) * W, (cy + bh / 2) * H]))
        for nm, M, bc in (("P0", v7, {0}), ("PA", A, {0, 1}), ("PB", B, {0})):
            res = M.predict(im, imgsz=640, conf=0.35, verbose=False)[0]
            pr = [(int(b.cls) in bc, b.xyxy[0].tolist()) for b in res.boxes]
            for c, g in gt:
                hit_b = any(isb and iou(p, g) >= .5 for isb, p in pr)
                hit_n = any((not isb) and iou(p, g) >= .5 for isb, p in pr)
                if c == 0:
                    old[nm]["botol_n"] += 1; old[nm]["botol_terdeteksi"] += hit_b
                else:
                    old[nm]["nonbotol_n"] += 1; old[nm]["nonbotol_disebut_botol"] += hit_b; old[nm]["nonbotol_benar"] += hit_n

groups = [("W16", "berwarna"), ("W16", "kosong"), ("W16", "seperempat"), ("C13", "kosong"), ("P15", "berwarna"), ("P15", "kosong"), ("P15", "air_bening")]
out, lines = {"folds": sorted({r["fold"] for r in recs}), "cls": a.cls}, [f"Classifier: {a.cls}; fold: {sorted({r['fold'] for r in recs})}"]
lines.append("kelompok             n | deteksi botol P0  PA  PB | disebut berwarna PA  PB  PB7 | DITERIMA (botol & tak berwarna) PA PB PB7")
for s_, l_ in groups:
    R = [r for r in recs if r["src"] == s_ and r["liquid"] == l_]
    if not R:
        continue
    n = len(R); f = lambda key: sum(bool(r[key]) for r in R)
    acc = lambda p: sum(r[f"{p}_det"] and not r[f"{p}_berwarna"] for r in R)
    out[f"{s_}_{l_}"] = dict(n=n, det_P0=f("P0_det"), det_PA=f("PA_det"), det_PB=f("PB_det"), berwarna_PA=f("PA_berwarna"), berwarna_PB=f("PB_berwarna"),
                            berwarna_PB7=f("PB7_berwarna"), terima_PA=acc("PA"), terima_PB=acc("PB"), terima_PB7=acc("PB7"))
    o = out[f"{s_}_{l_}"]
    lines.append(f"{s_} {l_:11s} {n:4d} | {o['det_P0']:4d} {o['det_PA']:4d} {o['det_PB']:4d} | {o['berwarna_PA']:4d} {o['berwarna_PB']:4d} {o['berwarna_PB7']:4d} | "
                 f"{o['terima_PA']:4d} {o['terima_PB']:4d} {o['terima_PB7']:4d}")
pf = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for r in recs:
    if r["src"] == "W16" and r["liquid"] in ("berwarna", "kosong"):
        g = r["name"]
        for p in ("PA", "PB", "PB7"):
            ok = r[f"{p}_berwarna"] if r["liquid"] == "berwarna" else (r[f"{p}_det"] and not r[f"{p}_berwarna"])
            pf[p][(r["fold"], r["liquid"])][0] += ok; pf[p][(r["fold"], r["liquid"])][1] += 1
lines.append("W16 per botol uji (benar/n; berwarna = disebut berwarna, kosong = diterima):")
for p in ("PA", "PB", "PB7"):
    lines.append(f"  {p}: " + "  ".join(f"b0{k[0]} {k[1][:3]} {v[0]}/{v[1]}" for k, v in sorted(pf[p].items())))
lines.append("Regresi data warisan old_test (conf .35, IoU .5):")
for nm in ("P0", "PA", "PB"):
    o = old[nm]
    if o["botol_n"]:
        lines.append(f"  {nm}: botol terdeteksi {o['botol_terdeteksi']}/{o['botol_n']} = {o['botol_terdeteksi'] / o['botol_n']:.3f}; "
                     f"non-botol disebut botol {o['nonbotol_disebut_botol']}/{o['nonbotol_n']} = {o['nonbotol_disebut_botol'] / o['nonbotol_n']:.3f}; "
                     f"non-botol benar {o['nonbotol_benar']}/{o['nonbotol_n']}")
    out[f"old_{nm}"] = dict(o)
out["records"] = recs
json.dump(out, open(f"{OUT}/det_eval_{a.cls}.json", "w"), indent=1)
open(f"{OUT}/det_eval_{a.cls}.txt", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
