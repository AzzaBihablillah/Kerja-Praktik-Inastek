"""Rincian P15 (sesi lain) per produk utk tiap varian: produk dibaca dari nomor foto (dicek visual di lembar d2_warna)."""
import csv, json, os, re, sys
from collections import defaultdict
import numpy as np
OUT = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"), "4_eksperimen", "cairan")
crows = {r["id"]: r for r in csv.DictReader(open(f"{OUT}/crops.csv"))}
def product(path):
    m = re.search(r"/(warna|kosong|fanta|bening)/(?:foto|kosong)_(\d+)\.png", path.replace("\\", "/")); d, n = m.group(1), int(m.group(2))
    if d == "warna":
        return "frestea" if 21 <= n <= 31 else ("coca-cola" if n in (6, 7, 8, 9, 10, 11, 12, 13, 14, 15) else "fanta")
    return {"kosong": "coca-cola kosong", "fanta": "fanta+air putih", "bening": "air putih (lemin/primo/coke)"}[d]
pp = defaultdict(lambda: defaultdict(list))
for l in open(f"{OUT}/cls_preds.jsonl"):
    p = json.loads(l)
    if p["src"] == "P15":
        pp[p["variant"]][p["id"]].append(p["p_berwarna"])
for v, d in sorted(pp.items()):
    agg = defaultdict(lambda: [0, 0])
    for i, ps in d.items():
        agg[product(crows[i]["path"])][0] += np.mean(ps) >= .5; agg[product(crows[i]["path"])][1] += 1
    print(v, " | ".join(f"{k}: disebut berwarna {a}/{n}" for k, (a, n) in sorted(agg.items())))
