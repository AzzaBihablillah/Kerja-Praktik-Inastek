"""Langkah 4c: metrik tanpa ambang (AUC) + metrik khusus produk WHITELIST (Coca-Cola, Fanta, Sprite Nipis Mint).

Alasan: ambang 0.5 menyembunyikan trade-off recall vs salah alarm. Dan di sistem, botol kosong merek lain sudah
ditolak oleh pemeriksaan merek, jadi salah alarm yang paling merugikan = botol KOSONG produk whitelist ditolak.
Whitelist W16: b01 Sprite Nipis Mint, b02 Coca-Cola, b05 Coca-Cola; b04 = botol berlabel Fanta (user menyebutnya
'fanta fake kuning' -> dilaporkan terpisah); b03 Frestea = bukan whitelist. P15: Coca-Cola & Fanta (Frestea bukan).
Ambang 'tw' = ambang terbaik pada fold LAIN (tanpa melihat botol uji): dipilih utk spesifisitas >= 0.95 pada negatif
latih-luar (C13 + W16 kosong dari fold lain), lalu diterapkan ke fold uji.
"""
import csv, json, os, re
from collections import defaultdict
import numpy as np

OUT = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"), "4_eksperimen", "cairan")
crows = {r["id"]: r for r in csv.DictReader(open(f"{OUT}/crops.csv"))}


def p15_product(path):
    m = re.search(r"/(warna|kosong|fanta|bening)/(?:foto|kosong)_(\d+)\.png", path.replace("\\", "/")); d, n = m.group(1), int(m.group(2))
    if d == "warna":
        return "frestea" if 21 <= n <= 31 else ("coca-cola" if 6 <= n <= 15 else "fanta")
    return {"kosong": "coca-cola", "fanta": "fanta", "bening": "lain"}[d]


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


WL_W16 = {"W16_b01", "W16_b02", "W16_b05"}
preds = [json.loads(l) for l in open(f"{OUT}/cls_preds.jsonl")]
lines = []
res = {}
for v in sorted({p["variant"] for p in preds}):
    P = [p for p in preds if p["variant"] == v]
    w = [p for p in P if p["src"] == "W16"]; c = [p for p in P if p["src"] == "C13"]
    p15 = defaultdict(list)
    for p in P:
        if p["src"] == "P15":
            p15[p["id"]].append(p["p_berwarna"])
    p15m = {i: float(np.mean(x)) for i, x in p15.items()}
    pos_w = [p["p_berwarna"] for p in w if p["liquid"] == "berwarna"]
    neg_w = [p["p_berwarna"] for p in w if p["liquid"] == "kosong"]
    neg_c = [p["p_berwarna"] for p in c]
    pos_p = [s for i, s in p15m.items() if crows[i]["liquid"] == "berwarna"]
    neg_p = [s for i, s in p15m.items() if crows[i]["liquid"] in ("kosong", "air_bening")]
    r = dict(AUC_W16=auc(pos_w, neg_w), AUC_W16pos_vs_C13=auc(pos_w, neg_c), AUC_P15=auc(pos_p, neg_p))
    # ambang dari fold lain (spesifisitas >= .95 di negatif fold lain)
    tw_ok = defaultdict(lambda: [0, 0])
    for k in sorted({p["fold"] for p in P}):
        negs = sorted(p["p_berwarna"] for p in P if p["src"] in ("W16", "C13") and p["fold"] != k and p["liquid"] == "kosong")
        t = negs[int(0.95 * (len(negs) - 1))] if negs else 0.5
        for p in P:
            if p["fold"] == k and p["src"] in ("W16", "C13") and p["liquid"] in ("berwarna", "kosong"):
                key = ("W16_" + p["liquid"]) if p["src"] == "W16" else "C13_kosong"
                tw_ok[key][0] += (p["p_berwarna"] > t) == (p["liquid"] == "berwarna"); tw_ok[key][1] += 1
    r["ambang_foldlain"] = {k: f"{a}/{n}" for k, (a, n) in tw_ok.items()}
    # whitelist @0.5
    wl = lambda cond: [p for p in w if p["group"] in WL_W16 and cond(p)]
    r["WL_W16_recall"] = f"{sum(p['p_berwarna'] >= .5 for p in wl(lambda p: p['liquid'] == 'berwarna'))}/{len(wl(lambda p: p['liquid'] == 'berwarna'))}"
    r["WL_W16_terima_kosong"] = f"{sum(p['p_berwarna'] < .5 for p in wl(lambda p: p['liquid'] == 'kosong'))}/{len(wl(lambda p: p['liquid'] == 'kosong'))}"
    b4 = [p for p in w if p["group"] == "W16_b04"]
    r["b04_labelFanta"] = f"recall {sum(p['p_berwarna'] >= .5 for p in b4 if p['liquid'] == 'berwarna')}/8, terima kosong {sum(p['p_berwarna'] < .5 for p in b4 if p['liquid'] == 'kosong')}/8"
    wp = [(i, s) for i, s in p15m.items() if p15_product(crows[i]["path"]) in ("coca-cola", "fanta")]
    r["WL_P15_recall"] = f"{sum(s >= .5 for i, s in wp if crows[i]['liquid'] == 'berwarna')}/{sum(1 for i, s in wp if crows[i]['liquid'] == 'berwarna')}"
    r["WL_P15_terima_kosong_atau_air"] = f"{sum(s < .5 for i, s in wp if crows[i]['liquid'] != 'berwarna')}/{sum(1 for i, s in wp if crows[i]['liquid'] != 'berwarna')}"
    res[v] = r
    lines.append(f"{v:22s} AUC W16 {r['AUC_W16']:.3f} | W16+ vs C13 {r['AUC_W16pos_vs_C13']:.3f} | P15 {r['AUC_P15']:.3f} || "
                 f"WHITELIST: W16 recall {r['WL_W16_recall']}, terima kosong {r['WL_W16_terima_kosong']}; P15 recall {r['WL_P15_recall']}, "
                 f"terima kosong/air {r['WL_P15_terima_kosong_atau_air']} || b04 {r['b04_labelFanta']} || ambang fold-lain {r['ambang_foldlain']}")
open(f"{OUT}/cls_auc.txt", "w").write("\n".join(lines) + "\n")
json.dump(res, open(f"{OUT}/cls_auc.json", "w"), indent=1)
print("\n".join(lines))
