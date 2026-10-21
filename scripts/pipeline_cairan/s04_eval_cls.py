"""Langkah 4: evaluasi classifier per varian + baseline M0 (fitur warna + regresi logistik) dengan fold yang SAMA.

Metrik (ambang 0.5):
  W16 (botol uji yang tidak ikut latih): recall berwarna (penuh+setengah), spesifisitas kosong, akurasi seimbang; per botol.
  C13 (botol kosong lain yang tidak ikut latih): spesifisitas, jumlah botol yang punya >=1 salah alarm.
  P15 (sesi lain, TIDAK pernah dilatih): rata-rata prob dari 5 model fold -> recall warna, spesifisitas kosong,
       dan % air bening yang disebut 'berwarna' (di luar lingkup; hanya perilaku).
  seperempat W16: % disebut berwarna (tidak dilatih; hanya perilaku).
Keluaran: out/cls_summary.txt + out/cls_summary.json
"""
import csv, json, math, os
from collections import defaultdict
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan")
crows = {r["id"]: r for r in csv.DictReader(open(f"{OUT}/crops.csv"))}
c13_groups = sorted({r["group"] for r in crows.values() if r["src"] == "C13"}, key=lambda g: int(g.split("_b")[1]))
c13_fold = {g: (i % 5) + 1 for i, g in enumerate(c13_groups)}


def fold_of(r):
    return int(r["group"][-2:]) if r["src"] == "W16" else (c13_fold[r["group"]] if r["src"] == "C13" else 0)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# ---- M0: fitur warna + logistik, fold sama, latih W16(non-1/4)+C13 ----
FK = ["f_dist_med", "f_dist_p25", "f_dsat_med", "f_dsat_p25", "f_dL", "f_da", "f_db"]


def fit(X, y, lam=1.0, it=600, lr=0.3):
    mu, sd = X.mean(0), X.std(0) + 1e-6; Z = (X - mu) / sd; w = np.zeros(Z.shape[1]); b = 0.0
    sw = np.where(y == 1, 0.5 / max(1, (y == 1).sum()), 0.5 / max(1, (y == 0).sum())) * len(y)
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Z @ w + b))); w -= lr * (Z.T @ (sw * (p - y)) / len(y) + lam * w / len(y)); b -= lr * (sw * (p - y)).mean()
    return lambda Xq: 1 / (1 + np.exp(-(((Xq - mu) / sd) @ w + b)))


preds = [json.loads(l) for l in open(f"{OUT}/cls_preds.jsonl")] if os.path.exists(f"{OUT}/cls_preds.jsonl") else []
belt = [r for r in crows.values() if r["src"] in ("W16", "C13", "P15")]
for fold in range(1, 6):
    tr = [r for r in belt if r["src"] in ("W16", "C13") and r["liquid"] in ("berwarna", "kosong") and fold_of(r) != fold]
    X = np.array([[float(r[k]) for k in FK] for r in tr]); y = np.array([int(r["liquid"] == "berwarna") for r in tr])
    f = fit(X, y)
    te = [r for r in belt if (r["src"] in ("W16", "C13") and fold_of(r) == fold) or r["src"] == "P15"]
    p = f(np.array([[float(r[k]) for k in FK] for r in te]))
    for r, pb in zip(te, p):
        preds.append(dict(variant="M0_warna_logistik", fold=fold, id=r["id"], src=r["src"], group=r["group"], liquid=r["liquid"], level=r["level"], p_berwarna=float(pb)))

summary, lines = {}, []
for v in sorted({p["variant"] for p in preds}):
    P = [p for p in preds if p["variant"] == v]
    folds = sorted({p["fold"] for p in P})
    s = {"folds": folds}
    w = [p for p in P if p["src"] == "W16"]
    pos = [p for p in w if p["liquid"] == "berwarna"]; neg = [p for p in w if p["liquid"] == "kosong"]; q = [p for p in w if p["liquid"] == "seperempat"]
    tp = sum(p["p_berwarna"] >= .5 for p in pos); tn = sum(p["p_berwarna"] < .5 for p in neg)
    rec = tp / max(1, len(pos)); spe = tn / max(1, len(neg))
    s["W16_recall"] = [tp, len(pos), round(rec, 3), [round(x, 3) for x in wilson(tp, len(pos))]]
    s["W16_spec"] = [tn, len(neg), round(spe, 3), [round(x, 3) for x in wilson(tn, len(neg))]]
    s["W16_bacc"] = round((rec + spe) / 2, 3)
    s["W16_per_bottle"] = {}
    for g in sorted({p["group"] for p in w}):
        gp = [p for p in pos if p["group"] == g]; gn = [p for p in neg if p["group"] == g]
        s["W16_per_bottle"][g] = dict(recall=f"{sum(p['p_berwarna'] >= .5 for p in gp)}/{len(gp)}", spec=f"{sum(p['p_berwarna'] < .5 for p in gn)}/{len(gn)}")
    s["W16_seperempat_as_berwarna"] = f"{sum(p['p_berwarna'] >= .5 for p in q)}/{len(q)}"
    c = [p for p in P if p["src"] == "C13"]
    s["C13_spec"] = [sum(p["p_berwarna"] < .5 for p in c), len(c), round(sum(p["p_berwarna"] < .5 for p in c) / max(1, len(c)), 3)]
    fp_groups = sorted({p["group"] for p in c if p["p_berwarna"] >= .5})
    s["C13_bottles_with_FP"] = fp_groups
    pp = defaultdict(list)
    for p in P:
        if p["src"] == "P15":
            pp[p["id"]].append(p["p_berwarna"])
    pm = {i: np.mean(v_) for i, v_ in pp.items()}
    def rate(lab, pos_=True):
        ids = [i for i in pm if crows[i]["liquid"] == lab]
        k = sum((pm[i] >= .5) == pos_ for i in ids); return [k, len(ids), round(k / max(1, len(ids)), 3)]
    s["P15_recall_warna"] = rate("berwarna", True); s["P15_spec_kosong"] = rate("kosong", False); s["P15_airbening_as_berwarna"] = rate("air_bening", True)
    summary[v] = s
    lines.append(f"\n=== {v}  (fold {folds})")
    lines.append(f"  W16 recall berwarna {s['W16_recall'][0]}/{s['W16_recall'][1]} = {s['W16_recall'][2]} CI{s['W16_recall'][3]}")
    lines.append(f"  W16 spesifisitas    {s['W16_spec'][0]}/{s['W16_spec'][1]} = {s['W16_spec'][2]} CI{s['W16_spec'][3]}   bacc {s['W16_bacc']}")
    lines.append("  per botol: " + "  ".join(f"{g[-3:]} r{d['recall']} s{d['spec']}" for g, d in s["W16_per_bottle"].items()))
    lines.append(f"  seperempat disebut berwarna: {s['W16_seperempat_as_berwarna']}")
    lines.append(f"  C13 spesifisitas {s['C13_spec'][0]}/{s['C13_spec'][1]} = {s['C13_spec'][2]}; botol dgn salah alarm: {fp_groups}")
    lines.append(f"  P15 (sesi lain): recall warna {s['P15_recall_warna']}, spes kosong {s['P15_spec_kosong']}, air bening->berwarna {s['P15_airbening_as_berwarna']}")
open(f"{OUT}/cls_summary.txt", "w").write("\n".join(lines) + "\n")
json.dump(summary, open(f"{OUT}/cls_summary.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
print("\n".join(lines))
