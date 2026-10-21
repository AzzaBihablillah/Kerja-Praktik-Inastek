"""Ringkasan kurva belajar seimbang (s22c): per k -> AUC botol uji (berisi vs kosong botol uji + C13 fold), recall/spes @0.5,
AUC & recall P15; per botol uji. Keluaran: out/curve_cnn_bal_report.txt"""
import collections, json, os
import numpy as np

OUT = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"), "4_eksperimen", "cairan")


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


R = [json.loads(l) for l in open(f"{OUT}/curve_cnn_bal.jsonl")]
by = collections.defaultdict(list)
for r in R:
    pos = [p for l, p in r["w16"] if l == "berwarna"]; negw = [p for l, p in r["w16"] if l == "kosong"]; neg = negw + r["c13"]
    p15p = [p for l, p in r["p15"] if l == "berwarna"]; p15n = [p for l, p in r["p15"] if l != "berwarna"]
    by[r["k"]].append(dict(test=r["test"], auc=auc(pos, neg), rec=np.mean([p >= .5 for p in pos]), spw=np.mean([p < .5 for p in negw]),
                           spc=np.mean([p < .5 for p in r["c13"]]), auc15=auc(p15p, p15n), rec15=np.mean([p >= .5 for p in p15p]),
                           sp15=np.mean([p < .5 for p in p15n]), ratio=f"{r['latih_pos']}:{r['latih_neg']}"))
m = lambda xs: np.nanmean(xs)
L = ["k | n | rasio latih pos:neg (contoh) | AUC botol uji [min-maks] | recall@0.5 | spes kosong W16 | spes C13 | AUC P15 | recall P15 | spes P15"]
for k in sorted(by):
    g = by[k]
    L.append(f"{k} | {len(g)} | {g[0]['ratio']} | {m([x['auc'] for x in g]):.3f} [{min(x['auc'] for x in g):.2f}-{max(x['auc'] for x in g):.2f}] | "
             f"{m([x['rec'] for x in g]):.2f} | {m([x['spw'] for x in g]):.2f} | {m([x['spc'] for x in g]):.2f} | {m([x['auc15'] for x in g]):.3f} | "
             f"{m([x['rec15'] for x in g]):.2f} | {m([x['sp15'] for x in g]):.2f}")
L.append("\nper botol uji, recall@0.5 (k=1/2/3/4) | AUC (k=1/2/3/4):")
for b in range(1, 6):
    L.append(f"  b0{b}: " + " / ".join(f"{m([x['rec'] for x in by[k] if x['test'] == b]):.2f}" for k in sorted(by)) + "  |  " +
             " / ".join(f"{m([x['auc'] for x in by[k] if x['test'] == b]):.2f}" for k in sorted(by)))
open(f"{OUT}/curve_cnn_bal_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
