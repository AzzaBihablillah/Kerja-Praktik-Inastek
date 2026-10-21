"""Ringkasan kurva belajar CNN dari out/curve_cnn.jsonl: rata-rata (dan rentang) per k botol berwarna latih."""
import json, os, collections
OUT = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"), "4_eksperimen", "cairan")
R = [json.loads(l) for l in open(f"{OUT}/curve_cnn.jsonl")]
by = collections.defaultdict(list)
for r in R:
    by[r["k"]].append(r)
f = lambda a: a[0] / a[1] if a[1] else float("nan")
lines = ["k_botol_latih | n_latihan | W16 recall botol uji (rata2 [min-maks]) | W16 spes | C13 spes | P15 recall | P15 spes"]
for k in sorted(by):
    g = by[k]
    rec = [f(r["w16_recall"]) for r in g]; sp = [f(r["w16_spec"]) for r in g]; c = [f(r["c13_spec"]) for r in g]; p = [f(r["p15_recall"]) for r in g]; ps = [f(r["p15_spec"]) for r in g]
    m = lambda x: sum(x) / len(x)
    lines.append(f"{k} | {len(g)} | {m(rec):.2f} [{min(rec):.2f}-{max(rec):.2f}] | {m(sp):.2f} | {m(c):.2f} | {m(p):.2f} [{min(p):.2f}-{max(p):.2f}] | {m(ps):.2f}")
lines.append("\nper botol uji (rata-rata recall W16 lintas k): " + ", ".join(f"b0{b}: " + " ".join(f"k{k}={sum(f(r['w16_recall']) for r in by[k] if r['test']==b)/max(1,sum(1 for r in by[k] if r['test']==b)):.2f}" for k in sorted(by)) for b in range(1, 6)))
open(f"{OUT}/curve_cnn_report.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
