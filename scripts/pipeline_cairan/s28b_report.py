"""Ringkasan CV per KELOMPOK CAIRAN (s28): AUC gabungan, hasil @0,5 dan @ambang TERKALIBRASI.
Kalibrasi tanpa mengintip botol uji: untuk botol g, ambang = ambang yang memaksimalkan akurasi seimbang pada prediksi
out-of-fold SEMUA botol lain (berisi vs kosong), lalu diterapkan ke g. C13 dan P15 memakai ambang median.
  python s27b_report.py RA [RB ...]"""
import json, os, sys
import numpy as np

OUT = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"), "4_eksperimen", "cairan")


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def best_thr(pos, neg):
    cands = np.unique(np.concatenate([pos, neg, [0.5]]))
    sc = [((pos >= t).mean() + (neg < t).mean()) / 2 for t in cands]
    return float(cands[int(np.argmax(sc))])


for rec in sys.argv[1:]:
    R = [json.loads(l) for l in open(f"{OUT}/cv_lg_{rec}.jsonl")]
    oof = {r["test"]: [(lab, lvl, p) for (_, lab, lvl, _g), p in zip(r["test_rows"], r["test_p"])] for r in R}
    bott = {r["test"]: sorted({t[3] for t in r["test_rows"]}) for r in R}
    allpos = [p for g in oof for lab, lvl, p in oof[g] if lab == "berwarna"]; allneg = [p for g in oof for lab, lvl, p in oof[g] if lab == "kosong" and lvl == "kosong"]
    L = [f"\n===== {rec}: {len(R)} kelompok cairan | AUC gabungan (berisi vs kosong, out-of-fold) {auc(allpos, allneg):.3f}"]
    thrs = []
    tot = {"p5": [0, 0], "n5": [0, 0], "pc": [0, 0], "nc": [0, 0]}
    for g in sorted(oof):
        others_pos = np.array([p for h in oof if h != g for lab, lvl, p in oof[h] if lab == "berwarna"])
        others_neg = np.array([p for h in oof if h != g for lab, lvl, p in oof[h] if lab == "kosong" and lvl == "kosong"])
        t = best_thr(others_pos, others_neg); thrs.append(t)
        pos = [p for lab, lvl, p in oof[g] if lab == "berwarna"]; neg = [p for lab, lvl, p in oof[g] if lab == "kosong" and lvl == "kosong"]
        q = [p for lab, lvl, p in oof[g] if lvl == "seperempat"]
        for key, arr, thr, want in (("p5", pos, .5, 1), ("n5", neg, .5, 0), ("pc", pos, t, 1), ("nc", neg, t, 0)):
            tot[key][0] += sum((p >= thr) == bool(want) for p in arr); tot[key][1] += len(arr)
        L.append(f"  {g:13s} {','.join(x[-3:] for x in bott[g]):12s} ambang {t:.2f} | berisi @0.5 {sum(p >= .5 for p in pos)}/{len(pos)}  @kalib {sum(p >= t for p in pos)}/{len(pos)}"
                 + (f" | kosong @0.5 {sum(p < .5 for p in neg)}/{len(neg)}  @kalib {sum(p < t for p in neg)}/{len(neg)}" if neg else "")
                 + (f" | seperempat->berwarna @kalib {sum(p >= t for p in q)}/{len(q)}" if q else ""))
    L.append(f"  TOTAL berisi @0.5 {tot['p5'][0]}/{tot['p5'][1]}, @kalib {tot['pc'][0]}/{tot['pc'][1]} | kosong @0.5 {tot['n5'][0]}/{tot['n5'][1]}, @kalib {tot['nc'][0]}/{tot['nc'][1]}")
    tm = float(np.median(thrs))
    c13 = {}
    for r in R:
        for grp, p in r["c13"]:
            c13.setdefault(grp, []).append(p)
    c13p = [np.mean(v) for v in c13.values() for _ in [0]]
    c13all = [p for r in R for _, p in r["c13"]]
    L.append(f"  C13 (botol kosong merek lain, 13 Okt) diterima: @0.5 {np.mean([p < .5 for p in c13all]):.2f}, @ambang median {tm:.2f}: {np.mean([p < tm for p in c13all]):.2f}"
             f" | botol dgn salah alarm @median: {sorted({g for r in R for g, p in r['c13'] if p >= tm})}")
    p15 = {}
    for r in R:
        for i, (lab, p) in enumerate(r["p15"]):
            p15.setdefault(i, [lab, []])[1].append(p)
    pp = [(lab, np.mean(v)) for lab, v in p15.values()]
    L.append(f"  P15 (sesi pilot): AUC {auc([p for l, p in pp if l == 'berwarna'], [p for l, p in pp if l != 'berwarna']):.3f} | berisi @median {sum(p >= tm for l, p in pp if l == 'berwarna')}/{sum(l == 'berwarna' for l, _ in pp)}"
             f" | kosong+air diterima {sum(p < tm for l, p in pp if l != 'berwarna')}/{sum(l != 'berwarna' for l, _ in pp)}")
    open(f"{OUT}/cv_lg_{rec}_report.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
