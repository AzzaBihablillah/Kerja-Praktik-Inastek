"""Langkah 9: sapuan ambang classifier pada pipeline PB (ver8 + classifier) dari rekaman s08 (tanpa inferensi ulang)."""
import json, os
OUT = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets"), "4_eksperimen", "cairan")
lines = []
for v in ("V2_fold_belt", "V4_fold_w16only"):
    R = json.load(open(f"{OUT}/det_eval_{v}.json"))["records"]
    lines.append(f"== {v}: ambang | W16 berisi->berwarna | P15 berisi->berwarna | DITERIMA: W16 kosong | P15 kosong | P15 air bening | C13 kosong")
    for t in (0.9, 0.7, 0.5, 0.3, 0.2, 0.1):
        def rate(src, liq, want_berwarna):
            G = [r for r in R if r["src"] == src and r["liquid"] == liq]
            ok = sum((r["PB_det"] and r["PB_p"] is not None and r["PB_p"] >= t) if want_berwarna else (r["PB_det"] and (r["PB_p"] is None or r["PB_p"] < t)) for r in G)
            return f"{ok}/{len(G)}={ok / len(G):.2f}"
        lines.append(f"   {t:.1f} | {rate('W16', 'berwarna', 1)} | {rate('P15', 'berwarna', 1)} | {rate('W16', 'kosong', 0)} | {rate('P15', 'kosong', 0)} | "
                     f"{rate('P15', 'air_bening', 0)} | {rate('C13', 'kosong', 0)}")
open(f"{OUT}/threshold_sweep.txt", "w").write("\n".join(lines) + "\n"); print("\n".join(lines))
