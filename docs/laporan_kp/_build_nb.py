"""Build laporan-deteksi-cap-nocap.ipynb from the .md report + embedded plot images.
Self-contained: images are base64-embedded as cell attachments (no external file deps).
"""
import base64, json, os, re

DOCS = os.path.dirname(os.path.abspath(__file__))
DS = os.path.join(DOCS, "..", "datasets")
MD = os.path.join(DOCS, "laporan-deteksi-cap-nocap.md")
OUT = os.path.join(DOCS, "laporan-deteksi-cap-nocap.ipynb")

# images to insert, keyed by the section number they follow
IMG = {
    "4": [
        ("_nohand_full.jpg", "Contact sheet 78 cutout `nohand/` (top-down, tanpa tangan)."),
        ("_held_nohand_all.jpg", "182 cutout `held_nohand/` setelah tangan dibuang, diurut `kept_frac` (jelek→bagus). Hijau ≥0,90 bersih; oranye <0,55 tidak tertolong."),
        ("_hand_residual_check.jpg", "15 cutout keflag di verifikasi akhir — 9 false-positive (logo/cap oranye) dikembalikan, 6 asli ber-tangan dikarantina."),
    ],
    "6": [
        ("_synth_sample.jpg", "Contoh 12 gambar sintetis `synth_cap_v1` (train) dengan ground-truth box. Background 100% di-generate, warna acak; botol dirotasi/diskala/didegradasi."),
    ],
    "7": [
        ("cap_runs/synth_cap_v1_640/results.png", "v1 — kurva training 100 epoch. Loss train & val turun mulus, metrik plateau ~epoch 40–50, tidak overfit."),
        ("cap_runs/synth_cap_v1_640_test/confusion_matrix_normalized.png", "v1 — confusion matrix di TEST split."),
        ("cap_runs/synth_cap_v1_640_test/BoxPR_curve.png", "v1 — Precision-Recall curve di TEST split."),
        ("_conveyor_eval.jpg", "v1 di 12 foto conveyor top-down asli (belt hitam) — box ketat di area tutup, conf 0,80–0,89. Sumber sebagian tumpang tindih dg training."),
    ],
    "8": [
        ("cap_runs/synth_cap_v2_640/results.png", "v2 — kurva training 100 epoch."),
        ("cap_runs/synth_cap_v2_640/confusion_matrix_normalized.png", "v2 — confusion matrix (val). cap↔no_cap tertukar hanya 1–2%."),
    ],
    "9": [
        ("cap_runs/synth_cap_v3_640/results.png", "v3 — kurva training 70 epoch (degradasi simetris cap & no_cap)."),
        ("cap_runs/synth_cap_v3_640/confusion_matrix_normalized.png", "v3 — confusion matrix (val). Performa cap & no_cap seimbang."),
    ],
}


def b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("ascii")


def md_cell(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": [l + "\n" for l in src.rstrip().split("\n")]}


def img_cell(fname, caption, key):
    path = os.path.normpath(os.path.join(DS, fname))
    mime = "image/jpeg" if path.lower().endswith((".jpg", ".jpeg")) else "image/png"
    ext = "jpg" if mime == "image/jpeg" else "png"
    aname = key + "." + ext
    return {"cell_type": "markdown",
            "metadata": {},
            "attachments": {aname: {mime: b64(path)}},
            "source": [f"![{key}](attachment:{aname})\n", "\n", f"*{caption}*\n"]}


text = open(MD, encoding="utf-8").read()
# split on top-level "## N. " headings, keep the heading
parts = re.split(r"(?m)^(?=## )", text)
cells = []
for part in parts:
    if not part.strip():
        continue
    cells.append(md_cell(part))
    m = re.match(r"## (\d+)\.", part)
    if m and m.group(1) in IMG:
        for i, (fn, cap) in enumerate(IMG[m.group(1)]):
            cells.append(img_cell(fn, cap, f"s{m.group(1)}_{i}"))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13"},
        "title": "Laporan Deteksi Tutup Botol (cap / no_cap) — RVM",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print(f"wrote {OUT}  ({os.path.getsize(OUT)/1024:.0f} KB, {len(cells)} cells)")
