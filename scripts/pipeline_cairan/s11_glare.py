"""Langkah 11: augmentasi kilau (specular glare) sintetis pada crop 'fold' (1 salinan tambahan per crop, seed per id).
Resep dari ide 'ellipsoidal specular patches + Gaussian blur' (arXiv 2604.09142) disesuaikan utk botol PET rebah:
  - 1-3 elips putih memanjang (sejajar sumbu botol = horizontal), intensitas acak 0.25-0.85, tepi di-blur sigma 2-8 px
  - 0-2 garis kilau tipis (streak), kontras tinggi
Hanya dipakai pada data LATIH (flag --glare di s03); data uji tidak diubah.
  python s11_glare.py [--sheet]
"""
import csv, os, sys, zlib
import cv2, numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); SRC = os.path.join(OUT, "crops_fold"); DST = os.path.join(OUT, "crops_fold_glare")
os.makedirs(DST, exist_ok=True)


def add_glare(img, seed):
    rng = np.random.default_rng(seed); h, w = img.shape[:2]
    mask = np.zeros((h, w), np.float32)
    for _ in range(int(rng.integers(1, 4))):
        cx, cy = rng.uniform(0.1, 0.9) * w, rng.uniform(0.1, 0.9) * h
        ax, ay = rng.uniform(0.08, 0.35) * w, rng.uniform(0.012, 0.07) * h
        tmp = np.zeros((h, w), np.float32); cv2.ellipse(tmp, (int(cx), int(cy)), (int(ax), max(1, int(ay))), float(rng.uniform(-12, 12)), 0, 360, 1.0, -1)
        mask = np.maximum(mask, tmp * rng.uniform(0.25, 0.85))
    for _ in range(int(rng.integers(0, 3))):
        x0, y0 = rng.uniform(0.05, 0.6) * w, rng.uniform(0.1, 0.9) * h
        x1, y1 = x0 + rng.uniform(0.2, 0.5) * w, y0 + rng.uniform(-0.02, 0.02) * h
        tmp = np.zeros((h, w), np.float32); cv2.line(tmp, (int(x0), int(y0)), (int(x1), int(y1)), 1.0, int(rng.integers(1, 4)))
        mask = np.maximum(mask, tmp * rng.uniform(0.4, 0.9))
    mask = cv2.GaussianBlur(mask, (0, 0), float(rng.uniform(2, 8)))
    pad = (img == 114).all(axis=2)                                   # jangan beri kilau di area padding abu
    mask[pad] = 0
    out = img.astype(np.float32) * (1 - mask[..., None]) + 255.0 * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    files = sorted(f for f in os.listdir(SRC) if f.endswith(".jpg"))
    for f in files:
        o = f"{DST}/{f}"
        if not os.path.exists(o):
            cv2.imwrite(o, add_glare(cv2.imread(f"{SRC}/{f}"), zlib.crc32(f.encode())), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(len(files), "crop kilau")
    if "--sheet" in sys.argv:
        sel = [f for f in files if f.startswith(("W16", "C13", "SODA", "P15"))][::13][:16]
        t = [cv2.resize(cv2.imread(f"{SRC}/{f}"), (200, 200)) for f in sel] + [cv2.resize(cv2.imread(f"{DST}/{f}"), (200, 200)) for f in sel]
        cv2.imwrite(f"{OUT}/glare_sheet.jpg", cv2.vconcat([cv2.hconcat(t[:8]), cv2.hconcat(t[8:16]), cv2.hconcat(t[16:24]), cv2.hconcat(t[24:32])]))
