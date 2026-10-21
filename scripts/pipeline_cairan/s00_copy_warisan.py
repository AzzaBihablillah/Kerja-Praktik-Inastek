"""Salin data warisan dari Downloads ke datasets/5_data_luar/_warisan/ (asli tidak diubah).
bottle_detection diperkecil (sisi terpanjang 1280) agar hemat disk & cepat; label YOLO ternormalisasi jadi tetap berlaku.
foto_botol & foto_botol_update disalin apa adanya."""
import os, shutil, sys, cv2
SRC = "C:/Users/ASUS/Downloads"
DST = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/5_data_luar/_warisan"
os.makedirs(DST, exist_ok=True)
for d in ("foto_botol", "foto_botol_update"):
    if not os.path.exists(f"{DST}/{d}"):
        shutil.copytree(f"{SRC}/{d}", f"{DST}/{d}")
bd, out = f"{SRC}/bottle_detection", f"{DST}/bottle_detection_1280"
os.makedirs(f"{out}/images", exist_ok=True); os.makedirs(f"{out}/labels", exist_ok=True)
for f in ("classes.txt", "notes.json"):
    shutil.copy2(f"{bd}/{f}", out)
n = bad = 0
for f in sorted(os.listdir(f"{bd}/images")):
    stem = os.path.splitext(f)[0]
    o = f"{out}/images/{stem}.jpg"
    if os.path.exists(o):
        n += 1; continue
    im = cv2.imread(f"{bd}/images/{f}")
    if im is None:
        bad += 1; print("gagal baca", f); continue
    h, w = im.shape[:2]; s = 1280 / max(h, w)
    if s < 1:
        im = cv2.resize(im, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(o, im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    shutil.copy2(f"{bd}/labels/{stem}.txt", f"{out}/labels/{stem}.txt")
    n += 1
print("bottle_detection:", n, "gambar, gagal", bad)
