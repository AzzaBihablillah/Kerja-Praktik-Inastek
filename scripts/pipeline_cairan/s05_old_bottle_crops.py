"""Langkah 5a: potong semua kotak 'bottle' di data warisan bottle_detection -> lembar bernomor untuk dipilah visual
(berwarna = berisi cairan berwarna yang terlihat; selain itu = bukan). Hasil pemilahan ditulis tangan ke
out/old_relabel_manual.csv oleh Claude (BELUM diverifikasi manusia)."""
import csv, os
import cv2, numpy as np

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
BD = os.path.join(ROOT, "5_data_luar/_warisan", "bottle_detection_1280")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); SD = os.path.join(OUT, "old_sheets"); os.makedirs(SD, exist_ok=True)
items = []
for f in sorted(os.listdir(f"{BD}/images")):
    stem = os.path.splitext(f)[0]
    lines = [l.split() for l in open(f"{BD}/labels/{stem}.txt").read().splitlines() if l.strip()]
    for k, l in enumerate(lines):
        if int(l[0]) == 0:
            items.append((stem, k, list(map(float, l[1:5]))))
print("kotak bottle:", len(items))
with open(f"{OUT}/old_bottle_boxes.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["idx", "stem", "line", "cx", "cy", "w", "h"])
    for i, (s, k, b) in enumerate(items):
        w.writerow([i, s, k, *b])
T = 150
tiles = []
cache = {}
for i, (s, k, (cx, cy, bw, bh)) in enumerate(items):
    im = cache.get(s)
    if im is None:
        cache.clear(); im = cache[s] = cv2.imread(f"{BD}/images/{s}.jpg")
    H, W = im.shape[:2]
    x1, y1, x2, y2 = int((cx - bw / 2) * W), int((cy - bh / 2) * H), int((cx + bw / 2) * W), int((cy + bh / 2) * H)
    c = im[max(0, y1):y2, max(0, x1):x2]
    if c.size == 0:
        c = np.zeros((10, 10, 3), np.uint8)
    s_ = T / max(c.shape[:2]); c = cv2.resize(c, (max(1, int(c.shape[1] * s_)), max(1, int(c.shape[0] * s_))))
    t = np.full((T + 16, T, 3), 40, np.uint8); oy, ox = 16 + (T - c.shape[0]) // 2, (T - c.shape[1]) // 2
    t[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    cv2.putText(t, str(i), (3, 12), 0, 0.42, (0, 255, 255), 1)
    tiles.append(t)
PER = 80                                                    # 10 x 8 per lembar
for s in range(0, len(tiles), PER):
    ch = tiles[s:s + PER]
    while len(ch) % 10:
        ch.append(np.zeros_like(ch[0]))
    cv2.imwrite(f"{SD}/old_{s // PER:02d}.jpg", cv2.vconcat([cv2.hconcat(ch[i:i + 10]) for i in range(0, len(ch), 10)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("lembar:", (len(tiles) + PER - 1) // PER)
