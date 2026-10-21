"""Langkah 16: potong semua kotak kelas coca_cola (dan nanti unknown) dari data warisan brand2 -> lembar bernomor utk dipilah visual
(berwarna = cairan berwarna terlihat / botol penuh; kosong = kosong atau cairan bening). Kotak juga disimpan utk crop belt-style."""
import csv, os, sys
import cv2, numpy as np
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); B2 = "C:/Users/ASUS/Downloads/brand2"
cls_want = sys.argv[1] if len(sys.argv) > 1 else "coca_cola"
names = open(f"{B2}/classes.txt").read().split()
SD = f"{OUT}/brand2_sheets_{cls_want}"; os.makedirs(SD, exist_ok=True)
items = []
for f in sorted(os.listdir(f"{B2}/labels")):
    stem = f[:-4]
    for k, l in enumerate(open(f"{B2}/labels/{f}").read().splitlines()):
        p = l.split()
        if len(p) == 5 and names[int(p[0])] == cls_want:
            items.append((stem, k, list(map(float, p[1:]))))
print("kotak", cls_want, len(items))
with open(f"{OUT}/brand2_{cls_want}_boxes.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["idx", "stem", "line", "cx", "cy", "w", "h"])
    for i, (s, k, b) in enumerate(items):
        w.writerow([i, s, k, *b])
T = 150; tiles = []; cache = {}
for i, (s, k, (cx, cy, bw, bh)) in enumerate(items):
    if s not in cache:
        cache.clear()
        for ext in (".jpg", ".jpeg", ".png", ".webp"):
            if os.path.exists(f"{B2}/images/{s}{ext}"):
                cache[s] = cv2.imread(f"{B2}/images/{s}{ext}"); break
    im = cache.get(s)
    t = np.full((T + 16, T, 3), 40, np.uint8)
    if im is not None:
        H, W = im.shape[:2]; x1, y1, x2, y2 = int((cx - bw / 2) * W), int((cy - bh / 2) * H), int((cx + bw / 2) * W), int((cy + bh / 2) * H)
        c = im[max(0, y1):y2, max(0, x1):x2]
        if c.size:
            s_ = T / max(c.shape[:2]); c = cv2.resize(c, (max(1, int(c.shape[1] * s_)), max(1, int(c.shape[0] * s_))))
            oy, ox = 16 + (T - c.shape[0]) // 2, (T - c.shape[1]) // 2; t[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    cv2.putText(t, str(i), (3, 12), 0, 0.42, (0, 255, 255), 1); tiles.append(t)
PER = 80
for s in range(0, len(tiles), PER):
    ch = tiles[s:s + PER]
    while len(ch) % 10:
        ch.append(np.zeros_like(ch[0]))
    cv2.imwrite(f"{SD}/b2_{s // PER:02d}.jpg", cv2.vconcat([cv2.hconcat(ch[i:i + 10]) for i in range(0, len(ch), 10)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
print("lembar", (len(tiles) + PER - 1) // PER)
