"""Clean per-bottle cutout from cap_shoot_v1 (2MP): plain rembg -> largest CC -> tight crop -> RGBA PNG.
Cap-box label transformed into crop coords.

Split by SLOT NUMBER (uniform for b01 bare slots and b02-b13 named slots):
  slot 01-03  = topdown, bottle not lifted, no hand   -> nohand/  (use for compositing)
  slot 04-10  = lift45 / side45, bottle held/angled    -> held/    (gloved hand in frame)
"""
import glob, os, re, time, csv
import cv2, numpy as np
from rembg import remove, new_session

SRC = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/cap_shoot_v1"
OUT = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/cap_shoot_v1_cutout"
PAD = 0.04
sess = new_session("u2net")

for d in ("nohand", "held", "nohand/labels", "held/labels"):
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

imgs = sorted(glob.glob(SRC + "/images/*.jpg"))
print(len(imgs), "images")
flags = []
for i, ip in enumerate(imgs):
    stem = os.path.splitext(os.path.basename(ip))[0]
    m = re.search(r"_(?:cap|nocap)_(\d{2})", stem)
    slot = int(m.group(1)) if m else 99
    bucket = "nohand" if slot <= 3 else "held"

    img = cv2.imread(ip); H, W = img.shape[:2]
    lp = os.path.join(SRC, "labels", stem + ".txt")
    cb = None
    if os.path.exists(lp) and os.path.getsize(lp):
        c, cx, cy, bw, bh = map(float, open(lp).read().split()[:5])
        cb = (int((cx - bw / 2) * W), int((cy - bh / 2) * H),
              int((cx + bw / 2) * W), int((cy + bh / 2) * H))

    sc = 900 / max(H, W)
    small = cv2.resize(img, (int(W * sc), int(H * sc)))
    a = np.array(remove(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), session=sess, alpha_matting=False))[:, :, 3]
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
    mask = a > 25

    nl, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if nl > 1:
        mask = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    if mask.sum() < 400:
        flags.append((stem, "mask_fail")); continue

    ys, xs = np.where(mask)
    x0, y0, x1_, y1_ = xs.min(), ys.min(), xs.max(), ys.max()
    px = int((x1_ - x0) * PAD); py = int((y1_ - y0) * PAD)
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(W, x1_ + px), min(H, y1_ + py)

    rgba = np.dstack([img, (mask.astype(np.uint8) * 255)])
    crop = rgba[cy0:cy1, cx0:cx1]
    cv2.imwrite(os.path.join(OUT, bucket, stem + ".png"), crop)

    if cb:
        nx1 = (cb[0] - cx0) / (cx1 - cx0); ny1 = (cb[1] - cy0) / (cy1 - cy0)
        nx2 = (cb[2] - cx0) / (cx1 - cx0); ny2 = (cb[3] - cy0) / (cy1 - cy0)
        ncx, ncy = (nx1 + nx2) / 2, (ny1 + ny2) / 2
        nbw, nbh = abs(nx2 - nx1), abs(ny2 - ny1)
        cid = 0 if "_cap_" in stem else 1
        open(os.path.join(OUT, bucket, "labels", stem + ".txt"), "w").write(
            f"{cid} {ncx:.6f} {ncy:.6f} {nbw:.6f} {nbh:.6f}\n")

    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(imgs)}"); time.sleep(0.5)

with open(OUT + "/_review.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "reason"]); w.writerows(flags)
n_nh = len(glob.glob(OUT + "/nohand/*.png"))
n_hd = len(glob.glob(OUT + "/held/*.png"))
print(f"done -> nohand {n_nh} | held {n_hd} | {len(flags)} flagged")
