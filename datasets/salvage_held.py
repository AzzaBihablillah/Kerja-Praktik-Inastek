"""Salvage held/ cutouts: remove gloved hand + forearm, keep the (partial) bottle.

method: rembg bottle blob  MINUS  (yolov8-seg person mask  OR  tight orange-glove HSV),
protecting the labelled cap box.  Output RGBA -> held_nohand/ , transformed label, contact sheet.
flag when the hand ate too much of the bottle.
"""
import glob, os, re, csv, time
import cv2, numpy as np
from rembg import remove, new_session
from ultralytics import YOLO

SRC = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/cap_shoot_v1"
HELD = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/cap_shoot_v1_cutout/held"
OUT = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/cap_shoot_v1_cutout/held_nohand"
PAD = 0.04
os.makedirs(OUT, exist_ok=True)
os.makedirs(OUT + "/labels", exist_ok=True)

rs = new_session("u2net")
seg = YOLO("yolov8n-seg.pt")

stems = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(HELD + "/*.png"))
print(len(stems), "held cutouts to salvage")
rows, tiles = [], []
for i, stem in enumerate(stems):
    ip = f"{SRC}/images/{stem}.jpg"
    img = cv2.imread(ip)
    if img is None:
        rows.append((stem, "src_missing", "")); continue
    H, W = img.shape[:2]

    lp = f"{SRC}/labels/{stem}.txt"
    cb = None
    if os.path.exists(lp) and os.path.getsize(lp):
        _, cx, cy, bw, bh = map(float, open(lp).read().split()[:5])
        cb = [int((cx - bw / 2) * W), int((cy - bh / 2) * H),
              int((cx + bw / 2) * W), int((cy + bh / 2) * H)]

    # 1. bottle+glove blob
    sc = 900 / max(H, W)
    small = cv2.resize(img, (int(W * sc), int(H * sc)))
    a = np.array(remove(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), session=rs, alpha_matting=False))[:, :, 3]
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
    blob = a > 25
    nl, lab, st, _ = cv2.connectedComponentsWithStats(blob.astype(np.uint8), 8)
    if nl > 1:
        blob = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    before = int(blob.sum())

    # 2. person (hand/arm) mask
    person = np.zeros((H, W), bool)
    r = seg.predict(img, imgsz=1024, conf=0.25, classes=[0], device=0, verbose=False, retina_masks=True)[0]
    if r.masks is not None:
        for m in r.masks.data.cpu().numpy():
            person |= cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST) > 0.5
    person = cv2.dilate(person.astype(np.uint8), np.ones((9, 9), np.uint8), 1) > 0

    # 3. tight orange-glove HSV (skip orange bottle b05; never inside cap box)
    if not stem.startswith("b05_"):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        glove = cv2.inRange(hsv, (5, 90, 70), (18, 255, 255)) > 0
        glove = cv2.morphologyEx(glove.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8)) > 0
        person |= glove

    if cb:                                        # protect cap box
        x1, y1, x2, y2 = [max(0, v) for v in cb]
        person[y1:y2, x1:x2] = False

    # 4. subtract, keep largest CC
    mask = blob & ~person
    nl, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if nl > 1:
        mask = lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    after = int(mask.sum())
    if after < 400:
        rows.append((stem, "mask_fail", "")); continue
    keep = after / before
    reason = "OK" if keep >= 0.55 else f"hand ate {(1-keep)*100:.0f}% - body heavily occluded"
    rows.append((stem, reason, f"{keep:.2f}"))

    # 5. crop + save
    ys, xs = np.where(mask)
    x0, y0, x1_, y1_ = xs.min(), ys.min(), xs.max(), ys.max()
    px, py = int((x1_ - x0) * PAD), int((y1_ - y0) * PAD)
    cx0, cy0 = max(0, x0 - px), max(0, y0 - py)
    cx1, cy1 = min(W, x1_ + px), min(H, y1_ + py)
    rgba = np.dstack([img, mask.astype(np.uint8) * 255])[cy0:cy1, cx0:cx1]
    cv2.imwrite(f"{OUT}/{stem}.png", rgba)

    if cb:
        nx1 = (cb[0] - cx0) / (cx1 - cx0); ny1 = (cb[1] - cy0) / (cy1 - cy0)
        nx2 = (cb[2] - cx0) / (cx1 - cx0); ny2 = (cb[3] - cy0) / (cy1 - cy0)
        cid = 0 if "_cap_" in stem else 1
        open(f"{OUT}/labels/{stem}.txt", "w").write(
            f"{cid} {(nx1+nx2)/2:.6f} {(ny1+ny2)/2:.6f} {abs(nx2-nx1):.6f} {abs(ny2-ny1):.6f}\n")

    if i % 6 == 0:
        v = rgba[:, :, :3].copy(); v[rgba[:, :, 3] < 128] = (30, 30, 30)
        s = 190 / max(v.shape[:2]); v = cv2.resize(v, (int(v.shape[1] * s), int(v.shape[0] * s)))
        pad = np.full((200, 200, 3), 40, np.uint8); pad[:v.shape[0], :v.shape[1]] = v
        cv2.putText(pad, stem[:24].replace("b0", "b"), (2, 196), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        cv2.putText(pad, f"{keep:.2f}", (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 0) if keep >= 0.55 else (0, 128, 255), 1)
        tiles.append(pad)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(stems)}"); time.sleep(0.4)

with open(OUT + "/_review.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "reason", "kept_frac"]); w.writerows(rows)
if tiles:
    cols = 10; TH = 200
    r = (len(tiles) + cols - 1) // cols
    g = np.full((r * TH, cols * 200, 3), 40, np.uint8)
    for j, t in enumerate(tiles):
        g[(j // cols) * TH:(j // cols) * TH + 200, (j % cols) * 200:(j % cols) * 200 + 200] = t
    cv2.imwrite(os.path.dirname(OUT) + "/../_held_salvage_check.jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 82])
bad = sum(1 for x in rows if x[1] not in ("OK",) and "ate" in x[1])
print(f"done -> {OUT} | {len([x for x in rows if x[1]=='OK'])} clean | {bad} heavily occluded | see _review.csv")
