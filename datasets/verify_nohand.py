"""Final hand check on the clean cutout set (nohand/ + held_nohand/ top level).
Flag = orange-glove HSV fraction OR yolov8-seg person fraction over threshold.
Flagged files -> held_nohand/_hand_residual/ ; contact sheet of flagged.
"""
import glob, os, shutil, cv2, numpy as np
from ultralytics import YOLO

ROOT = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/cap_shoot_v1_cutout"
paths = sorted(glob.glob(ROOT + "/nohand/*.png")) + sorted(glob.glob(ROOT + "/held_nohand/*.png"))
print(len(paths), "clean cutouts to verify")
seg = YOLO("yolov8n-seg.pt")
QUAR = ROOT + "/held_nohand/_hand_residual"
os.makedirs(QUAR, exist_ok=True)

flagged, tiles = [], []
for i, p in enumerate(paths):
    stem = os.path.splitext(os.path.basename(p))[0]
    im = cv2.imread(p, cv2.IMREAD_UNCHANGED)
    if im is None or im.shape[2] != 4:
        continue
    bgr, al = im[:, :, :3], im[:, :, 3]
    fg = al > 128
    area = max(int(fg.sum()), 1)

    orange_frac = 0.0
    if not stem.startswith("b05_"):                       # b05 = natsbee orange bottle
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        glove = (cv2.inRange(hsv, (5, 95, 80), (18, 255, 255)) > 0) & fg
        glove = cv2.morphologyEx(glove.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)) > 0
        orange_frac = glove.sum() / area

    vis = bgr.copy(); vis[~fg] = (255, 255, 255)
    r = seg.predict(vis, imgsz=768, conf=0.30, classes=[0], device=0, verbose=False, retina_masks=True)[0]
    person = np.zeros(fg.shape, bool)
    if r.masks is not None:
        for m in r.masks.data.cpu().numpy():
            person |= cv2.resize(m, fg.shape[::-1], interpolation=cv2.INTER_NEAREST) > 0.5
    person_frac = (person & fg).sum() / area

    if orange_frac > 0.010 or person_frac > 0.025:
        flagged.append((stem, round(orange_frac, 4), round(person_frac, 4)))
        shutil.move(p, f"{QUAR}/{stem}.png")
        lp = os.path.join(os.path.dirname(p), "labels", stem + ".txt")
        if os.path.exists(lp):
            os.makedirs(QUAR + "/labels", exist_ok=True)
            shutil.move(lp, f"{QUAR}/labels/{stem}.txt")
        t = cv2.resize(vis, (200, int(200 * vis.shape[0] / max(vis.shape[1], 1))))
        cv2.putText(t, f"o{orange_frac:.3f} p{person_frac:.3f}", (2, 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(t, stem[:22], (2, t.shape[0] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 128, 0), 1)
        tiles.append(t)

print(f"\nFLAGGED {len(flagged)} (moved to _hand_residual/):")
for f in flagged:
    print("  ", f)
nh = len(glob.glob(ROOT + "/nohand/*.png"))
hn = len(glob.glob(ROOT + "/held_nohand/*.png"))
print(f"\nclean now: nohand {nh} + held_nohand {hn} = {nh + hn}")
if tiles:
    TH = max(t.shape[0] for t in tiles); cols = 8
    rows = (len(tiles) + cols - 1) // cols
    g = np.full((rows * TH, cols * 200, 3), 230, np.uint8)
    for j, t in enumerate(tiles):
        g[(j // cols) * TH:(j // cols) * TH + t.shape[0], (j % cols) * 200:(j % cols) * 200 + t.shape[1]] = t
    cv2.imwrite("C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/_hand_residual_check.jpg", g,
                [cv2.IMWRITE_JPEG_QUALITY, 82])
    print("contact sheet -> datasets/_hand_residual_check.jpg")
