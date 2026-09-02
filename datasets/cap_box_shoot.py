"""Tight cap/neck box for shoot_v1: rembg mask -> oriented axis -> narrower end = cap end.
Writes X-AnyLabeling .json + a contact sheet + flags ambiguous ones.
"""
import glob, os, json, time, csv
import cv2, numpy as np
from rembg import remove, new_session
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

S = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/shoot_v1"
NAMES = {0: "cap", 1: "no_cap"}
CAP_FRAC = 0.24          # portion of bottle length taken as cap/neck region
sess = new_session("u2net")


def bottle_mask(img):
    H, W = img.shape[:2]
    sc = 900 / max(H, W)
    small = cv2.resize(img, (int(W * sc), int(H * sc)))
    a = np.array(remove(cv2.cvtColor(small, cv2.COLOR_BGR2RGB), session=sess, alpha_matting=False))[:, :, 3]
    a = cv2.resize(a, (W, H), interpolation=cv2.INTER_NEAREST)
    m = (a > 25).astype(np.uint8)
    nl, lab, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if nl > 1:
        m = (lab == 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    return m


def cap_box(m, slot_name):
    ys, xs = np.where(m)
    if len(xs) < 300:
        return None, "mask_fail"
    pts = np.column_stack([xs, ys]).astype(np.float32)
    (cxr, cyr), (rw, rh), ang = cv2.minAreaRect(pts)
    if rw < rh:                                   # make angle align long axis to x
        ang += 90
        rw, rh = rh, rw
    Rm = cv2.getRotationMatrix2D((cxr, cyr), ang, 1.0)
    H, W = m.shape
    mr = cv2.warpAffine(m, Rm, (W, H), flags=cv2.INTER_NEAREST)
    ys2, xs2 = np.where(mr)
    x0, x1 = xs2.min(), xs2.max()
    L = x1 - x0
    if L < 20:
        return None, "too_thin"
    # column heights along the long axis
    heights = np.array([(mr[:, x] > 0).sum() for x in range(x0, x1 + 1)], float)
    k = max(3, L // 20)
    hl = heights[:max(1, int(L * 0.18))].mean()
    hr = heights[-max(1, int(L * 0.18)):].mean()
    left_is_cap = hl < hr
    ratio = min(hl, hr) / max(hl, hr, 1e-6)
    # cap region: slice CAP_FRAC of L from the narrower end
    if left_is_cap:
        cx0, cx1 = x0, x0 + int(L * CAP_FRAC)
    else:
        cx0, cx1 = x1 - int(L * CAP_FRAC), x1
    seg = mr[:, cx0:cx1 + 1]
    sy = np.where(seg.any(axis=1))[0]
    if len(sy) < 5:
        return None, "seg_empty"
    cy0, cy1 = sy.min(), sy.max()
    # corners of the cap box in rotated space -> back to original
    corners = np.array([[cx0, cy0], [cx1, cy0], [cx1, cy1], [cx0, cy1]], np.float32)
    Rinv = cv2.invertAffineTransform(Rm)
    orig = (Rinv[:, :2] @ corners.T).T + Rinv[:, 2]
    bx1, by1 = orig.min(0)
    bx2, by2 = orig.max(0)
    bx1, by1 = max(0, bx1), max(0, by1)
    bx2, by2 = min(W, bx2), min(H, by2)
    flag = ""
    if ratio > 0.82:
        flag = f"end-width ambiguous ({ratio:.2f})"
    return (bx1, by1, bx2, by2), flag


jpgs = sorted(glob.glob(S + "/b*/*.jpg"))
print(len(jpgs), "images")
flags = []
tiles = []
for i, jp in enumerate(jpgs):
    rel = os.path.relpath(jp, S).replace("\\", "/")
    slot = rel.split("_")[-1].split(".")[0]
    img = cv2.imread(jp)
    H, W = img.shape[:2]
    m = bottle_mask(img)
    box, flag = cap_box(m, slot)
    shapes = []
    cid = 0 if "_cap_" in rel else 1
    if box:
        bx1, by1, bx2, by2 = box
        shapes = [{"label": NAMES[cid],
                   "points": [[round(bx1, 1), round(by1, 1)], [round(bx2, 1), round(by2, 1)]],
                   "group_id": None, "difficult": False, "shape_type": "rectangle",
                   "flags": {}, "attributes": {}}]
        if flag:
            flags.append((rel, flag))
    else:
        flags.append((rel, flag or "no_box"))
    json.dump({"version": "4.0.5", "flags": {}, "shapes": shapes, "imagePath": os.path.basename(jp),
               "imageData": None, "imageHeight": H, "imageWidth": W},
              open(os.path.splitext(jp)[0] + ".json", "w"), indent=2)
    if i % 10 == 0 and box:
        t = cv2.imread(jp, cv2.IMREAD_REDUCED_COLOR_2)
        s = t.shape[1] / W
        cv2.rectangle(t, (int(bx1 * s), int(by1 * s)), (int(bx2 * s), int(by2 * s)),
                      (0, 0, 255) if cid == 0 else (0, 200, 0), 2)
        t = cv2.resize(t, (240, int(240 * t.shape[0] / t.shape[1])))
        cv2.putText(t, rel.split("/")[-1][:22], (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)
        tiles.append(t)
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(jpgs)}")
        time.sleep(0.5)

with open(S + "/_review.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "reason"])
    w.writerows(flags)
if tiles:
    cols = 6
    TH = max(t.shape[0] for t in tiles)
    rows = (len(tiles) + cols - 1) // cols
    g = np.full((rows * TH, cols * 240, 3), 255, np.uint8)
    for i, t in enumerate(tiles):
        g[(i // cols) * TH:(i // cols) * TH + t.shape[0], (i % cols) * 240:(i % cols) * 240 + 240] = t
    cv2.imwrite(S + "/../capbox_check.jpg", g, [cv2.IMWRITE_JPEG_QUALITY, 78])
print(f"done | {len(flags)} flagged -> _review.csv | contact sheet: capbox_check.jpg")
