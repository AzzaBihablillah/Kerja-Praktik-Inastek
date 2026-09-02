"""Cut-paste synthetic dataset for cap / no_cap.

149 clean bottle cutouts (nohand/ + held_nohand/) -> composited onto fully-random
generated backgrounds (option A: every colour uniform-random, no belt bias) with
heavy augmentation.  Cap-box label transformed through every geometric step.

Grouped split by cutout stem (all variants of one cutout stay in one split), 80/10/10,
stratified by class.  Output YOLO -> datasets/synth_cap_v3/{images,labels}/{train,val,test}
Conveyor real data is appended afterwards by add_conveyor.py.
"""
import glob, os, gc, math, random
import cv2, numpy as np

D = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/cap_shoot_v1_cutout"
OUT = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/synth_cap_v3"
FRAME = 640
VARIANTS = {0: 30, 1: 44}                 # per-class variant count (cap=0, no_cap=1) -- moderate no_cap oversample (~1.5x)
SPLIT = (0.80, 0.10, 0.10)
SEED = 0

for s in ("train", "val", "test"):
    os.makedirs(f"{OUT}/images/{s}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{s}", exist_ok=True)


def load_cutouts():
    items = []
    for folder in ("nohand", "held_nohand"):
        for p in sorted(glob.glob(f"{D}/{folder}/*.png")):
            stem = os.path.splitext(os.path.basename(p))[0]
            lp = f"{D}/{folder}/labels/{stem}.txt"
            if not os.path.exists(lp) or not os.path.getsize(lp):
                continue
            cid, cx, cy, bw, bh = map(float, open(lp).read().split()[:5])
            items.append(dict(stem=stem, png=p, cls=int(cid),
                              box=np.array([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2])))
    return items


def _noise(img, sd, g):
    n = g.standard_normal(img.shape) * sd
    return np.clip(img.astype(np.float32) + n, 0, 255).astype(np.uint8)


def rand_bg(h, w, g):
    mode = int(g.integers(0, 5))
    if mode == 0:
        bg = np.full((h, w, 3), g.integers(0, 256, 3), np.uint8)
    elif mode == 1:
        cols = g.integers(0, 256, (int(g.integers(2, 4)), 3)).astype(np.float32)
        if g.random() < 0.5:
            ang = g.uniform(0, math.pi)
            gx = np.cos(ang) * np.arange(w)[None, :] + np.sin(ang) * np.arange(h)[:, None]
            t = (gx - gx.min()) / (np.ptp(gx) + 1e-6)
        else:
            yy, xx = np.mgrid[0:h, 0:w]
            cy, cx = g.uniform(0, h), g.uniform(0, w)
            d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
            t = d / (d.max() + 1e-6)
        seg = np.linspace(0, 1, len(cols))
        bg = np.stack([np.interp(t, seg, cols[:, c]) for c in range(3)], -1).astype(np.uint8)
    elif mode == 2:
        base = g.integers(0, 256, 3); other = g.integers(0, 256, 3)
        bg = np.full((h, w, 3), base, np.uint8)
        period = int(g.integers(10, 70)); ang = g.uniform(-1, 1)
        yy, xx = np.mgrid[0:h, 0:w]
        band = ((xx * math.cos(ang) + yy * math.sin(ang)) // period).astype(int) % 2
        bg[band == 1] = other
    elif mode == 3:
        bg = np.full((h, w, 3), g.integers(0, 256, 3), np.uint8).astype(np.float32)
        for _ in range(int(g.integers(3, 9))):
            ov = np.zeros((h, w, 3), np.float32)
            cv2.ellipse(ov, (int(g.uniform(0, w)), int(g.uniform(0, h))),
                        (int(g.uniform(w * .1, w * .6)), int(g.uniform(h * .1, h * .6))),
                        g.uniform(0, 180), 0, 360, [float(x) for x in g.integers(0, 256, 3)], -1)
            a = g.uniform(0.1, 0.4)
            bg = bg * (1 - a) + ov * a
        bg = bg.astype(np.uint8)
    else:
        c1 = g.integers(0, 256, 3); c2 = g.integers(0, 256, 3)
        cell = int(g.integers(16, 90))
        yy, xx = np.mgrid[0:h, 0:w]
        chk = (((xx // cell) + (yy // cell)) % 2).astype(bool)
        bg = np.where(chk[..., None], c1, c2).astype(np.uint8)

    bg = _noise(bg, g.uniform(2, 13), g)
    bg = cv2.convertScaleAbs(bg, alpha=g.uniform(0.55, 1.4), beta=g.uniform(-25, 25))
    if g.random() < 0.3:
        ky = cv2.getGaussianKernel(h, g.uniform(h * .3, h * .8))
        kx = cv2.getGaussianKernel(w, g.uniform(w * .3, w * .8))
        m = ky @ kx.T; m = m / m.max()
        bg = (bg.astype(np.float32) * (0.35 + 0.65 * m[..., None])).astype(np.uint8)
    return bg


def rotate_keep(img, ang):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    M[0, 2] += nw / 2 - w / 2
    M[1, 2] += nh / 2 - h / 2
    return M, (nw, nh)


def paste(canvas, it, g):
    obj = cv2.imread(it["png"], cv2.IMREAD_UNCHANGED)
    if obj is None or obj.shape[2] != 4:
        return canvas, None
    oh, ow = obj.shape[:2]
    bgr, al = obj[:, :, :3], obj[:, :, 3]
    box = it["box"] * [ow, oh, ow, oh]

    ang = float(np.clip(g.normal(0, 18), -42, 42))
    if g.random() < 0.08:
        ang = g.uniform(-180, 180)
    M, (nw, nh) = rotate_keep(bgr, ang)
    rb = cv2.warpAffine(bgr, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    ra = cv2.warpAffine(al, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)

    ch, cw = canvas.shape[:2]
    fit = (min(cw, ch) * g.uniform(0.30, 0.82)) / max(nw, nh)
    rb = cv2.resize(rb, None, fx=fit, fy=fit)
    ra = cv2.resize(ra, None, fx=fit, fy=fit)
    rh, rw = rb.shape[:2]
    if rw >= cw or rh >= ch:
        return canvas, None
    px, py = int(g.integers(0, cw - rw + 1)), int(g.integers(0, ch - rh + 1))

    rb = cv2.convertScaleAbs(rb, alpha=g.uniform(0.80, 1.20), beta=g.uniform(-20, 20))
    if g.random() < 0.35:
        hsv = cv2.cvtColor(rb, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 0] = (hsv[..., 0] + int(g.integers(-9, 10))) % 180
        rb = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)

    sm = np.zeros((ch, cw), np.float32)
    sm[py:py + rh, px:px + rw] = ra.astype(np.float32) / 255.0
    dx, dy = int(g.integers(-14, 15)), int(g.integers(4, 19))
    sm = cv2.GaussianBlur(np.roll(np.roll(sm, dy, 0), dx, 1), (0, 0), g.uniform(6, 16))
    canvas = (canvas.astype(np.float32) * (1 - g.uniform(0.25, 0.5) * sm[..., None])).astype(np.uint8)

    a3 = ra[..., None] / 255.0
    roi = canvas[py:py + rh, px:px + rw].astype(np.float32)
    canvas[py:py + rh, px:px + rw] = (rb * a3 + roi * (1 - a3)).astype(np.uint8)

    c = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]], np.float32)
    c = (M[:, :2] @ c.T).T + M[:, 2]
    c = c * fit + [px, py]
    return canvas, (c, it["cls"], (px, py, rw, rh))


def coarse_dropout(canvas, region, g):
    px, py, rw, rh = region
    for _ in range(int(g.integers(1, 5))):
        dw = int(g.integers(max(1, rw // 8), max(2, rw // 3)))
        dh = int(g.integers(max(1, rh // 8), max(2, rh // 3)))
        x = int(g.integers(px, max(px + 1, px + rw - dw)))
        y = int(g.integers(py, max(py + 1, py + rh - dh)))
        canvas[y:y + dh, x:x + dw] = g.integers(60, 190, 3)
    return canvas


def perspective(canvas, boxes, g):
    h, w = canvas.shape[:2]
    j = 0.08
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = (src + g.uniform(-j, j, (4, 2)) * [w, h]).astype(np.float32)
    H = cv2.getPerspectiveTransform(src, dst)
    canvas = cv2.warpPerspective(canvas, H, (w, h), borderMode=cv2.BORDER_REFLECT)
    out = [(cv2.perspectiveTransform(c[None].astype(np.float32), H)[0], cls) for c, cls in boxes]
    return canvas, out


def full_frame_aug(canvas, boxes, g):
    if g.random() < 0.25:
        canvas, boxes = perspective(canvas, boxes, g)
    if g.random() < 0.18:
        k = int(g.integers(3, 12)); ker = np.zeros((k, k)); ker[k // 2, :] = 1.0 / k
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2, k / 2), g.uniform(0, 180), 1), (k, k))
        canvas = cv2.filter2D(canvas, -1, ker)
    if g.random() < 0.15:
        k = int(g.choice([3, 5, 7])); canvas = cv2.GaussianBlur(canvas, (k, k), 0)
    if g.random() < 0.5:
        canvas = cv2.convertScaleAbs(canvas, alpha=g.uniform(0.75, 1.3), beta=g.uniform(-28, 28))
    if g.random() < 0.35:
        hsv = cv2.cvtColor(canvas, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 0] = (hsv[..., 0] + g.uniform(-10, 10)) % 180
        hsv[..., 1] = np.clip(hsv[..., 1] * g.uniform(0.7, 1.3), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] * g.uniform(0.75, 1.25), 0, 255)
        canvas = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if g.random() < 0.15:
        lab = cv2.cvtColor(canvas, cv2.COLOR_BGR2LAB)
        lab[..., 0] = cv2.createCLAHE(2.0, (8, 8)).apply(lab[..., 0])
        canvas = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if g.random() < 0.25:
        h, w = canvas.shape[:2]
        pts = g.integers(0, [w, h], (int(g.integers(3, 6)), 2))
        m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [pts.astype(np.int32)], 255)
        m = cv2.GaussianBlur(m, (0, 0), g.uniform(8, 30)).astype(np.float32) / 255.0
        canvas = (canvas.astype(np.float32) * (1 - m[..., None] * g.uniform(0.2, 0.55))).astype(np.uint8)
    if g.random() < 0.4:
        canvas = _noise(canvas, g.uniform(3, 16), g)
    return canvas, boxes


def hard_degrade(canvas, g):
    """Quality loss (low-res + blur + noise + jpeg). Applied EQUALLY to cap and no_cap
    so image sharpness never becomes a class cue (the v2 mistake). Boxes unaffected."""
    h, w = canvas.shape[:2]
    s = int(g.integers(150, 360))                       # simulate low-res sensor
    canvas = cv2.resize(canvas, (s, s), interpolation=cv2.INTER_AREA)
    canvas = cv2.resize(canvas, (w, h), interpolation=cv2.INTER_LINEAR)
    if g.random() < 0.7:
        k = int(g.choice([3, 5, 7, 9])); canvas = cv2.GaussianBlur(canvas, (k, k), 0)
    if g.random() < 0.4:
        k = int(g.integers(5, 15)); ker = np.zeros((k, k)); ker[k // 2, :] = 1.0 / k
        ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((k / 2, k / 2), g.uniform(0, 180), 1), (k, k))
        canvas = cv2.filter2D(canvas, -1, ker)
    canvas = _noise(canvas, g.uniform(8, 26), g)
    q = int(g.integers(18, 46))                          # brutal jpeg round-trip
    ok, enc = cv2.imencode(".jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, q])
    if ok:
        canvas = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return canvas


def emit(boxes, W, H):
    lines = []
    for c, cls in boxes:
        x1, y1 = c.min(0); x2, y2 = c.max(0)
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 4 or y2 - y1 < 4:
            continue
        lines.append(f"{cls} {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} {(x2-x1)/W:.6f} {(y2-y1)/H:.6f}")
    return lines


def assign_splits(items):
    rng = random.Random(SEED)
    split_of = {}
    for cls in (0, 1):
        stems = sorted({it["stem"] for it in items if it["cls"] == cls})
        rng.shuffle(stems)
        n = len(stems); n_tr = int(n * SPLIT[0]); n_va = int(n * SPLIT[1])
        for i, st in enumerate(stems):
            split_of[st] = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    return split_of


def main():
    items = load_cutouts()
    by_stem = {it["stem"]: it for it in items}
    stems = list(by_stem)
    split_of = assign_splits(items)
    cnt = {"train": 0, "val": 0, "test": 0}

    for idx, it in enumerate(items, 1):
        split = split_of[it["stem"]]
        for v in range(VARIANTS[it["cls"]]):
            g = np.random.default_rng(hash((it["stem"], v)) & 0xFFFFFFFF)
            canvas = rand_bg(FRAME, FRAME, g)
            canvas, r = paste(canvas, it, g)
            if r is None:
                continue
            c, cls, region = r
            boxes = [(c, cls)]
            if g.random() < 0.15:
                o = by_stem[stems[int(g.integers(len(stems)))]]
                canvas, r2 = paste(canvas, o, g)
                if r2 is not None:
                    boxes.append((r2[0], r2[1]))
            if g.random() < 0.22:
                canvas = coarse_dropout(canvas, region, g)
            canvas, boxes = full_frame_aug(canvas, boxes, g)
            if g.random() < 0.40:                               # bad-camera robustness, BOTH classes
                canvas = hard_degrade(canvas, g)
            lines = emit(boxes, FRAME, FRAME)
            if not lines:
                continue
            name = f"syn_{it['stem']}_{v:02d}"
            q = int(g.uniform(30, 92))                          # same jpeg range for both classes
            cv2.imwrite(f"{OUT}/images/{split}/{name}.jpg", canvas, [cv2.IMWRITE_JPEG_QUALITY, q])
            open(f"{OUT}/labels/{split}/{name}.txt", "w").write("\n".join(lines) + "\n")
            cnt[split] += 1
        if idx % 20 == 0:
            print(f"  {idx}/{len(items)} cutouts | {sum(cnt.values())} imgs", flush=True)
            gc.collect()

    print(f"SYNTH done: train {cnt['train']} val {cnt['val']} test {cnt['test']} = {sum(cnt.values())}", flush=True)


if __name__ == "__main__":
    main()
