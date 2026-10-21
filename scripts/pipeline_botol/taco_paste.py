"""Negatif non_bottle sintetis: potongan objek TACO (poligon segmentasi asli) ditempel ke foto belt nyata.
Dipakai untuk (1) UJI: apakah detektor botol keliru menerima sampah non-botol sebagai botol, (2) LATIH ver9 (bila perlu).

Sumber objek: TACO (CC BY 4.0, Proenca & Simoes 2020; _dl_taco/raw640 + annotations.json), kategori non-botol (film, pembungkus,
  kantong, gelas, kaleng, karton, wadah, styrofoam, foil, dll.), hanya gambar TANPA botol plastik berlabel, sisi panjang >= 120 px.
  Split per GAMBAR TACO: id % 5 == 0 -> uji, lainnya -> latih (objek uji tidak pernah dilatih).
Latar: latih = foto belt yang sudah ada di data latih ver8b (C13, W16, P15; 960x540/640x360);
       uji   = sesi 20 Okt (tidak pernah dilatih detektor), di-resize ke 960x540.
Tempel: rotasi acak, skala sisi panjang 15-40% lebar frame (perbesaran <= 3x), tepi mask dihaluskan, kecerahan disesuaikan sebagian
  ke belt, blur ringan; diletakkan di pita belt tanpa menimpa botol. Mitigasi jalan pintas "tepi tempelan = non_bottle":
  pada set latih, 50% gambar juga mendapat tempelan POTONGAN BOTOL ASLI (dari foto belt lain) berlabel bottle.
Keterbatasan: tetap sintetis; angka akhir non-botol harus dari foto nyata di belt (plastik diremas, kantong, gelas)."""
import glob, json, os, random
import cv2, numpy as np

DS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
T = f"{DS}/5_data_luar/_dl_taco"
DD = f"{DS}/4_eksperimen/cairan/det_data"
NEG = {"Plastic film", "Other plastic wrapper", "Single-use carrier bag", "Garbage bag", "Crisp packet", "Disposable plastic cup", "Drink can",
       "Food Can", "Food can", "Glass bottle", "Drink carton", "Other carton", "Disposable food container", "Foam food container", "Foam cup",
       "Styrofoam piece", "Aluminium foil", "Paper bag", "Plastic lid", "Paper cup", "Meal carton", "Tupperware", "Spread tub",
       "Squeezable tube", "Aerosol", "Glass jar", "Glass cup", "Other plastic", "Corrugated carton", "Other plastic container",
       "Other plastic cup", "Polypropylene bag", "Plastic glooves", "Six pack rings", "Egg carton", "Pizza box", "Normal paper"}
BOT = {"Clear plastic bottle", "Other plastic bottle"}


def load_objects(min_side=120):
    d = json.load(open(f"{T}/annotations.json")); cat = {c["id"]: c["name"] for c in d["categories"]}
    by = {}
    for a in d["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    objs = {"latih": [], "uji": []}
    for im in d["images"]:
        p = f"{T}/raw640/taco_{im['id']:04d}.jpg"
        A = by.get(im["id"], [])
        if not os.path.exists(p) or {cat[a["category_id"]] for a in A} & BOT:
            continue
        img = None
        for a in A:
            if cat[a["category_id"]] not in NEG or not a.get("segmentation") or isinstance(a["segmentation"], dict):
                continue
            if img is None:
                img = cv2.imread(p)
                if img is None:
                    break
                sx, sy = img.shape[1] / im["width"], img.shape[0] / im["height"]
            x, y, w, h = a["bbox"]
            if max(w * sx, h * sy) < min_side:
                continue
            mask = np.zeros(img.shape[:2], np.uint8)
            for poly in a["segmentation"]:
                pts = (np.array(poly, np.float32).reshape(-1, 2) * [sx, sy]).astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)
            ys, xs = np.where(mask > 0)
            if len(xs) < 400:
                continue
            x1, x2, y1, y2 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
            rgba = np.dstack([img[y1:y2, x1:x2], mask[y1:y2, x1:x2]])
            objs["uji" if im["id"] % 5 == 0 else "latih"].append(dict(rgba=rgba, cat=cat[a["category_id"]], img=im["id"]))
    return objs


def _rotate(rgba, ang):
    h, w = rgba.shape[:2]; M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    c, s = abs(M[0, 0]), abs(M[0, 1]); nw, nh = int(h * s + w * c), int(h * c + w * s)
    M[0, 2] += nw / 2 - w / 2; M[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(rgba, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))


def _blend(frame, rgba, x0, y0):
    h, w = rgba.shape[:2]; a = rgba[:, :, 3:4].astype(np.float32) / 255
    roi = frame[y0:y0 + h, x0:x0 + w].astype(np.float32)
    frame[y0:y0 + h, x0:x0 + w] = (rgba[:, :, :3] * a + roi * (1 - a)).astype(np.uint8)


def _free(box, taken):
    for t in taken:
        if not (box[2] < t[0] or box[0] > t[2] or box[3] < t[1] or box[1] > t[3]):
            return False
    return True


def paste_object(frame, obj, taken, rng):
    """Tempel satu objek TACO; kembalikan kotak (x1,y1,x2,y2) atau None."""
    H, W = frame.shape[:2]
    rgba = obj["rgba"].copy()
    if rng.random() < 0.5:
        rgba = rgba[:, ::-1]
    rgba = _rotate(rgba, rng.uniform(0, 360))
    ys, xs = np.where(rgba[:, :, 3] > 0); rgba = rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    L = rng.uniform(0.15, 0.40) * W; s = min(L / max(rgba.shape[:2]), 3.0)
    if max(rgba.shape[:2]) * s > 0.45 * W or rgba.shape[0] * s > 0.42 * H:
        s = min(0.45 * W / max(rgba.shape[:2]), 0.42 * H / rgba.shape[0])
    rgba = cv2.resize(rgba, (max(8, int(rgba.shape[1] * s)), max(8, int(rgba.shape[0] * s))), interpolation=cv2.INTER_LINEAR)
    al = cv2.GaussianBlur(cv2.erode(rgba[:, :, 3], np.ones((3, 3), np.uint8)), (5, 5), 0); rgba[:, :, 3] = al
    belt = frame[int(0.42 * H):int(0.68 * H)].reshape(-1, 3).mean(0); objm = rgba[:, :, :3][al > 128].reshape(-1, 3).mean(0) if (al > 128).any() else belt
    gain = np.clip((belt.mean() / max(objm.mean(), 1)) ** 0.35, 0.6, 1.4) * rng.uniform(0.85, 1.15)
    rgba[:, :, :3] = np.clip(cv2.GaussianBlur(rgba[:, :, :3], (3, 3), 0.8).astype(np.float32) * gain, 0, 255).astype(np.uint8)
    h, w = rgba.shape[:2]
    for _ in range(40):
        cx, cy = rng.uniform(0.08, 0.92) * W, rng.uniform(0.42, 0.68) * H
        x0, y0 = int(cx - w / 2), int(cy - h / 2)
        if x0 < 0 or y0 < 0 or x0 + w > W or y0 + h > H:
            continue
        box = (x0, y0, x0 + w, y0 + h)
        if _free(box, taken):
            _blend(frame, rgba, x0, y0)
            ys, xs = np.where(al > 40)
            return (x0 + xs.min(), y0 + ys.min(), x0 + xs.max(), y0 + ys.max())
    return None


def paste_bottle(frame, src, sbox, taken, rng):
    """Tempel potongan botol asli (kotak + margin belt) dari foto belt lain, tepi persegi dihaluskan."""
    H, W = frame.shape[:2]; sh, sw = src.shape[:2]
    x1, y1, x2, y2 = sbox; mx, my = 0.06 * (x2 - x1), 0.10 * (y2 - y1)
    cx1, cy1 = max(0, int(x1 - mx)), max(0, int(y1 - my))
    c = src[cy1:min(sh, int(y2 + my)), cx1:min(sw, int(x2 + mx))]
    k = W / sw; c = cv2.resize(c, (max(8, int(c.shape[1] * k)), max(8, int(c.shape[0] * k))))
    h, w = c.shape[:2]
    bx1, by1, bx2, by2 = (x1 - cx1) * k, (y1 - cy1) * k, (x2 - cx1) * k, (y2 - cy1) * k      # kotak botol di koordinat potongan
    if rng.random() < 0.5:
        c = np.ascontiguousarray(c[:, ::-1]); bx1, bx2 = w - bx2, w - bx1
    if w >= W or h >= H:
        return None
    a = np.zeros((h, w), np.uint8); f = max(3, int(0.05 * min(h, w)))
    a[f:-f, f:-f] = 255; a = cv2.GaussianBlur(a, (2 * f + 1, 2 * f + 1), 0)
    rgba = np.dstack([c, a])
    for _ in range(40):
        x0, y0 = int(rng.uniform(0, W - w)), int(rng.uniform(0.40 * H, 0.70 * H) - h / 2)
        if y0 < 0 or y0 + h > H:
            continue
        if _free((x0, y0, x0 + w, y0 + h), taken):
            _blend(frame, rgba, x0, y0)
            return (int(x0 + bx1), int(y0 + by1), int(x0 + bx2), int(y0 + by2)), (x0, y0, x0 + w, y0 + h)
    return None


def yolo_line(cls, b, W, H):
    return f"{cls} {(b[0] + b[2]) / 2 / W:.6f} {(b[1] + b[3]) / 2 / H:.6f} {(b[2] - b[0]) / W:.6f} {(b[3] - b[1]) / H:.6f}"
