"""Pewarnaan ulang TUTUP buatan (bukan cincin) untuk memutus jalan pintas 'tutup sewarna cincin' pada model tutup.
Kotak label = leher+tutup (kelas cap) atau leher+cincin (no_cap). Arah tutup = menjauhi pusat badan botol (kotak botol ver9c).
Bagian kotak paling luar (FRAKSI_TUTUP sepanjang arah leher) = tutup -> warnanya diganti (hue acak + saturasi acak, termasuk putih),
tepi dilembutkan. Kelas no_cap: tidak diwarnai ulang di bagian tutup (tidak ada tutup), hanya cincin+leher digeser warnanya agar
warna cincin juga bervariasi."""
import numpy as np, cv2

FRAKSI_TUTUP = 0.45


def arah(neck, bottle):
    nx, ny = (neck[0] + neck[2]) / 2, (neck[1] + neck[3]) / 2
    bx, by = (bottle[0] + bottle[2]) / 2, (bottle[1] + bottle[3]) / 2
    v = np.array([nx - bx, ny - by], float); n = np.linalg.norm(v)
    return v / n if n > 1 else np.array([1.0, 0.0])


def mask_tutup(shape, neck, u, frac=FRAKSI_TUTUP, feather=0.08):
    """Mask lembut (0..1): bagian kotak leher yang paling jauh dari badan botol."""
    H, W = shape[:2]; x1, y1, x2, y2 = [int(round(v)) for v in neck]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    m = np.zeros((H, W), np.float32)
    if x2 <= x1 or y2 <= y1:
        return m
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    t = (xx - cx) * u[0] + (yy - cy) * u[1]
    L = abs((x2 - x1) * u[0]) + abs((y2 - y1) * u[1])          # panjang kotak sepanjang arah leher
    t0 = L / 2 - frac * L                                         # batas tutup
    f = max(1.0, feather * L)
    m[y1:y2, x1:x2] = np.clip((t - t0) / f + 0.5, 0, 1)
    return m


def warnai(img, mask, rng):
    """Ganti warna area mask: 1/4 putih/abu, sisanya hue acak dengan saturasi kuat."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    if rng.random() < 0.25:
        s2 = s * rng.uniform(0.0, 0.2); v2 = np.clip(v * rng.uniform(1.05, 1.3) + 20, 0, 255); h2 = h
    else:
        h2 = np.full_like(h, rng.uniform(0, 180)); s2 = np.clip(np.maximum(s, 0) * 0.3 + rng.uniform(120, 230), 0, 255)
        v2 = np.clip(v * rng.uniform(0.7, 1.15), 0, 255)
    new = cv2.cvtColor(np.stack([h2, s2, v2], -1).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    a = mask[..., None]
    return (img.astype(np.float32) * (1 - a) + new * a).astype(np.uint8)


def geser_kotak(img, neck, rng, pad=0.15):
    """Geser hue seluruh kotak leher (cincin ikut) - untuk no_cap & sebagian cap."""
    H, W = img.shape[:2]; x1, y1, x2, y2 = neck; w, h = x2 - x1, y2 - y1
    x1, y1, x2, y2 = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h)), int(min(W, x2 + pad * w)), int(min(H, y2 + pad * h))
    out = img.copy(); roi = cv2.cvtColor(out[y1:y2, x1:x2], cv2.COLOR_BGR2HSV).astype(np.int32)
    roi[..., 0] = (roi[..., 0] + int(rng.uniform(20, 160))) % 180
    out[y1:y2, x1:x2] = cv2.cvtColor(roi.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return out


def mask_benda(img, neck, k=3, thr=18.0, soft=10.0, pad=0.6):
    """Mask lembut piksel BENDA (tutup/cincin) di dalam kotak leher: warna latar (belt + plastik bening) dipelajari dari
    sekeliling kotak leher (k-means Lab), piksel yang mirip latar tidak diwarnai -> tidak ada 'kotak berwarna' buatan."""
    H, W = img.shape[:2]; x1, y1, x2, y2 = neck; w, h = x2 - x1, y2 - y1
    X1, Y1, X2, Y2 = int(max(0, x1 - pad * w)), int(max(0, y1 - pad * h)), int(min(W, x2 + pad * w)), int(min(H, y2 + pad * h))
    lab = cv2.cvtColor(img[Y1:Y2, X1:X2], cv2.COLOR_BGR2LAB).astype(np.float32)
    inner = np.zeros(lab.shape[:2], bool)
    inner[int(y1) - Y1:int(y2) - Y1, int(x1) - X1:int(x2) - X1] = True
    bg = lab[~inner].reshape(-1, 3)
    m = np.zeros((H, W), np.float32)
    if len(bg) < 50:
        m[int(y1):int(y2), int(x1):int(x2)] = 1; return m
    bg = bg[np.random.default_rng(0).choice(len(bg), min(4000, len(bg)), replace=False)]
    _, _, C = cv2.kmeans(bg, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0), 2, cv2.KMEANS_PP_CENTERS)
    d = np.min(np.linalg.norm(lab[..., None, :] - C[None, None], axis=-1), axis=-1)
    f = np.clip((d - thr) / soft, 0, 1); f[~inner] = 0
    m[Y1:Y2, X1:X2] = cv2.GaussianBlur(f, (0, 0), 1.0)
    return m


def geser_benda(img, mask, rng):
    """Geser hue hanya pada piksel benda (mask) - untuk cincin/leher."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
    hsv[..., 0] = (hsv[..., 0] + int(rng.uniform(20, 160))) % 180
    new = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    a = mask[..., None]
    return (img.astype(np.float32) * (1 - a) + new * a).astype(np.uint8)


# ---- versi 2 (30 Okt, setelah audit user: tutup separuh berwarna, salah ujung, warna belang) ----
def mask_tutup2(img, neck, bottle, mb, thr=22.0):
    """Mask tutup yang MENGIKUTI WARNA tutup asli (bukan garis lurus): warna tutup diambil dari ujung terluar kotak leher
    (sisi menjauhi badan botol), lalu semua piksel benda yang warnanya mirip & tidak di 30% kotak terdalam ikut diwarnai.
    Mengembalikan (mask, alasan_lewati) - alasan_lewati != None berarti foto ini tidak layak diwarnai."""
    H, W = img.shape[:2]; x1, y1, x2, y2 = [int(round(v)) for v in neck]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None, "kecil"
    # arah: kotak leher harus di ujung kotak botol (sepertiga terluar), kalau tidak arah tidak yakin
    u = arah(neck, bottle); bw, bh = bottle[2] - bottle[0], bottle[3] - bottle[1]
    nx, ny = (neck[0] + neck[2]) / 2, (neck[1] + neck[3]) / 2; bx, by = (bottle[0] + bottle[2]) / 2, (bottle[1] + bottle[3]) / 2
    half = abs(bw / 2 * u[0]) + abs(bh / 2 * u[1])
    if half <= 0 or ((nx - bx) * u[0] + (ny - by) * u[1]) / half < 0.55:
        return None, "arah"
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    t = (xx - (x1 + x2) / 2) * u[0] + (yy - (y1 + y2) / 2) * u[1]
    L = abs((x2 - x1) * u[0]) + abs((y2 - y1) * u[1]); tn = t / L + 0.5            # 0 = sisi badan, 1 = ujung luar
    obj = mb[y1:y2, x1:x2] > 0.5
    lab = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.float32)
    outer = obj & (tn > 0.8)
    if outer.sum() < 15:
        return None, "ujung_kosong"
    ref = np.median(lab[outer], axis=0)
    if ref[0] < 255 * 0.22:
        return None, "gelap"
    cand = (tn > 0.3) & (tn < 1.01)
    sim = np.linalg.norm(lab - ref, axis=-1) < thr
    m = (sim & cand & obj).astype(np.float32)
    # tutup harus padat: di bagian terluar, sebagian besar piksel benda mirip warna tutup (bukan belang)
    if m[outer].mean() < 0.55:
        return None, "belang"
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    full = np.zeros((H, W), np.float32); full[y1:y2, x1:x2] = cv2.GaussianBlur(m, (0, 0), 0.8)
    return full, None


def mask_tutup3(img, neck, bottle, mb):
    """Versi 3: kemiripan pada KROMA (a,b) saja, tahan bayangan/kilap; tutup putih/abu (kroma rendah) dikenali dari kroma rendah
    + terang; lalu komponen yang menyentuh ujung luar diambil, lubang ditutup."""
    H, W = img.shape[:2]; x1, y1, x2, y2 = [int(round(v)) for v in neck]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if x2 - x1 < 6 or y2 - y1 < 6:
        return None, "kecil"
    u = arah(neck, bottle); bw, bh = bottle[2] - bottle[0], bottle[3] - bottle[1]
    nx, ny = (neck[0] + neck[2]) / 2, (neck[1] + neck[3]) / 2; bx, by = (bottle[0] + bottle[2]) / 2, (bottle[1] + bottle[3]) / 2
    half = abs(bw / 2 * u[0]) + abs(bh / 2 * u[1])
    if half <= 0 or ((nx - bx) * u[0] + (ny - by) * u[1]) / half < 0.55:
        return None, "arah"
    yy, xx = np.mgrid[y1:y2, x1:x2].astype(np.float32)
    t = (xx - (x1 + x2) / 2) * u[0] + (yy - (y1 + y2) / 2) * u[1]
    L = abs((x2 - x1) * u[0]) + abs((y2 - y1) * u[1]); tn = t / L + 0.5
    obj = mb[y1:y2, x1:x2] > 0.3
    lab = cv2.cvtColor(img[y1:y2, x1:x2], cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[..., 1:] - 128; chroma = np.linalg.norm(ab, axis=-1)
    outer = obj & (tn > 0.75) & (tn < 0.97)
    if outer.sum() < 10:
        return None, "ujung_kosong"
    rL = np.median(lab[..., 0][outer]); rab = np.median(ab[outer], axis=0); rc = np.linalg.norm(rab)
    if rL < 35 or (rL < 60 and rc < 12):                   # hitam / abu gelap: tidak diwarnai (biru tua TETAP diwarnai)
        return None, "gelap"
    if rc < 12:                                             # tutup putih/abu
        sim = (chroma < 16) & (lab[..., 0] > rL - 55)
    else:
        sim = np.linalg.norm(ab - rab, axis=-1) < max(10.0, 0.35 * rc)
    cand = (sim & obj & (tn > 0.25)).astype(np.uint8)
    k = max(3, int(0.08 * min(x2 - x1, y2 - y1)) | 1)
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    n, lbl = cv2.connectedComponents(cand)
    keep = np.zeros_like(cand)
    for i in range(1, n):
        comp = lbl == i
        tv = tn[comp]
        # komponen tutup: menyentuh bagian luar DAN memanjang ke dalam (bukan potongan benda tipis di tepi kotak)
        if (comp & (tn > 0.75)).sum() >= 8 and tv.min() < 0.6 and tv.max() - tv.min() > 0.3:
            keep |= comp.astype(np.uint8)
    if keep.sum() == 0:
        return None, "ujung_kosong"
    # tutup lubang
    ff = keep.copy(); hh, ww = ff.shape; fm = np.zeros((hh + 2, ww + 2), np.uint8)
    inv = (1 - ff).astype(np.uint8); cv2.floodFill(inv, fm, (0, 0), 0)
    keep = keep | inv
    if keep[outer].mean() < 0.7:
        return None, "belang"
    full = np.zeros((H, W), np.float32); full[y1:y2, x1:x2] = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), 0.8)
    return full, None


def warnai2(img, mask, rng):
    """Ganti warna dengan mempertahankan KECERAHAN asli (bayangan/kilap tetap): hue & saturasi baru, V asli
    (putih: saturasi ~0, V dinaikkan sedikit)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)[..., 0]
    if rng.random() < 0.25:
        h2, s2, v2 = h, s * 0.1, np.clip(lab * 1.15 + 25, 0, 255)
    else:
        h2 = np.full_like(h, rng.uniform(0, 180)); s2 = np.full_like(s, rng.uniform(140, 235)); v2 = np.clip(lab * rng.uniform(0.95, 1.2) + 15, 0, 255)
    new = cv2.cvtColor(np.stack([h2, s2, v2], -1).astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    a = mask[..., None]
    return (img.astype(np.float32) * (1 - a) + new * a).astype(np.uint8)
