"""Deteksi "ada benda" di zona periksa TANPA bergantung kelas detektor (untuk menangkap benda bukan botol).

Ide: area belt (dua sabuk hijau + celah gelap di tengah) hampir SERAGAM sepanjang arah gerak (kiri-kanan). Jadi warna
"belt kosong" untuk tiap baris = median warna baris itu sendiri (diambil di lebar tengah frame). Piksel di zona periksa
yang jauh dari warna barisnya = benda. Tidak perlu foto belt kosong, tahan terhadap perubahan cahaya dan belt yang bergerak.
Batas atas/bawah area belt dicari otomatis dari baris yang dominan hijau (sabuk), lalu disimpan (kalibrasi sekali per sesi).

  pb = Presence(); boxes = pb.objects(frame_bgr, zone=(0.25, 0.75))   # [(x1, y1, x2, y2, luas_frac), ...] piksel frame masukan"""
import cv2
import numpy as np

W = 640                       # lebar kerja (hemat CPU Nano)
DIFF_T = 28.0                 # jarak warna Lab dari warna baris -> dianggap benda
MIN_AREA = 0.004              # luas gumpalan minimal (fraksi luas frame) agar dianggap benda
ROW_X = (0.12, 0.88)          # lebar yang dipakai menghitung warna baris (hindari tepi rangka)


def green_mask(hsv):
    return (hsv[..., 0] >= 35) & (hsv[..., 0] <= 95) & (hsv[..., 1] >= 35) & (hsv[..., 2] >= 25)


class Presence(object):
    def __init__(self, diff_t=DIFF_T, min_area=MIN_AREA):
        self.diff_t, self.min_area = diff_t, min_area
        self.band = None                                   # (y0, y1) baris area belt, fraksi tinggi

    def calibrate(self, frame):
        """Cari baris area belt: dari baris hijau pertama sampai terakhir (sabuk atas, celah, sabuk bawah)."""
        sm = cv2.resize(frame, (W, round(frame.shape[0] * W / frame.shape[1])), interpolation=cv2.INTER_AREA)
        g = green_mask(cv2.cvtColor(sm, cv2.COLOR_BGR2HSV))
        x0, x1 = int(ROW_X[0] * W), int(ROW_X[1] * W)
        rows = np.where(g[:, x0:x1].mean(1) > 0.35)[0]
        if len(rows) < 10:
            return None
        h = sm.shape[0]
        self.band = (max(0, rows[0] - 2) / h, min(h, rows[-1] + 3) / h)
        return self.band

    def mask(self, frame):
        sm = cv2.resize(frame, (W, round(frame.shape[0] * W / frame.shape[1])), interpolation=cv2.INTER_AREA)
        if self.band is None and self.calibrate(frame) is None:
            return sm, None
        h = sm.shape[0]; y0, y1 = int(self.band[0] * h), int(self.band[1] * h)
        lab = cv2.cvtColor(cv2.GaussianBlur(sm, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
        x0, x1 = int(ROW_X[0] * W), int(ROW_X[1] * W)
        ref = np.median(lab[:, x0:x1], axis=1, keepdims=True)              # warna "kosong" tiap baris
        d = np.linalg.norm(lab - ref, axis=2)
        m = np.zeros(d.shape, np.uint8); m[y0:y1] = (d[y0:y1] > self.diff_t).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return sm, m

    def objects(self, frame, zone=(0.25, 0.75)):
        """Gumpalan benda yang PUSATNYA di zona (fraksi lebar). Koordinat dikembalikan dalam piksel frame masukan."""
        sm, m = self.mask(frame)
        if m is None:
            return []
        k = frame.shape[1] / W; h, w = m.shape
        n, lab, st, cen = cv2.connectedComponentsWithStats(m, 8)
        out = []
        for i in range(1, n):
            x, y, bw, bh, a = st[i]
            if a < self.min_area * h * w or not (zone[0] * w <= cen[i][0] <= zone[1] * w):
                continue
            out.append((x * k, y * k, (x + bw) * k, (y + bh) * k, a / float(h * w)))
        return out


def outside(blob, boxes, pad=0.12):
    """Fraksi kotak gumpalan yang TIDAK tertutup kotak botol mana pun (diperbesar pad). 0 = seluruhnya bagian botol."""
    x1, y1, x2, y2 = blob[:4]; area = max(1.0, (x2 - x1) * (y2 - y1)); cov = 0.0
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        ix = max(0.0, min(x2, b[2] + pad * bw) - max(x1, b[0] - pad * bw)); iy = max(0.0, min(y2, b[3] + pad * bh) - max(y1, b[1] - pad * bh))
        cov = max(cov, ix * iy / area)
    return 1.0 - cov


# ---------------------------------------------------------------------------------------------------------------
# Patokan BELT KOSONG per piksel (26 Okt). Cara yang tepat untuk kamera diam: pantulan cahaya & kemiringan sabuk ikut
# masuk patokan. Patokan = median beberapa frame tanpa benda (diambil saat sesi mulai), diperbarui pelan saat belt kosong.
REF_FRAMES = 15               # frame kosong untuk membangun patokan
REF_ALPHA = 0.03              # laju pembaruan patokan saat belt kosong (tahan perubahan cahaya pelan)
REF_DIFF_T = 22.0             # jarak warna Lab dari patokan -> benda
REF_MIN_AREA = 0.004          # luas gumpalan minimal (fraksi frame)


class BeltRef(object):
    """ref = BeltRef(); ref.feed_empty(frame) sampai ref.ready; lalu ref.objects(frame, zone) -> gumpalan benda."""

    def __init__(self, n=REF_FRAMES, diff_t=REF_DIFF_T, min_area=REF_MIN_AREA):
        self.n, self.diff_t, self.min_area = n, diff_t, min_area
        self.buf = []; self.ref = None; self.shape = None

    @property
    def ready(self):
        return self.ref is not None

    def _lab(self, frame):
        sm = cv2.resize(frame, (W, round(frame.shape[0] * W / frame.shape[1])), interpolation=cv2.INTER_AREA)
        return sm, cv2.cvtColor(cv2.GaussianBlur(sm, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)

    def reset(self):
        self.buf = []; self.ref = None

    def feed_empty(self, frame):
        """Frame yang diyakini kosong (tanpa kotak detektor apa pun). Membangun lalu memperbarui patokan."""
        sm, lab = self._lab(frame)
        if self.ref is None:
            if self.shape is not None and lab.shape != self.shape:
                self.buf = []
            self.shape = lab.shape; self.buf.append(lab)
            if len(self.buf) >= self.n:
                self.ref = np.median(np.stack(self.buf), axis=0).astype(np.float32); self.buf = []
                self.ref_bgr = sm.copy()
        else:
            self.ref += REF_ALPHA * (lab - self.ref)

    def mask(self, frame):
        sm, lab = self._lab(frame)
        if self.ref is None or lab.shape != self.ref.shape:
            return sm, None
        d = np.linalg.norm(lab - self.ref, axis=2)
        m = (d > self.diff_t).astype(np.uint8) * 255
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
        return sm, m

    def objects(self, frame, zone=(0.25, 0.75), mask=None):
        """Gumpalan (x1, y1, x2, y2, luas_frac) berpusat di zona, koordinat frame masukan."""
        if mask is None:
            _, mask = self.mask(frame)
        if mask is None:
            return []
        k = frame.shape[1] / W; h, w = mask.shape
        n, _, st, cen = cv2.connectedComponentsWithStats(mask, 8)
        out = []
        for i in range(1, n):
            x, y, bw, bh, a = st[i]
            if a >= self.min_area * h * w and zone[0] * w <= cen[i][0] <= zone[1] * w:
                out.append((x * k, y * k, (x + bw) * k, (y + bh) * k, a / float(h * w)))
        return out
