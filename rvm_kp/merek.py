"""Model merek (bonus): coca_cola (semua produk Coca-Cola Company) vs non_coca_cola. YOLOv8n-cls 256.
Input = kotak botol dari detektor botol, dipotong dengan `brand_crop` yang SAMA dengan saat latih
(scripts/pipeline_merek: sumbu panjang mendatar, tidak dilipat, letterbox abu 114). Hasil CV: docs/CATATAN-PERCOBAAN-TA.md §5."""
import cv2
import numpy as np

S = 256


def brand_crop(im, x1, y1, x2, y2, pad=0.04, size=S):
    H, W = im.shape[:2]; w, h = x2 - x1, y2 - y1
    x1, x2 = max(0, int(x1 - pad * w)), min(W, int(x2 + pad * w)); y1, y2 = max(0, int(y1 - pad * h)), min(H, int(y2 + pad * h))
    c = im[y1:y2, x1:x2]
    if c.shape[0] > c.shape[1]:
        c = cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)
    s = size / max(c.shape[:2]); c = cv2.resize(c, (max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 114, np.uint8); oy, ox = (size - c.shape[0]) // 2, (size - c.shape[1]) // 2
    out[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return out


class Merek(object):
    """p_coca(frames, box): rata-rata peluang coca_cola pada beberapa frame diam (BGR) untuk satu kotak botol."""

    def __init__(self, path, thr=0.5):
        from ultralytics import YOLO
        self.m = YOLO(path, task="classify"); self.thr = thr
        self.ci = [i for i, n in self.m.names.items() if n == "coca_cola"][0]

    def p_coca(self, frames, box):
        crops = [brand_crop(f, *box) for f in frames]
        if not crops:
            return None
        res = self.m.predict(crops, imgsz=S, verbose=False)
        return float(np.mean([float(r.probs.data[self.ci]) for r in res]))
