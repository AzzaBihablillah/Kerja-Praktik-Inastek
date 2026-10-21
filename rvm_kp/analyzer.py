"""Analisis satu frame RVM (KP PT Inastek): botol -> tutup -> cairan berwarna -> kategori ukuran.

Satu kode untuk semua perangkat; backend ditentukan oleh FILE model yang dipakai (Ultralytics memuat semuanya dengan API sama):
  *.pt                -> PyTorch (CUDA bila ada; laptop RTX 4050)
  *.engine            -> TensorRT (Jetson Nano; dibuat DI perangkat target)
  *_openvino_model/   -> OpenVINO (PC/mini-PC Intel tanpa GPU NVIDIA)
  *.onnx              -> ONNX Runtime
Profil (config.PROFILES) menentukan resolusi kerja, mode deteksi tutup ("full" = seluruh frame, "roi" = hanya area tiap botol), dan FP16.

Lingkup (keputusan mitra): cairan BERWARNA saja; air bening di luar lingkup. Ambang cairan perlu dikalibrasi saat pemasangan.
Kompatibel Python 3.8 (Docker JetPack 4 di Jetson Nano).
"""
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from ultralytics import YOLO

from . import config


def fold_crop(im, x1, y1, x2, y2, pad=0.04, size=224):
    """Crop kotak botol -> sumbu panjang horizontal -> potong dua & tumpuk -> pad abu 114 ke size x size (sama dgn data latih)."""
    H, W = im.shape[:2]; w, h = x2 - x1, y2 - y1
    x1, x2 = max(0, int(x1 - pad * w)), min(W, int(x2 + pad * w)); y1, y2 = max(0, int(y1 - pad * h)), min(H, int(y2 + pad * h))
    c = im[y1:y2, x1:x2]
    if c.shape[0] > c.shape[1]:
        c = cv2.rotate(c, cv2.ROTATE_90_CLOCKWISE)
    hw = c.shape[1] // 2; c = cv2.vconcat([c[:, :hw], c[:, hw:hw * 2]])
    ch, cw = c.shape[:2]; s = size / max(ch, cw)
    c = cv2.resize(c, (max(1, round(cw * s)), max(1, round(ch * s))), interpolation=cv2.INTER_AREA)
    out = np.full((size, size, 3), 114, np.uint8); oy, ox = (size - c.shape[0]) // 2, (size - c.shape[1]) // 2
    out[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
    return out


def roi_canvas(im, x1, y1, x2, y2, imgsz, scale, margin):
    """Potong area botol (+margin), skala `scale` (dibatasi agar muat), tempel di tengah kanvas abu imgsz x imgsz.
    Mengembalikan kanvas dan fungsi pemeta kotak kanvas -> koordinat frame. Skala dijaga tetap (tidak diperbesar memenuhi kanvas)
    agar ukuran tutup di mata model sama dengan saat latih/uji full-frame."""
    H, W = im.shape[:2]; w, h = x2 - x1, y2 - y1
    rx1, ry1 = max(0, int(x1 - margin * w)), max(0, int(y1 - margin * h))
    rx2, ry2 = min(W, int(x2 + margin * w)), min(H, int(y2 + margin * h))
    c = im[ry1:ry2, rx1:rx2]
    s = min(scale, imgsz / max(c.shape[0], c.shape[1]))
    c = cv2.resize(c, (max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))),
                   interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    out = np.full((imgsz, imgsz, 3), 114, np.uint8); oy, ox = (imgsz - c.shape[0]) // 2, (imgsz - c.shape[1]) // 2
    out[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c

    def to_frame(b):
        return ((b[0] - ox) / s + rx1, (b[1] - oy) / s + ry1, (b[2] - ox) / s + rx1, (b[3] - oy) / s + ry1)
    return out, to_frame


@dataclass
class BottleResult:
    box: tuple                     # x1, y1, x2, y2 (piksel frame MASUKAN)
    conf: float                    # keyakinan botol
    cap: str                       # "tutup" | "tanpa_tutup" | "?" (tidak terdeteksi)
    cap_conf: float                # keyakinan deteksi tutup/leher (0 bila "?")
    p_liquid: float                # probabilitas cairan berwarna
    has_liquid: bool               # p_liquid >= ambang
    size: str                      # "kecil" | "tanggung" | "besar" | "?" (terpotong tepi frame)
    area_1920: float               # luas kotak setara frame 1920 px (untuk kalibrasi ukuran)
    timings_ms: dict = field(default_factory=dict)


class RVMAnalyzer:
    """an = RVMAnalyzer("tensorrt", profile="nano"); hasil = an.analyze(frame_bgr); an.last_timings = waktu per tahap (ms)."""

    def __init__(self, backend="pytorch", models=None, liquid_thr=config.LIQUID_THR, device=None, profile=None,
                 cap_mode=None, half=None, frame_width=None, roi_imgsz=None, roi_scale=None):
        p = dict(config.PROFILES[profile or config.DEFAULT_PROFILE])
        self.cap_mode = cap_mode or p["cap_mode"]
        self.frame_width = frame_width or p["frame_width"]
        self.roi_imgsz = roi_imgsz or config.CAP_ROI_IMGSZ
        self.roi_scale = roi_scale or config.CAP_ROI_SCALE
        m = dict(config.MODEL_SETS[backend]) if models is None else models
        self.backend = backend
        # FP16 hanya berarti di PyTorch+CUDA saat inferensi; engine TensorRT sudah FP16 sejak dibuat (--half); ONNX/OpenVINO diabaikan.
        self.half = bool(p["half"] if half is None else half) and backend == "pytorch"
        if p.get("cpu_threads"):
            cv2.setNumThreads(p["cpu_threads"])
            try:
                import torch
                torch.set_num_threads(p["cpu_threads"])
            except Exception:
                pass
        self.bottle = YOLO(m["bottle"], task="detect")
        self.cap = YOLO(m["cap_roi"] if self.cap_mode == "roi" else m["cap"], task="detect")
        self.liquid = YOLO(m["liquid"], task="classify")
        self.liquid_thr, self.device = liquid_thr, device
        self._cb = None
        self.last_timings = {}

    def _pred(self, model, src, imgsz, conf=None):
        kw = dict(imgsz=imgsz, verbose=False)
        if conf is not None:
            kw["conf"] = conf
        if self.device is not None:
            kw["device"] = self.device
        if self.half:
            kw["half"] = True
        return model.predict(src, **kw)[0]

    @staticmethod
    def _best_cap(boxes):
        if not len(boxes):
            return "?", 0.0, None
        b = max(boxes, key=lambda z: float(z.conf))
        return ["tutup", "tanpa_tutup"][int(b.cls)], float(b.conf), [float(v) for v in b.xyxy[0]]

    def analyze(self, frame):
        T = dict(pra=0.0, botol=0.0, tutup=0.0, cairan=0.0)
        t = time.perf_counter()
        H0, W0 = frame.shape[:2]; k = 1.0
        if W0 > self.frame_width:                                   # tiru kamera beresolusi lebih rendah
            k = self.frame_width / W0
            frame = cv2.resize(frame, (self.frame_width, round(H0 * k)), interpolation=cv2.INTER_AREA)
        H, W = frame.shape[:2]
        T["pra"] = (time.perf_counter() - t) * 1e3

        t = time.perf_counter(); rb = self._pred(self.bottle, frame, config.IMGSZ_BOTTLE, config.CONF_BOTTLE)
        T["botol"] = (time.perf_counter() - t) * 1e3
        rc = None
        if self.cap_mode == "full":
            t = time.perf_counter(); rc = self._pred(self.cap, frame, config.IMGSZ_CAP_FULL, config.CONF_CAP)
            T["tutup"] += (time.perf_counter() - t) * 1e3

        out = []
        for b in rb.boxes:
            if int(b.cls) != 0:
                continue
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
            t = time.perf_counter()
            if self.cap_mode == "roi":
                cv, to_frame = roi_canvas(frame, x1, y1, x2, y2, self.roi_imgsz, self.roi_scale * 1920 / W, config.CAP_ROI_MARGIN)
                cap, cconf, _ = self._best_cap(self._pred(self.cap, cv, self.roi_imgsz, config.CONF_CAP).boxes)
            else:                                                   # cara lama: tutup full-frame dicocokkan lewat posisi x
                caps = [c for c in rc.boxes if x1 - 20 <= float(c.xyxy[0][0] + c.xyxy[0][2]) / 2 <= x2 + 20]
                cap, cconf, _ = self._best_cap(caps)
            T["tutup"] += (time.perf_counter() - t) * 1e3

            t = time.perf_counter()
            pl = self._pred(self.liquid, fold_crop(frame, x1, y1, x2, y2), config.IMGSZ_LIQUID)
            if self._cb is None:
                self._cb = [i for i, v in pl.names.items() if v == "berwarna"][0]
            p = float(pl.probs.data[self._cb]); T["cairan"] += (time.perf_counter() - t) * 1e3

            clipped = x1 < 4 or x2 > W - 4
            area = (x2 - x1) * (y2 - y1) * (1920 / W) ** 2
            size = "?" if clipped else ("kecil" if area < config.AREA_KECIL_TANGGUNG else ("tanggung" if area < config.AREA_TANGGUNG_BESAR else "besar"))
            out.append(BottleResult((x1 / k, y1 / k, x2 / k, y2 / k), float(b.conf), cap, cconf, p, p >= self.liquid_thr, size, area))
        T["total"] = sum(T.values())
        for r in out:
            r.timings_ms = T
        self.last_timings = T
        return out
