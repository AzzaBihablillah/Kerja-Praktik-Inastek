"""Pembuka kamera belt (Logitech C922) yang sama untuk Windows (laptop) dan Linux (Jetson Nano).
  from rvm_kp.kamera import Kamera
  with Kamera(0, 1280, 720) as cam:  frame = cam.read()
Windows: DirectShow; Linux/Jetson: V4L2 (/dev/videoN). Format MJPG agar 720p/1080p 30 fps lewat USB.
Pembacaan di thread terpisah: read() selalu memberi frame TERBARU (frame lama dibuang), seperti VideoCapture di open_vino/."""
import platform, threading, time
import cv2


def buka(index=0, width=1280, height=720, fps=30):
    api = cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_V4L2
    cap = cv2.VideoCapture(index, api)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height); cap.set(cv2.CAP_PROP_FPS, fps)
    if not cap.isOpened():
        raise RuntimeError(f"kamera {index} tidak bisa dibuka")
    ok, f = cap.read()
    if not ok or f is None:
        raise RuntimeError(f"kamera {index} tidak memberi frame")
    if f.shape[1] != width:
        print(f"[kamera] PERINGATAN: minta {width}x{height}, dapat {f.shape[1]}x{f.shape[0]} (cabut-colok kamera / cek v4l2-ctl --list-formats-ext)")
    return cap


class Kamera:
    def __init__(self, index=0, width=1280, height=720, fps=30):
        self.cap = buka(index, width, height, fps); self._f = None; self._ok = True
        self._lock = threading.Lock(); self._run = True
        self._t = threading.Thread(target=self._loop, daemon=True); self._t.start()

    def _loop(self):
        while self._run:
            ok, f = self.cap.read()
            with self._lock:
                self._ok, self._f = ok, (f if ok else self._f)

    def read(self, timeout=2.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self._lock:
                if self._f is not None:
                    f, self._f = self._f, None
                    return f
            time.sleep(0.002)
        raise RuntimeError("kamera tidak memberi frame baru (timeout)")

    def close(self):
        self._run = False; self._t.join(timeout=1); self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
