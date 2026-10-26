"""Mode simulasi: belt virtual + hasil analisis tiruan (tanpa kamera, model, atau PLC).
Untuk mencoba antarmuka, alur poin, dan pelacak di laptop mana pun. Keluaran meniru rvm_kp.BottleResult (atribut sama)."""
import random
import threading
import time
from collections import namedtuple

import cv2
import numpy as np

SimResult = namedtuple("SimResult", "box conf cap cap_conf p_liquid has_liquid size area_1920")
BENDA = (300, 230)                 # benda bukan botol (gumpalan plastik/kardus tiruan)

W, H = 1280, 720
BOX = {"kecil": (330, 210), "tanggung": (440, 260), "besar": (580, 310)}      # px pada frame 1280 (luas ~ batas rvm_kp)
LIQ_COLORS = [(40, 60, 120), (30, 120, 220), (60, 160, 60), (150, 80, 200)]   # BGR: cola, oranye, hijau, ungu


class SimBottle:
    def __init__(self, size, cap, liquid, benda=False):
        self.size, self.cap, self.liquid, self.benda = size, cap, liquid, benda
        self.w, self.h = BENDA if benda else BOX[size]
        self.x = -self.w - 10.0
        self.y = H / 2 - self.h / 2 + random.uniform(-60, 60)
        self.color = random.choice(LIQ_COLORS)
        self.neck_left = random.random() < 0.5


class Simulator:
    """read() -> frame BGR; analyze(frame) -> list SimResult. Kecepatan belt & fps meniru Nano (lambat)."""

    def __init__(self, fps=8, speed=300, liquid_thr=0.5):
        self.fps, self.speed, self.liquid_thr = fps, speed, liquid_thr
        self.bottles = []; self.lock = threading.Lock(); self.t_last = time.time(); self.offset = 0.0
        self.auto = False; self._next_auto = 0.0
        self.last_timings = {}; self.last_nonbottle = []

    def spawn(self, size=None, cap=False, liquid=False, benda=False):
        with self.lock:
            self.bottles.append(SimBottle(size or random.choice(list(BOX)), bool(cap), bool(liquid), bool(benda)))

    def _auto(self, now):
        if self.auto and now >= self._next_auto:
            r = random.random()
            self.spawn(random.choice(list(BOX)), cap=r < 0.15, liquid=0.15 <= r < 0.27, benda=0.27 <= r < 0.35)
            self._next_auto = now + random.uniform(2.5, 5.0)

    def read(self, timeout=2.0):
        time.sleep(max(0.0, 1.0 / self.fps - (time.time() - self.t_last)))
        now = time.time(); dt = now - self.t_last; self.t_last = now
        self._auto(now)
        with self.lock:
            for b in self.bottles:
                b.x += self.speed * dt
            self.bottles = [b for b in self.bottles if b.x < W + 40]
            self.offset = (self.offset + self.speed * dt) % 80
            return self._draw()

    def _draw(self):
        f = np.full((H, W, 3), (46, 50, 44), np.uint8)
        cv2.rectangle(f, (0, 110), (W, H - 110), (62, 66, 60), -1)
        for x in np.arange(-80 + self.offset, W, 80):
            cv2.line(f, (int(x), 110), (int(x), H - 110), (70, 75, 68), 2)
        for b in self.bottles:
            x1, y1, x2, y2 = int(b.x), int(b.y), int(b.x + b.w), int(b.y + b.h)
            if b.benda:
                cv2.ellipse(f, ((x1 + x2) // 2, (y1 + y2) // 2), (b.w // 2, b.h // 2), 15, 0, 360, (150, 120, 90), -1)
                cv2.putText(f, "BENDA", (x1 + 60, (y1 + y2) // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (60, 50, 40), 2)
                continue
            body = (int(b.x + (0.18 if b.neck_left else 0) * b.w), y1 + 12, int(b.x + (1 if b.neck_left else 0.82) * b.w), y2 - 12)
            cv2.rectangle(f, body[:2], body[2:], (205, 214, 210), -1)
            if b.liquid:
                cv2.rectangle(f, (body[0] + 8, body[1] + 10), (body[2] - 8, body[3] - 8), b.color, -1)
            cv2.rectangle(f, (body[0] + int(0.3 * (body[2] - body[0])), body[1]), (body[0] + int(0.62 * (body[2] - body[0])), body[3]),
                          (60, 60, 200), -1)                                   # label
            nx1, nx2 = (x1, body[0]) if b.neck_left else (body[2], x2)
            cy = (y1 + y2) // 2
            cv2.rectangle(f, (nx1 + (14 if b.neck_left else 0), cy - 26), (nx2 - (0 if b.neck_left else 14), cy + 26), (205, 214, 210), -1)
            if b.cap:
                cx1 = nx1 if b.neck_left else nx2 - 18
                cv2.rectangle(f, (cx1, cy - 30), (cx1 + 18, cy + 30), (30, 30, 190), -1)
        cv2.putText(f, "SIMULASI", (24, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (120, 130, 118), 2)
        return f

    def analyze(self, frame):
        t = time.perf_counter(); out = []; nb = []
        with self.lock:
            for b in self.bottles:
                x1, x2 = max(0.0, b.x), min(float(W), b.x + b.w)
                if x2 - x1 < 0.35 * b.w:
                    continue
                y1, y2 = b.y, b.y + b.h
                if b.benda:                                        # kelas non_bottle (seperti detektor ver9c)
                    if random.random() < 0.9:
                        nb.append(SimResult((x1, y1, x2, y2), 0.8, "?", 0.0, 0.0, False, "?", (x2 - x1) * (y2 - y1) * (1920 / W) ** 2))
                    continue
                r = random.random()
                truth = "tutup" if b.cap else "tanpa_tutup"
                cap = truth if r < 0.86 else ("?" if r < 0.94 else ("tanpa_tutup" if b.cap else "tutup"))
                p = float(np.clip(random.gauss(0.9 if b.liquid else 0.07, 0.08), 0, 1))
                clipped = x1 < 4 or x2 > W - 4
                area = (x2 - x1) * (y2 - y1) * (1920 / W) ** 2
                out.append(SimResult((x1, y1, x2, y2), 0.9, cap, 0.0 if cap == "?" else 0.8, p, p >= self.liquid_thr,
                                     "?" if clipped else b.size, area))
        self.last_nonbottle = nb
        ms = (time.perf_counter() - t) * 1e3
        self.last_timings = dict(pra=0.0, botol=ms, tutup=0.0, cairan=0.0, total=ms)
        return out

    def close(self):
        pass
