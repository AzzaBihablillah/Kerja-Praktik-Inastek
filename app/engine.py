"""Mesin RVM: thread kamera/analisis + status sesi. Satu-satunya tempat yang memegang kamera dan model.

Sumber & analis:
  nyata    : rvm_kp.kamera.Kamera + rvm_kp.RVMAnalyzer(backend, profile="nano")
  simulasi : app.simulator.Simulator (frame & hasil tiruan)
Analisis hanya berjalan saat sesi aktif (hemat Jetson Nano saat mesin menganggur)."""
import importlib.util
import os
import threading
import time
import traceback

import cv2

from . import config
from .tracker import Tracker, decide, iou, center

LIME = (50, 230, 196)       # BGR, sama dengan --lime di CSS
CORAL = (64, 96, 232)
INK = (34, 36, 26)


def load_rvm_config():
    """Baca rvm_kp/config.py TANPA mengimpor paket rvm_kp (yang langsung memuat Ultralytics)."""
    p = os.path.join(config.REPO, "rvm_kp", "config.py")
    spec = importlib.util.spec_from_file_location("rvm_kp_config_ro", p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def rc_imgsz(an, which):
    """Ukuran input model rvm_kp (dibaca dari modul config milik analyzer)."""
    import sys
    c = sys.modules[type(an).__module__.rsplit(".", 1)[0] + ".config"]
    return {"bottle": c.IMGSZ_BOTTLE, "cap": c.IMGSZ_CAP_FULL, "liquid": c.IMGSZ_LIQUID}[which]


class Machine:
    def __init__(self, db, conveyor, emit, sim=False, backend=config.BACKEND, profile=config.PROFILE, cam=config.CAMERA_INDEX):
        self.db, self.conveyor, self.emit = db, conveyor, emit
        self.sim, self.backend, self.profile, self.cam_index = sim, backend, profile, cam
        rc = load_rvm_config()
        self.liquid_thr = rc.LIQUID_THR
        self.tracker = Tracker()
        self.lock = threading.RLock()
        self.src = self.an = None
        self.status = "memuat"              # memuat | siap | galat_kamera | galat_model
        self.status_msg = ""
        self.fps = 0.0
        self.jpeg = b""
        self._stop = threading.Event()
        self._last_stamp = ""
        self.an2 = None; self.brand2 = None
        self.session = None                 # dict saat sesi aktif
        self.last_summary = None
        self._t = threading.Thread(target=self._run, name="rvm-engine", daemon=True)

    # ------------------------------------------------------------------ siklus hidup
    def start(self):
        self._t.start()

    def stop(self):
        self._stop.set(); self._t.join(timeout=3)
        if self.session:
            self.end_session("server_berhenti")
        try:
            if self.src:
                self.src.close()
        except Exception:
            pass

    def _open(self):
        if self.sim:
            from .simulator import Simulator
            self.src = self.an = Simulator(liquid_thr=self.liquid_thr)
            return
        if self.an is None:
            self.status, self.status_msg = "memuat", "memuat model %s (profil %s)..." % (self.backend, self.profile)
            self._push_state()
            from rvm_kp import RVMAnalyzer                 # impor berat (Ultralytics) hanya di mode nyata
            self.an = RVMAnalyzer(self.backend, profile=self.profile)
            self._warmup()
            if config.SHADOW and self.an2 is None:
                if self.backend != "pytorch":
                    print("[kandidat] dilewati: model kandidat bayangan hanya untuk backend pytorch")
                else:
                    import sys as _sys
                    rc = _sys.modules[type(self.an).__module__.rsplit(".", 1)[0] + ".config"]
                    ms = dict(rc.MODEL_SETS["pytorch"], bottle=config.KANDIDAT["bottle"], liquid=config.KANDIDAT["liquid"])
                    self.an2 = RVMAnalyzer("pytorch", models=ms, profile=self.profile, size_rule=config.KANDIDAT["size_rule"])
                    print("[kandidat] aktif (bayangan):", os.path.basename(ms["bottle"]), os.path.basename(ms["liquid"]), "ukuran", config.KANDIDAT["size_rule"])
                    if config.KANDIDAT.get("brand") and os.path.exists(config.KANDIDAT["brand"]):
                        from rvm_kp.merek import Merek
                        self.brand2 = Merek(config.KANDIDAT["brand"])
                        print("[kandidat] merek (bayangan):", os.path.basename(config.KANDIDAT["brand"]))
        from rvm_kp.kamera import Kamera
        self.src = Kamera(self.cam_index, config.CAMERA_W, config.CAMERA_H)

    def _warmup(self):
        """Inferensi pertama (CUDA/TensorRT) lambat beberapa detik: lakukan saat memuat, bukan di botol pertama pengguna."""
        import numpy as np
        try:
            an = self.an
            an._pred(an.bottle, np.full((720, 1280, 3), 114, np.uint8), rc_imgsz(an, "bottle"))
            an._pred(an.cap, np.full((an.roi_imgsz, an.roi_imgsz, 3), 114, np.uint8), an.roi_imgsz if an.cap_mode == "roi" else rc_imgsz(an, "cap"))
            an._pred(an.liquid, np.full((224, 224, 3), 114, np.uint8), rc_imgsz(an, "liquid"))
        except Exception as e:                          # pemanasan gagal tidak menghentikan mesin
            print("[engine] pemanasan model dilewati:", e)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._open()
                self.status, self.status_msg = "siap", ""
                self._push_state()
                self._loop()
            except Exception as e:
                traceback.print_exc()
                self.status = "galat_model" if self.an is None and not self.sim else "galat_kamera"
                self.status_msg = str(e)[:200]
                if self.session is not None:            # kamera putus di tengah sesi: tutup sesi agar poin tetap bisa diklaim
                    self.end_session("galat_kamera")
                self._push_state()
                try:
                    if self.src and self.src is not self.an:
                        self.src.close()
                except Exception:
                    pass
                self.src = None
                if self.status == "galat_model":
                    return                                  # model gagal dimuat: tidak dicoba ulang terus-menerus
                self._stop.wait(3.0)

    def _loop(self):
        t_prev = time.time(); t_prev_preview = 0.0; t_tick = 0.0
        while not self._stop.is_set():
            frame = self.src.read()
            now = time.time()
            with self.lock:
                active = self.session is not None
            results = []
            if active:
                results = self.an.analyze(frame)
                done = self.tracker.update(results, frame.shape[1], now)
                if config.STOP_SCAN and not self.sim:
                    t = self._stop_candidate(frame.shape[1], now)
                    if t is not None:
                        b = self._stop_and_scan(t)
                        if b is not None:
                            with self.lock:
                                if self.session is not None:
                                    self.session["last_seen"] = time.time(); self._record(b)
                        now = time.time()
                with self.lock:
                    if self.session is not None:
                        if results:
                            self.session["last_seen"] = now
                        for t in done:
                            self._record(decide(t, self.liquid_thr))
            dt = now - t_prev; t_prev = now
            self.fps = 0.8 * self.fps + 0.2 * (1.0 / dt if dt > 0 else 0)
            if now - t_prev_preview >= 1.0 / (config.PREVIEW_FPS if active else 4):
                self._preview(frame, active); t_prev_preview = now
            if active and now - t_tick >= 1.0:
                t_tick = now
                with self.lock:
                    s = self.session
                    if s is not None:
                        left = config.IDLE_TIMEOUT - (now - s["last_seen"])
                        if left <= 0:
                            self.end_session("idle")
                        else:
                            self.emit(dict(type="tick", idle_left=int(left + 0.99), idle_max=config.IDLE_TIMEOUT,
                                           fps=round(self.fps, 1), ms=round(getattr(self.an, "last_timings", {}).get("total", 0))))
            if not active and not self.sim:
                time.sleep(0.05)

    # ------------------------------------------------------------------ berhenti-lalu-pindai
    def _stop_candidate(self, width, now):
        """Botol (terlihat di frame ini) yang belum dipindai dan pusatnya masuk zona pemicu."""
        z0, z1 = config.STOP_ZONE_X[0] * width, config.STOP_ZONE_X[1] * width
        for t in self.tracker.tracks:
            if t.decided or getattr(t, "scanned", False) or t.last < now:
                continue
            if z0 <= center(t.box)[0] <= z1:
                return t
        return None

    def _stop_and_scan(self, t):
        """Matikan konveyor, tunggu diam, analisis STOP_FRAMES frame diam, putuskan botol t dari frame diam saja, jalankan lagi.
        Kembalikan hasil keputusan (dict) atau None bila botol tidak terlihat saat diam (lanjut voting biasa)."""
        t.scanned = True; t0 = time.time()
        plc_ok = self.conveyor.run(False) if self.conveyor.enabled else None
        self._stop.wait(config.STOP_SETTLE_SEC)
        votes = []; frames = []
        for _ in range(config.STOP_FRAMES):
            if self._stop.is_set() or self.session is None or time.time() - t0 > config.STOP_MAX_SEC:
                break
            f = self.src.read(); res = self.an.analyze(f); frames.append(f)
            if res:
                r = max(res, key=lambda r: iou(r.box, t.box))
                size = max(t.box[2] - t.box[0], t.box[3] - t.box[1])
                (cx, cy), (tx, ty) = center(r.box), center(t.box)
                if iou(r.box, t.box) >= 0.2 or ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 < 0.5 * size:
                    votes.append(r); t.box, t.cur, t.last = r.box, r, time.time()
            self._preview(f, True)
        t_scan = time.time() - t0
        if self.conveyor.enabled and self.session is not None:      # sesi sudah ditutup saat memindai: jangan jalankan lagi
            if not self.conveyor.run(True):
                print("[scan] PERINGATAN: perintah konveyor JALAN gagal (PLC). Konveyor mungkin tetap berhenti.")
        print("[scan] botol #%d: berhenti %.2f s, %d frame diam, PLC berhenti=%s" % (t.id, t_scan, len(votes), plc_ok), flush=True)
        if not votes:
            return None
        t.votes = [(r.cap, r.cap_conf, r.p_liquid, r.size, r.area_1920) for r in votes]
        t.decided = True
        b = decide(t, self.liquid_thr); b["mode"] = "berhenti"; b["stop_sec"] = round(t_scan, 2)
        self._save_scan(t, b, frames, votes)
        if self.an2 is not None:
            self._shadow(t, frames, plc_ok)
        return b

    def _shadow(self, t, frames, plc_ok):
        """Model kandidat menilai frame diam yang sama; hasil hanya dicatat (SCAN_KANDIDAT.csv), tidak memengaruhi layar/poin."""
        try:
            import csv, datetime
            vv = []
            for f in frames:
                res = self.an2.analyze(f)
                if res:
                    r = max(res, key=lambda r: iou(r.box, t.box))
                    if iou(r.box, t.box) >= 0.2:
                        vv.append(r)

            class _T(object):
                pass
            tt = _T(); tt.id = t.id; tt.votes = [(r.cap, r.cap_conf, r.p_liquid, r.size, r.area_1920) for r in vv]
            k = decide(tt, self.liquid_thr) if vv else dict(decision="tidak_terdeteksi", reason="", p_liquid="", size="")
            pc = ""                                       # merek: kotak botol model AKTIF (seperti data latih: ver9c), rata-rata frame diam
            if self.brand2 is not None:
                try:
                    pc = round(self.brand2.p_coca(frames, t.box), 3)
                except Exception as e:
                    print("[kandidat] merek gagal:", e)
            if not config.SCAN_SAVE_DIR:
                return
            d = config.SCAN_SAVE_DIR.replace("{tanggal}", datetime.date.today().isoformat()); os.makedirs(d, exist_ok=True)
            p = os.path.join(d, "SCAN_KANDIDAT.csv"); new = not os.path.exists(p)
            with open(p, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["waktu", "sesi", "botol", "keputusan_kandidat", "alasan_kandidat", "p_cairan_kandidat", "p_per_frame",
                                "ukuran_kandidat", "plc_berhenti_ok", "p_coca_kandidat", "merek_kandidat"])
                w.writerow([self._last_stamp, self.session["id"] if self.session else 0, t.id, k["decision"], k.get("reason") or "",
                            k["p_liquid"], " ".join("%.2f" % r.p_liquid for r in vv), k["size"], plc_ok, pc,
                            "" if pc == "" else ("coca_cola" if pc >= self.brand2.thr else "non_coca_cola")])
        except Exception as e:
            print("[kandidat] gagal:", e)

    def _save_scan(self, t, b, frames, votes):
        """Simpan frame diam (PNG) + baris CSV: dipakai untuk memeriksa salah keputusan & sebagai data belt nyata."""
        if not config.SCAN_SAVE_DIR:
            return
        try:
            import csv, datetime
            d = config.SCAN_SAVE_DIR.replace("{tanggal}", datetime.date.today().isoformat()); os.makedirs(d, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%H%M%S")
            self._last_stamp = stamp
            sid = self.session["id"] if self.session else 0
            for i, f in enumerate(frames):
                cv2.imwrite(os.path.join(d, "s%d_b%d_%s_%d.png" % (sid, t.id, stamp, i + 1)), f, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            p = os.path.join(d, "SCAN.csv"); new = not os.path.exists(p)
            with open(p, "a", newline="") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["waktu", "sesi", "botol", "keputusan", "alasan", "p_cairan", "p_per_frame", "tutup_suara", "ukuran", "jeda_s", "berhenti_s", "label_manual"])
                w.writerow([stamp, sid, t.id, b["decision"], b.get("reason") or "", b["p_liquid"], " ".join("%.2f" % r.p_liquid for r in votes),
                            b["cap_votes"], b["size"], config.STOP_SETTLE_SEC, b["stop_sec"], ""])
        except Exception as e:
            print("[scan] simpan gagal:", e)

    def _preview(self, frame, active):
        f = frame
        h, w = f.shape[:2]
        if active:
            f = frame.copy()
            z0, z1 = int(config.ZONE_X[0] * w), int(config.ZONE_X[1] * w)
            for x in (z0, z1):
                cv2.line(f, (x, 0), (x, h), (200, 210, 200), 2, cv2.LINE_AA)
            for t in self.tracker.tracks:
                r = t.cur; x1, y1, x2, y2 = [int(v) for v in r.box]
                bad = r.cap == "tutup" or r.p_liquid >= self.liquid_thr
                col = CORAL if bad else LIME
                cv2.rectangle(f, (x1, y1), (x2, y2), col, 4, cv2.LINE_AA)
                lab = "#%d %s %s %s" % (t.id, {"tutup": "TUTUP", "tanpa_tutup": "TANPA TUTUP", "?": "LEHER ?"}[r.cap],
                                        "ISI" if r.p_liquid >= self.liquid_thr else "KOSONG", r.size.upper())
                (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(f, (x1, max(0, y1 - th - 16)), (x1 + tw + 16, max(th + 16, y1)), col, -1)
                cv2.putText(f, lab, (x1 + 8, max(th + 8, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, INK, 2, cv2.LINE_AA)
        if w > config.PREVIEW_WIDTH:
            f = cv2.resize(f, (config.PREVIEW_WIDTH, round(h * config.PREVIEW_WIDTH / w)), interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 72])
        if ok:
            self.jpeg = enc.tobytes()

    # ------------------------------------------------------------------ sesi
    def _record(self, b):
        s = self.session
        self.db.bottle_add(s["id"], b)
        s["bottles"].append(b)
        if b["decision"] == "terima":
            s["valid"] += 1; s["points"] += b["points"]; s["grams"] += b["grams"]
        else:
            s["invalid"] += 1
        self.emit(dict(type="bottle", bottle=b, session=self.session_public()))

    def session_public(self):
        s = self.session
        if s is None:
            return None
        return dict(id=s["id"], started=s["started"], points=s["points"], valid=s["valid"], invalid=s["invalid"], grams=s["grams"],
                    bottles=s["bottles"][-12:])

    def start_session(self):
        with self.lock:
            if self.status != "siap":
                return None, "Mesin belum siap: %s" % (self.status_msg or self.status)
            if self.session is not None:
                return None, "Sesi sudah berjalan."
            sid = self.db.session_start(); now = time.time()
            self.session = dict(id=sid, started=now, last_seen=now, points=0, valid=0, invalid=0, grams=0, bottles=[])
            self.tracker.reset()
            self.last_summary = None
        self.conveyor.run(True)
        self.emit(dict(type="session_start", session=self.session_public(), idle_max=config.IDLE_TIMEOUT))
        return self.session_public(), None

    def end_session(self, reason="manual"):
        with self.lock:
            if self.session is None:
                return None
            for t in self.tracker.flush():
                self._record(decide(t, self.liquid_thr))
            s = self.session; self.session = None
        self.conveyor.run(False)
        row, claim = self.db.session_end(s["id"], reason)
        self.last_summary = dict(session=dict(id=s["id"], points=row["points"], valid=row["valid"], invalid=row["invalid"],
                                              grams=row["grams"], durasi=int(time.time() - s["started"]), bottles=s["bottles"]),
                                 claim=claim, reason=reason, ts=time.time())
        self.emit(dict(type="session_end", **self.last_summary))
        return self.last_summary

    def _push_state(self):
        self.emit(dict(type="machine", **self.machine_status()))

    def machine_status(self):
        return dict(status=self.status, pesan=self.status_msg, sim=self.sim, backend="simulasi" if self.sim else self.backend,
                    profil=self.profile, kamera=self.cam_index, fps=round(self.fps, 1), ambang_cairan=self.liquid_thr,
                    sesi_aktif=self.session is not None, auto=getattr(self.an, "auto", None) if self.sim else None)
