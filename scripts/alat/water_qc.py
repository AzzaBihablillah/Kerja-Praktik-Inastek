"""Pemeriksaan otomatis satu foto botol rebah di belt: botol terdeteksi? tutup (model cap)? leher ke kiri/kanan?

Dipakai oleh capture_water.py (peringatan saat memotret) dan qc_water_set.py (audit folder).
Model cap TIDAK sempurna (tutup hitam di botol hitam sering lolos), jadi hasilnya PERINGATAN, bukan kebenaran.
"""
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = os.path.join(REPO, "datasets", "3_model_final")
BOTTLE_PT = os.path.join(MODELS, "botol_ver8b.pt")              # sejak 21 Okt: ver8b (ver7 luput pada botol berisi)
CAP_PT = os.path.join(MODELS, "cap_belt_final.pt")
CAP_NAMES = ["on", "off"]          # kelas 0 = cap (tutup terpasang), 1 = no_cap (lepas)

_models = {}


def _load():
    if not _models:
        from ultralytics import YOLO
        _models["bottle"] = YOLO(BOTTLE_PT)
        _models["cap"] = YOLO(CAP_PT)
    return _models


def analyze(frame, bottle_conf=0.15, cap_conf=0.25):
    """Kembalikan dict: bottle_box/bottle_conf (None kalau tidak ada), cap_state ('on'|'off'|None), cap_conf,
    neck_auto ('kanan'|'kiri'|None; dari posisi kotak tutup/leher relatif ke tengah botol)."""
    m = _load()
    out = dict(bottle_box=None, bottle_conf=0.0, cap_state=None, cap_conf=0.0, neck_auto=None)
    r = m["bottle"].predict(frame, imgsz=640, conf=bottle_conf, verbose=False)[0]
    bb = [b for b in r.boxes if int(b.cls) == 0]
    if bb:
        b = max(bb, key=lambda z: float(z.conf))
        out["bottle_box"] = [float(v) for v in b.xyxy[0]]
        out["bottle_conf"] = float(b.conf)
    r = m["cap"].predict(frame, imgsz=640, conf=cap_conf, verbose=False)[0]
    if len(r.boxes):
        c = max(r.boxes, key=lambda z: float(z.conf))
        out["cap_state"] = CAP_NAMES[int(c.cls)]
        out["cap_conf"] = float(c.conf)
        if out["bottle_box"] is not None:
            cx = float(c.xyxy[0][0] + c.xyxy[0][2]) / 2
            bx = (out["bottle_box"][0] + out["bottle_box"][2]) / 2
            out["neck_auto"] = "kanan" if cx > bx else "kiri"
    return out


def warnings(res, cap_label, neck_expected=None, strong=0.6):
    """Daftar peringatan (teks pendek) + tingkat: 0 aman, 1 periksa, 2 kuat (bunyikan alarm)."""
    w, lvl = [], 0
    if res["bottle_box"] is None:
        w.append("BOTOL TIDAK TERDETEKSI"); lvl = max(lvl, 1)
    if res["cap_state"] is None:
        w.append("model cap tidak melihat leher/tutup"); lvl = max(lvl, 1)
    elif res["cap_state"] != cap_label:
        w.append(f"MODEL: tutup {res['cap_state'].upper()} ({res['cap_conf']:.2f}) tapi label {cap_label.upper()}")
        lvl = max(lvl, 2 if res["cap_conf"] >= strong else 1)
    if neck_expected and res["neck_auto"] and res["neck_auto"] != neck_expected:
        w.append(f"leher terlihat di {res['neck_auto'].upper()}, diminta {neck_expected.upper()}"); lvl = max(lvl, 1)
    return w, lvl
