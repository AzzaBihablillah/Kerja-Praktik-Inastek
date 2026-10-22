"""Demo LIVE prototipe (tanpa PLC, tanpa menyimpan apa pun): botol (ver8b) -> tutup (cap_belt_final) -> cairan berwarna (classifier crop)
-> kategori ukuran (luas kotak). Untuk MENCOBA model sebelum mengambil data baru dan melihat warna/botol mana yang gagal.

  ../venv/Scripts/python.exe live_liquid_demo.py --cam 1                 # kamera C922 1080p
  ../venv/Scripts/python.exe live_liquid_demo.py --image foto.png        # satu foto
Tombol: [ / ]  turunkan / naikkan ambang cairan (0.05)  |  m  ganti classifier (RH -> RA -> RH+RA)  |  q keluar
Model cairan (21 Okt): RH = resep utama (geser hue), RA = tanpa geser hue, RH+RA = rata-rata keduanya.
Batasan (jujur): ambang belum dikalibrasi (model condong menyebut 'kosong'); ukuran dari luas kotak hanya valid di 1080p,
posisi kamera sama dengan 13/16 Okt, botol utuh di frame; air bening di luar lingkup.
"""
import argparse, os, sys, time
import cv2, numpy as np
from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "scripts", "pipeline_cairan"))
MODELS = os.path.join(REPO, "datasets", "3_model_final")
from common import fold_crop  # noqa: E402

DET = os.path.join(MODELS, "botol_ver9c.pt")      # 22 Okt: ver9c (dulu ver8b)
CLS = {"RH": os.path.join(MODELS, "cairan_RH.pt"), "RA": os.path.join(MODELS, "cairan_RA.pt")}
MODES = ["RH", "RA", "RH+RA"]
CAP = os.path.join(MODELS, "cap_belt_v2.pt")     # 22 Okt: v2 (dulu cap_belt_final)
AREA_KECIL_TANGGUNG, AREA_TANGGUNG_BESAR = 228_000, 357_000      # px^2 pada 1920x1080; kecil <=350 ml, tanggung 351-800, besar >800 (keputusan 20 Okt)

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=1)
ap.add_argument("--image")
ap.add_argument("--thr", type=float, default=0.5)
a = ap.parse_args()

det, cap_m = YOLO(DET), YOLO(CAP)
cls = {k: YOLO(v) for k, v in CLS.items()}
which, thr = "RH", a.thr


def size_cat(area, W):
    s = (1920 / W) ** 2                                                  # skala bila resolusi bukan 1080p (perkiraan kasar)
    area *= s
    return "kecil" if area < AREA_KECIL_TANGGUNG else ("tanggung" if area < AREA_TANGGUNG_BESAR else "besar")


def analyze(frame):
    H, W = frame.shape[:2]; out = frame.copy()
    r = det.predict(frame, imgsz=640, conf=0.35, verbose=False)[0]
    capr = cap_m.predict(frame, imgsz=640, conf=0.25, verbose=False)[0]
    for b in r.boxes:
        x1, y1, x2, y2 = b.xyxy[0].tolist()
        if int(b.cls) != 0:
            cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (128, 128, 128), 2)
            cv2.putText(out, f"non_bottle {float(b.conf):.2f}", (int(x1), int(y1) - 8), 0, 0.8, (128, 128, 128), 2); continue
        crop = fold_crop(frame, x1, y1, x2, y2); ps = []
        for key in (["RH", "RA"] if which == "RH+RA" else [which]):
            c = cls[key]; cb = [i for i, v in c.names.items() if v == "berwarna"][0]
            ps.append(float(c.predict(crop, imgsz=224, verbose=False)[0].probs.data[cb]))
        p = sum(ps) / len(ps)
        caps = [cb_ for cb_ in capr.boxes if x1 - 20 <= float(cb_.xyxy[0][0] + cb_.xyxy[0][2]) / 2 <= x2 + 20]
        tutup = (["TUTUP", "tanpa tutup"][int(max(caps, key=lambda z: float(z.conf)).cls)] if caps else "tutup ?")
        clipped = x1 < 4 or x2 > W - 4
        ukuran = "ukuran ? (terpotong)" if clipped else size_cat((x2 - x1) * (y2 - y1), W)
        berwarna = p >= thr
        col = (0, 0, 255) if berwarna else (0, 200, 0)
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, 3)
        lines = [f"botol {float(b.conf):.2f} | {tutup} | {ukuran}", f"cairan berwarna p={p:.2f} -> {'BERISI BERWARNA' if berwarna else 'kosong/bening'}"]
        for i, t in enumerate(lines):
            cv2.putText(out, t, (int(x1), max(30, int(y1) - 40 + 30 * i)), 0, 0.85, col, 2)
    cv2.rectangle(out, (0, H - 40), (W, H), (0, 0, 0), -1)
    cv2.putText(out, f"classifier {which} | ambang {thr:.2f}  ([ ] ubah, m ganti, q keluar) | prototipe: ambang belum dikalibrasi",
                (10, H - 12), 0, 0.8, (255, 255, 255), 2)
    return out


WIN = "demo prototipe cairan berwarna"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
if a.image:
    frame = cv2.imread(a.image)
    while True:
        cv2.imshow(WIN, analyze(frame)); k = cv2.waitKey(0) & 0xFF
        if k in (ord("q"), 27):
            break
        thr = min(0.95, thr + 0.05) if k == ord("]") else (max(0.05, thr - 0.05) if k == ord("[") else thr)
        which = MODES[(MODES.index(which) + 1) % len(MODES)] if k == ord("m") else which
    sys.exit(0)
capdev = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)
capdev.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); capdev.set(3, 1920); capdev.set(4, 1080); capdev.set(cv2.CAP_PROP_BUFFERSIZE, 1)
for _ in range(10):
    capdev.read(); time.sleep(0.03)
while True:
    ok, frame = capdev.read()
    if not ok:
        continue
    cv2.imshow(WIN, analyze(frame)); k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), 27):
        break
    thr = min(0.95, thr + 0.05) if k == ord("]") else (max(0.05, thr - 0.05) if k == ord("[") else thr)
    which = MODES[(MODES.index(which) + 1) % len(MODES)] if k == ord("m") else which
capdev.release(); cv2.destroyAllWindows()
