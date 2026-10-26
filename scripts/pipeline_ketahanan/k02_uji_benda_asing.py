"""Uji offline deteksi benda asing (rvm_kp/benda.py) pada foto belt 22 Okt.
  - non-botol (86 foto, `2026-10-22_nonbottle_v1`): HARUS ada benda di zona yang tidak tertutup kotak botol -> 'benda asing'
  - botol (64 foto air + 30 frame app): benda di zona HARUS tertutup kotak botol (bukan benda asing)
  - belt kosong (proksi): zona kiri x 3-22% di semua foto (benda selalu di tengah) -> tidak boleh ada benda
Detektor botol = model aktif ver9c (kelas 0). Keluaran: datasets/4_eksperimen/benda_asing/{laporan.txt, contoh_*.jpg}
  venv/Scripts/python.exe scripts/pipeline_ketahanan/k02_uji_benda_asing.py"""
import csv, glob, os, sys
from collections import Counter
import cv2, numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from rvm_kp.benda import Presence, outside
from ultralytics import YOLO

D = os.path.join(REPO, "datasets"); OUT = os.path.join(D, "4_eksperimen", "benda_asing"); os.makedirs(OUT, exist_ok=True)
det = YOLO(os.path.join(D, "3_model_final", "botol_ver9c.pt"))
FOREIGN = 0.5            # gumpalan dianggap benda asing bila > 50% kotaknya di luar kotak botol


def bottles(im):
    r = det.predict(im, imgsz=640, conf=0.35, verbose=False)[0]
    return [[float(v) for v in b.xyxy[0]] for b in r.boxes if int(b.cls) == 0], [[float(v) for v in b.xyxy[0]] for b in r.boxes if int(b.cls) == 1]


def judge(im, pb):
    bb, nb = bottles(im)
    blobs = pb.objects(im, zone=(0.25, 0.75))
    foreign = [b for b in blobs if outside(b, bb) > FOREIGN]
    empty = Presence(); empty.band = pb.band
    left = empty.objects(im, zone=(0.03, 0.22))
    return dict(n_botol=len(bb), n_nonbotol=len(nb), n_blob=len(blobs), asing=len(foreign) > 0, kosong_fp=len(left) > 0, blobs=blobs, bb=bb, foreign=foreign)


def main():
    sets = [("nonbotol", sorted(glob.glob(os.path.join(D, "1_foto_mentah", "2026-10-22_nonbottle_v1", "*.png")))),
            ("botol_air", sorted(glob.glob(os.path.join(D, "1_foto_mentah", "2026-10-22_water_v1", "*.png")))),
            ("botol_app", sorted(glob.glob(os.path.join(D, "1_foto_mentah", "2026-10-22_scan_app", "*_1.png"))))]
    benda = {r["file"]: r["benda"] for r in csv.DictReader(open(os.path.join(D, "1_foto_mentah", "2026-10-22_nonbottle_v1", "MANIFEST.csv"), encoding="utf-8"))}
    L = ["Deteksi benda asing (rvm_kp/benda.py) + detektor ver9c, foto belt 22 Okt. Benda asing = gumpalan di zona 25-75% yang >50% di luar kotak botol.", ""]
    miss = Counter(); tot = Counter(); samples = []
    for name, fs in sets:
        pb = Presence(); pb.calibrate(cv2.imread(fs[0]))                   # kalibrasi baris belt sekali per sesi
        R = []
        for f in fs:
            im = cv2.imread(f); j = judge(im, pb); R.append(j)
            if name == "nonbotol":
                k = benda.get(os.path.basename(f), "?").rsplit("-", 1)[0]
                tot[k] += 1; miss[k] += (not j["asing"]) and j["n_botol"] == 0
            if len(samples) < 12 and ((name == "nonbotol" and len(samples) % 2 == 0) or (name != "nonbotol" and len(samples) % 2 == 1)):
                samples.append((name, f, j, im))
        n = len(R)
        if name == "nonbotol":
            tangkap = sum(r["asing"] or r["n_botol"] > 0 for r in R)
            L.append(f"{name:10s} n={n:3d} | ditandai benda asing {sum(r['asing'] for r in R)}/{n} | dikira botol (ver9c) {sum(r['n_botol'] > 0 for r in R)}/{n} | "
                     f"TERTANGKAP (asing atau botol->diperiksa) {tangkap}/{n} | lolos tanpa reaksi {n - tangkap}/{n}")
        else:
            L.append(f"{name:10s} n={n:3d} | botol terdeteksi {sum(r['n_botol'] > 0 for r in R)}/{n} | SALAH ditandai benda asing {sum(r['asing'] for r in R)}/{n}")
        L.append(f"{'':10s}        belt kosong (zona kiri) salah ada benda {sum(r['kosong_fp'] for r in R)}/{n}; pita belt y={pb.band[0]:.2f}-{pb.band[1]:.2f}")
    L.append(""); L.append("Non-botol yang lolos tanpa reaksi, per jenis benda:")
    for k in sorted(tot):
        if miss[k]:
            L.append(f"   {k:30s} {miss[k]}/{tot[k]}")
    open(os.path.join(OUT, "laporan.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))
    tiles = []
    for name, f, j, im in samples:
        v = im.copy(); s = 0.33
        for b in j["bb"]:
            cv2.rectangle(v, tuple(map(int, b[:2])), tuple(map(int, b[2:4])), (0, 255, 0), 4)
        for b in j["blobs"]:
            cv2.rectangle(v, tuple(map(int, b[:2])), tuple(map(int, b[2:4])), (0, 0, 255) if b in j["foreign"] else (255, 200, 0), 3)
        v = cv2.resize(v, None, fx=s, fy=s) if v.shape[1] > 1000 else cv2.resize(v, None, fx=0.5, fy=0.5)
        v = cv2.resize(v, (640, 360)); cv2.putText(v, f"{name} asing={j['asing']}", (8, 26), 0, 0.8, (255, 255, 255), 2); tiles.append(v)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(OUT, "contoh.jpg"), np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]), [cv2.IMWRITE_JPEG_QUALITY, 80])


if __name__ == "__main__":
    main()
