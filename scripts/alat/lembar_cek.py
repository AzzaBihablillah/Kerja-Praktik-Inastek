"""Lembar cek bernomor dari frame diam aplikasi kiosk (stop-and-scan), untuk dilabeli user dari HP dalam beberapa menit.
  python scripts/alat/lembar_cek.py --dir 1_foto_mentah/2026-10-23_scan_app [--sesi 14 15]
Tiap botol: potongan frame diam pertama + keputusan mesin (hijau terima / merah tolak), skor cairan, ukuran.
Keluaran: <dir>/LEMBAR-CEK.jpg (dan LEMBAR-CEK-2.jpg dst. bila > 32 botol). Label user ditulis ke kolom label_manual SCAN.csv
(oleh Claude, dari jawaban chat "#9 air bening, #13 kosong"; nomor = urutan baris SCAN.csv yang tampil)."""
import argparse, csv, os, sys
import cv2, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from rvm_kp import config
from ultralytics import YOLO

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); ap.add_argument("--sesi", nargs="*", default=None)
a = ap.parse_args()
D = a.dir if os.path.isabs(a.dir) else os.path.join(REPO, "datasets", a.dir)
det = YOLO(config.MODEL_SETS["pytorch"]["bottle"])
R = list(csv.DictReader(open(f"{D}/SCAN.csv")))
if a.sesi:
    R = [r for r in R if r["sesi"] in a.sesi]
AL = {"": "DITERIMA", "tutup": "TOLAK: bertutup", "tutup_cairan": "TOLAK: bertutup+berisi", "cairan": "TOLAK: berisi", "leher": "TOLAK: leher?"}
tiles = []
for i, r in enumerate(R, 1):
    f = f"{D}/s{r['sesi']}_b{r['botol']}_{r['waktu']}_1.png"
    if not os.path.exists(f):
        continue
    im = cv2.imread(f)
    bb = [x.xyxy[0].tolist() for x in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(x.cls) == 0]
    if bb:
        x1, y1, x2, y2 = [int(v) for v in max(bb, key=lambda z: (z[2] - z[0]) * (z[3] - z[1]))]; m = 30
        im = im[max(0, y1 - m):y2 + m, max(0, x1 - m):x2 + m]
    im = cv2.resize(im, (420, int(420 * im.shape[0] / im.shape[1])))
    im = cv2.copyMakeBorder(im, 52, max(0, 170 - im.shape[0]), 0, 0, cv2.BORDER_CONSTANT)[:222]
    col = (60, 200, 60) if r["keputusan"] == "terima" else (60, 60, 230)
    cv2.rectangle(im, (0, 0), (420, 50), col, -1)
    cv2.putText(im, f"#{i}  {r['waktu'][:2]}:{r['waktu'][2:4]}:{r['waktu'][4:]}  sesi {r['sesi']}", (6, 20), 0, 0.6, (255, 255, 255), 2)
    cv2.putText(im, f"{AL.get(r['alasan'], r['alasan'])}  cairan {float(r['p_cairan']):.2f}  {r['ukuran']}", (6, 43), 0, 0.55, (255, 255, 255), 2)
    tiles.append(im)
for s in range(0, len(tiles), 32):
    ch = tiles[s:s + 32]
    while len(ch) % 4:
        ch.append(np.zeros_like(tiles[0]))
    out = f"{D}/LEMBAR-CEK{'' if s == 0 else '-' + str(s // 32 + 1)}.jpg"
    cv2.imwrite(out, np.vstack([np.hstack(ch[j:j + 4]) for j in range(0, len(ch), 4)]), [cv2.IMWRITE_JPEG_QUALITY, 80])
    print(out)
