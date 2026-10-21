"""Potret bebas (tanpa protokol level/tutup) untuk NEGATIF non_bottle, mis. plastik bening diremas, kantong, gelas plastik.
  python capture_free.py --cam 1 --name plastik-bening-remas [--out DIR]
Tombol: SPACE/s simpan | n ganti nama benda (ketik di terminal) | z undo | q keluar. 1920x1080 PNG, MANIFEST.csv."""
import argparse, csv, datetime, os, sys, time
import cv2

DS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=1)
ap.add_argument("--name", default="plastik-bening-remas")
ap.add_argument("--out", default=f"{DS}/1_foto_mentah/{datetime.date.today():%Y-%m-%d}_nonbottle_v1")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True); MAN = os.path.join(a.out, "MANIFEST.csv")
rows = list(csv.reader(open(MAN)))[1:] if os.path.exists(MAN) else []
cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); cap.set(3, 1920); cap.set(4, 1080); cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
for _ in range(15):
    cap.read(); time.sleep(0.03)
ok, f = cap.read()
if not ok or f.shape[:2] != (1080, 1920):
    sys.exit(f"STOP: kamera index {a.cam} memberi {None if not ok else f.shape[1::-1]}, bukan 1920x1080. Jalankan capture_water.py --list.")
name, msg = a.name, ""
WIN = "potret bebas [SPACE simpan | n nama | z undo | q keluar]"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
while True:
    ok, f = cap.read()
    if not ok:
        continue
    v = cv2.resize(f, (960, 540)); cv2.rectangle(v, (0, 0), (960, 40), (0, 0, 0), -1)
    cv2.putText(v, f"{name} | foto {len(rows)} | {msg}", (8, 27), 0, 0.6, (0, 255, 255), 2)
    cv2.imshow(WIN, v); k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), 27):
        break
    if k in (32, ord("s")):
        cap.read(); ok, f = cap.read()
        n = sum(1 for r in rows if r[1] == name) + 1; fn = f"{name}_{n:02d}.png"
        cv2.imwrite(os.path.join(a.out, fn), f, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        rows.append([fn, name, datetime.datetime.now().isoformat(timespec="seconds")]); msg = f"disimpan {fn}"
    elif k == ord("n"):
        name = input("nama benda baru: ").strip().replace(" ", "-") or name; msg = "nama diganti"
    elif k == ord("z") and rows:
        r = rows.pop()
        try:
            os.remove(os.path.join(a.out, r[0]))
        except OSError:
            pass
        msg = f"undo {r[0]}"
    if k in (32, ord("s"), ord("z")):
        with open(MAN, "w", newline="") as fh:
            w = csv.writer(fh); w.writerow(["file", "benda", "timestamp"]); w.writerows(rows)
cap.release(); cv2.destroyAllWindows(); print(len(rows), "foto di", a.out)
