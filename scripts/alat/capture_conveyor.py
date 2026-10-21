"""Ambil foto dataset dari kamera USB (C922) di konveyor, satu tombol satu foto.

Pakai:  python capture_conveyor.py [--cam 1] [--res 640x480] [--lock] [--out DIR]

Tombol (klik jendela preview dulu):
  SPACE / s  simpan frame (mentah, tanpa overlay)
  c          status = cap        x   status = no_cap
  b          botol berikutnya (dialog nama merek, boleh kosong)
  v          kembali ke botol sebelumnya (kalau salah pencet b)
  e          simpan frame belt kosong (tanpa botol)
  z          undo (hapus foto terakhir)
  l          kunci / buka kunci fokus + exposure + white balance
  q / Esc    keluar

Nama file: bNN_<merek>_<cap|nocap>_PP.png  + MANIFEST.csv (resume otomatis).
Default: 1920x1080 MJPG (maks C922), PNG lossless = arsip kualitas tertinggi; turunkan nanti.
Pola per botol: 3 foto cap + 3 foto no_cap, tutup di 1/3 -> tengah -> 2/3 lebar frame.
NB server.py sekarang membaca 640x480 (FOV beda dari 16:9) -> lihat catatan di chat.
"""
import argparse, csv, os, sys, time, datetime, re, collections
import cv2

DS = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets"

ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=1)
ap.add_argument("--res", default="1920x1080", help="C922 maks = 1920x1080 (MJPG)")
ap.add_argument("--fmt", default="png", choices=["png", "jpg"],
                help="png = lossless (default, arsip kualitas tertinggi); jpg = q100")
ap.add_argument("--lock", action="store_true", help="kunci fokus/exposure/WB dari awal")
ap.add_argument("--out", default=f"{DS}/1_foto_mentah/{datetime.date.today():%Y-%m-%d}_conveyor_v1")
ap.add_argument("--selftest", action="store_true", help="simpan 1 frame lalu keluar (tanpa jendela)")
a = ap.parse_args()
W, H = map(int, a.res.lower().split("x"))
os.makedirs(a.out, exist_ok=True)
MAN = os.path.join(a.out, "MANIFEST.csv")

rows = []                                   # [file, bottle, brand, state, pos, ts]
if os.path.exists(MAN):
    with open(MAN, newline="") as f:
        rows = list(csv.reader(f))[1:]
bottle = max([int(r[1]) for r in rows if r[1].isdigit()] + [0]) + (0 if not rows else 0)
bottle = max(bottle, 1)
brand = ""
for r in reversed(rows):
    if r[1].isdigit() and int(r[1]) == bottle:
        brand = r[2]; break
state = "cap"
counts = collections.Counter((r[1], r[3]) for r in rows)
brands = {int(r[1]): r[2] for r in rows if r[1].isdigit() and int(r[1]) > 0 and r[2]}
if counts[(str(bottle), "cap")] >= 3 and counts[(str(bottle), "no_cap")] < 3:
    state = "no_cap"                                        # lanjut ke status yang belum selesai
msg = f"lanjut dari {len(rows)} foto" if rows else "folder baru"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def ask_brand(n):
    """Kotak dialog nama merek (tkinter); fallback ke terminal, lalu kosong."""
    try:
        import tkinter as tk
        from tkinter import simpledialog
        r = tk.Tk(); r.withdraw(); r.attributes("-topmost", True)
        s = simpledialog.askstring("Botol baru", f"Nama merek botol {n:02d} (boleh kosong):", parent=r)
        r.destroy()
        return (s or "").strip()
    except Exception:
        try:
            return input(f"nama merek botol {n:02d} (boleh kosong): ").strip()
        except (EOFError, OSError):
            return ""


def write_manifest():
    with open(MAN, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "bottle", "brand", "state", "pos", "timestamp"])
        w.writerows(rows)


cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)
if not cap.isOpened():
    sys.exit(f"kamera index {a.cam} tidak bisa dibuka")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))   # wajib utk 1080p lancar di DSHOW
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
locked = False
print("resolusi aktual:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x", int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), flush=True)

POSNAME = {1: "1/3", 2: "TENGAH", 3: "2/3"}     # posisi tutup di lebar frame, urut foto ke-1,2,3


def set_lock(on):
    global locked
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0 if on else 1)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25 if on else 0.75)   # DSHOW: 0.25 manual, 0.75 auto
    cap.set(cv2.CAP_PROP_AUTO_WB, 0 if on else 1)
    locked = on
    print("LOCK", "ON" if on else "OFF",
          f"focus={cap.get(cv2.CAP_PROP_FOCUS)} exp={cap.get(cv2.CAP_PROP_EXPOSURE)} "
          f"wb={cap.get(cv2.CAP_PROP_WB_TEMPERATURE)}", flush=True)


for _ in range(15):
    cap.read(); time.sleep(0.03)
if a.lock:
    set_lock(True)


def grab():
    cap.read()                               # buang frame basi di buffer
    ok, f = cap.read()
    return f if ok else None


def save(frame, name, b, br, st, pos):
    path = os.path.join(a.out, name)
    if name.endswith(".png"):
        cv2.imwrite(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    rows.append([name, b, br, st, pos, datetime.datetime.now().isoformat(timespec="seconds")])
    write_manifest()


if a.selftest:
    f = grab()
    if f is None:
        sys.exit("selftest: tidak ada frame")
    save(f, f"selftest.{a.fmt}", "0", "test", "empty", 1)
    print("selftest OK", f.shape, "->", a.out)
    cap.release(); sys.exit(0)

WIN = "capture conveyor   [SPACE simpan | c/x status | b botol baru | e kosong | z undo | l lock | q keluar]"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
if not rows:
    brand = ask_brand(bottle)                                  # merek botol pertama
while True:
    ok, frame = cap.read()
    if not ok:
        continue
    fh, fw = frame.shape[:2]
    view = cv2.resize(frame, (960, int(960 * fh / fw)))        # preview kecil; file tersimpan tetap resolusi penuh
    h, w = view.shape[:2]
    for fx in (0.35, 0.65):                                   # zona aktif server.py (abu-abu)
        cv2.line(view, (int(w * fx), 0), (int(w * fx), h), (200, 200, 200), 1)
    n_here = counts[(str(bottle), state)]
    tgt = POSNAME.get(n_here + 1)
    if tgt:                                                   # garis target posisi tutup (kuning)
        tx = int(w * {"1/3": 1 / 3, "TENGAH": 0.5, "2/3": 2 / 3}[tgt])
        cv2.line(view, (tx, 54), (tx, h), (0, 255, 255), 2)
    col = (0, 200, 0) if state == "cap" else (0, 0, 255)
    cv2.rectangle(view, (0, 0), (w, 54), (0, 0, 0), -1)
    cv2.putText(view, f"botol {bottle:02d} {brand or '-'}   {state.upper()}   foto {n_here + 1}"
                      f"{' -> tutup di ' + tgt if tgt else ' (sudah 3, ganti status / b)'}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
    cv2.putText(view, f"total {len(rows)}   lock {'ON' if locked else 'off'}   {msg}",
                (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.imshow(WIN, view)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord("q"), 27) or cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break
    elif k in (32, ord("s")):
        f = grab()
        if f is not None:
            pos = counts[(str(bottle), state)] + 1
            name = f"b{bottle:02d}_{slug(brand) or 'x'}_{'nocap' if state == 'no_cap' else 'cap'}_{pos:02d}.{a.fmt}"
            save(f, name, str(bottle), brand, state, pos)
            counts[(str(bottle), state)] = pos
            msg = f"disimpan {name}"
    elif k == ord("c"):
        state = "cap"
    elif k == ord("x"):
        state = "no_cap"
    elif k == ord("b"):
        brands[bottle] = brand
        bottle += 1
        brand = ask_brand(bottle)
        brands[bottle] = brand
        state = "cap"
        msg = f"botol {bottle:02d} dimulai"
    elif k == ord("v") and bottle > 1:                       # salah pencet b -> kembali ke botol sebelumnya
        brands[bottle] = brand
        bottle -= 1
        brand = brands.get(bottle, "")
        state = "no_cap" if (counts[(str(bottle), "cap")] >= 3 and counts[(str(bottle), "no_cap")] < 3) else "cap"
        msg = f"kembali ke botol {bottle:02d} {brand}"
    elif k == ord("e"):
        f = grab()
        if f is not None:
            pos = sum(1 for r in rows if r[3] == "empty") + 1
            name = f"empty_{pos:02d}.{a.fmt}"
            save(f, name, "0", "", "empty", pos)
            msg = f"disimpan {name}"
    elif k == ord("z") and rows:
        last = rows.pop()
        try:
            os.remove(os.path.join(a.out, last[0]))
        except OSError:
            pass
        key = (last[1], last[3])
        counts[key] = max(0, counts[key] - 1)
        write_manifest()
        msg = f"undo {last[0]}"
    elif k == ord("l"):
        set_lock(not locked)

cap.release()
cv2.destroyAllWindows()
print(f"selesai: {len(rows)} foto di {a.out}")
