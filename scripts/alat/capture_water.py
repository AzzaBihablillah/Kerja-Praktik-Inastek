"""Ambil foto dataset CAIRAN BERWARNA di botol dari kamera USB (C922) di konveyor.
Satu tombol satu foto; label di nama file + MANIFEST.csv (klasifikasi, tanpa box).

Pakai:  python capture_water.py [--cam 1] [--res 1920x1080] [--fmt png] [--out DIR] [--list]

PROTOKOL (per botol, 4 SISI per kondisi; program memandu urutannya dan maju otomatis):
  1. PENUH     + tutup TERPASANG (cap)      4 foto
  2. 1/2       + tutup TERPASANG (cap)      4 foto   (segel -> tidak tumpah saat rebah)
  3. KOSONG    + tutup LEPAS (no_cap)       4 foto
  4. KOSONG    + tutup TERPASANG (cap)      4 foto   (agar tutup tidak jadi jalan pintas bagi model)
  = 16 foto/botol (level 1/4 dihapus dari protokol sejak 18 Okt). Foto ke-1..4 per kondisi = SISI 1..4: putar botol 90 derajat tiap foto (label berpindah sisi),
  leher bergantian KANAN/KIRI dan posisi 1/3, 2/3, tengah, tengah (garis kuning + teks di layar).

Tombol (klik jendela dulu):
  SPACE / s  simpan foto (maju otomatis ke langkah berikutnya setelah 4 sisi)
  k          LEWATI kondisi ini (botol itu tidak punya datanya) -> dicatat di manifest sbg SKIP, lanjut ke langkah berikutnya
  c / x      tutup: c = cap (terpasang), x = no_cap (lepas)      [t = bolak-balik]
  1 2 3 4    ganti tingkat isi manual: 1=kosong 2=1/4 3=1/2 4=penuh
  b          botol baru: KETIK nama minuman + Enter, KETIK warna cairan + Enter (langsung di jendela, tanpa popup)
  n          koreksi nama/warna botol yang sedang difoto (semua fotonya ikut diganti nama)
  v          kembali ke botol sebelumnya (kalau salah pencet b)
  e          foto belt KOSONG (tanpa botol): ambil di awal, tengah, akhir sesi utk memantau drift cahaya
  z          undo (hapus foto / batalkan lewati terakhir)
  q / Esc    keluar
Nama file: bNN_<nama>_<warna>_<kosong|seperempat|setengah|penuh>_<capon|capoff>_<sisi 01-04>.png
(arah leher TIDAK ada di nama file: di set 16 Okt labelnya salah; kolom manifest neck_nominal = yang DIMINTA,
 neck_auto = yang TERLIHAT menurut model.)
PEMERIKSAAN OTOMATIS (QC) tiap kali simpan: botol terdeteksi? model cap setuju dengan label tutup? leher di sisi yang diminta?
  Hasil di bilah bawah (hijau OK / kuning periksa / merah SALAH + bunyi bip) dan di kolom manifest cap_auto/cap_conf/qc.
  Foto TETAP disimpan; kalau salah tekan z (undo), perbaiki tutup/botol, foto ulang. Model tidak sempurna (tutup hitam di botol hitam
  sering lolos) -> ini peringatan, bukan hakim. Matikan dengan --noqc.
Default 1920x1080 MJPG, PNG lossless, TANPA kunci fokus/exposure.
Cakupan (rapat mitra 2026-10-16): hanya cairan BERWARNA; air bening di luar lingkup (isi warna 'bening').
"""
import argparse, csv, os, sys, time, datetime, re, collections
import cv2

DS = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets"
ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=1)
ap.add_argument("--res", default="1920x1080")
ap.add_argument("--fmt", default="png", choices=["png", "jpg"])
ap.add_argument("--out", default=f"{DS}/1_foto_mentah/{datetime.date.today():%Y-%m-%d}_water_v1")
ap.add_argument("--selftest", action="store_true")
ap.add_argument("--noqc", action="store_true", help="matikan pemeriksaan otomatis (model bottle + cap) saat menyimpan")
ap.add_argument("--center", action="store_true", help="semua sisi di TENGAH frame (botol besar 1-1.5 L agar tidak terpotong tepi frame)")
ap.add_argument("--list", action="store_true", help="tampilkan semua kamera (index + resolusi) lalu keluar; pilih index C922 utk --cam")
a = ap.parse_args()
if a.list:
    import numpy as _np
    tiles = []
    for i in range(0, 6):
        c = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not c.isOpened():
            continue
        c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")); c.set(3, 1920); c.set(4, 1080)
        for _ in range(8):
            c.read()
        ok, f = c.read(); c.release()
        if not ok:
            continue
        print(f"index {i}: {f.shape[1]}x{f.shape[0]}", flush=True)
        t = cv2.resize(f, (480, 270)); cv2.putText(t, f"--cam {i}  ({f.shape[1]}x{f.shape[0]})", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        tiles.append(t)
    if not tiles:
        sys.exit("tidak ada kamera yang terbuka")
    while len(tiles) % 3:
        tiles.append(_np.zeros_like(tiles[0]))
    cv2.imshow("kamera: pilih index yang menampilkan BELT (C922). Tekan tombol apa saja", cv2.vconcat([cv2.hconcat(tiles[j:j + 3]) for j in range(0, len(tiles), 3)]))
    cv2.waitKey(0); cv2.destroyAllWindows(); sys.exit(0)
W, H = map(int, a.res.lower().split("x"))
os.makedirs(a.out, exist_ok=True)
MAN = os.path.join(a.out, "MANIFEST.csv")

LEVELS = {1: "kosong", 2: "seperempat", 3: "setengah", 4: "penuh"}
LV_SHORT = {"kosong": "kosong", "seperempat": "1/4", "setengah": "1/2", "penuh": "penuh"}
STEPS = [("penuh", "on"), ("setengah", "on"), ("kosong", "off"), ("kosong", "on")]   # 1/4 dibuang (tidak konsisten di set 16 Okt); tombol 2 masih bisa manual
STEP_TXT = {("penuh", "on"): "isi PENUH, tutup TERPASANG",
            ("setengah", "on"): "tuang jadi 1/2, tutup TERPASANG (segel)",
            ("seperempat", "off"): "tuang jadi 1/4, BUKA tutup",
            ("kosong", "off"): "KOSONGKAN, tutup LEPAS",
            ("kosong", "on"): "PASANG tutup lagi (botol kosong)"}
SIDES = 4
POSX = [0.5] * 4 if a.center else [1 / 3, 2 / 3, 0.5, 0.5]   # posisi tengah botol di lebar frame per sisi (--center: semua di tengah)
DIRS = ["kanan", "kiri", "kanan", "kiri"]               # arah LEHER per sisi (mesin menerima kedua arah)
SIDE_TXT = ["sisi 1: label menghadap ATAS", "sisi 2: putar botol 90", "sisi 3: putar 180", "sisi 4: putar 270"]
NECKC = {"kanan": "r", "kiri": "l"}                     # huruf arah di nama file (jangan huruf pertama: kanan/kiri sama-sama k)
HDR = ["file", "bottle", "brand", "liquid", "level", "cap", "n", "timestamp", "neck_nominal", "neck_auto", "cap_auto", "cap_conf", "qc"]

rows = []
if os.path.exists(MAN):                                   # baca per nama kolom (manifest lama punya kolom 'neck'/'catatan')
    with open(MAN, newline="") as f:
        for d in csv.DictReader(f):
            d["neck_nominal"] = d.get("neck_nominal") or d.get("neck", "")
            rows.append([d.get(h, "") or "" for h in HDR])
bottle = max([int(r[1]) for r in rows if r[1].isdigit()] + [1])
brands = {int(r[1]): (r[2], r[3]) for r in rows if r[1].isdigit()}
brand, liquid = brands.get(bottle, ("", "cola"))
level, cap_state = "penuh", "on"
counts = collections.Counter((r[1], r[4], r[5]) for r in rows if r[0] != "SKIP")
skipped = {(r[1], r[4], r[5]) for r in rows if r[0] == "SKIP"}
msg = f"lanjut dari {len(rows)} baris" if rows else "folder baru"


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def pad(rec):
    return rec + [""] * (len(HDR) - len(rec))


qc_msg, qc_lvl = "", 0
water_qc = None
if not a.noqc and not a.list:
    import water_qc
    print("memuat model QC (bottle + cap)...", flush=True)
    water_qc._load()


def write_manifest():
    with open(MAN, "w", newline="") as f:
        w = csv.writer(f); w.writerow(HDR); w.writerows(rows)


def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---- langkah protokol -------------------------------------------------------
def step_text():
    return STEP_TXT.get((level, cap_state), f"{LV_SHORT[level]} tutup {cap_state.upper()} (manual)")


def goto_next_step():
    """Pindah ke langkah pertama yang belum selesai (<4 sisi) dan belum dilewati. False kalau botol sudah selesai."""
    global level, cap_state
    for lv, cp in STEPS:
        key = (str(bottle), lv, cp)
        if counts[key] < SIDES and key not in skipped:
            level, cap_state = lv, cp
            return True
    return False


def commit_photo(frame):
    """Simpan frame sbg sisi berikutnya utk (botol, level, tutup) saat ini; maju otomatis setelah 4 sisi."""
    global msg
    global qc_msg, qc_lvl
    key = (str(bottle), level, cap_state)
    n = counts[key] + 1
    nk = DIRS[min(n - 1, len(DIRS) - 1)]
    name = f"b{bottle:02d}_{slug(brand) or 'x'}_{liquid}_{level}_cap{cap_state}_{n:02d}.{a.fmt}"
    auto = ["", "", "", ""]
    if water_qc is not None:
        res = water_qc.analyze(frame)
        w, qc_lvl = water_qc.warnings(res, cap_state, nk)
        qc_msg = " | ".join(w) if w else f"QC OK: botol terdeteksi, model cap setuju ({res['cap_state']} {res['cap_conf']:.2f})"
        auto = [res["neck_auto"] or "", res["cap_state"] or "", f"{res['cap_conf']:.2f}", ";".join(w)]
        if qc_lvl == 2:
            try:
                import winsound; winsound.Beep(1000, 400)
            except Exception:
                print("\a", end="", flush=True)
    save(frame, name, [name, str(bottle), brand, liquid, level, cap_state, str(n), now(), nk] + auto)
    counts[key] = n
    msg = f"disimpan {name}"
    if qc_lvl:
        msg += "  (QC: periksa, z = ulang)"
    if n == SIDES:
        msg = "4 sisi selesai. " + (("LANJUT: " + step_text()) if goto_next_step() else "BOTOL SELESAI - tekan b untuk botol baru")


def skip_step():
    """Lewati kondisi saat ini (botol tidak punya datanya): dicatat sbg baris SKIP supaya analisis tahu ini disengaja."""
    global msg
    key = (str(bottle), level, cap_state)
    if key in skipped or counts[key] >= SIDES:
        msg = "kondisi ini sudah selesai/dilewati"; return
    rows.append(pad(["SKIP", str(bottle), brand, liquid, level, cap_state, "0", now(), "-"])); write_manifest()
    skipped.add(key)
    msg = f"DILEWATI: {LV_SHORT[level]} tutup {cap_state.upper()}. " + (("LANJUT: " + step_text()) if goto_next_step() else "BOTOL SELESAI - tekan b")


def undo_last():
    global msg, level, cap_state
    if not rows:
        return
    last = rows.pop()
    if last[0] == "SKIP":
        skipped.discard((last[1], last[4], last[5])); msg = "batal lewati"
    else:
        try:
            os.remove(os.path.join(a.out, last[0]))
        except OSError:
            pass
        key = (last[1], last[4], last[5]); counts[key] = max(0, counts[key] - 1); msg = f"undo {last[0]}"
    if last[1] == str(bottle) and last[4] != "ref":
        level, cap_state = last[4], last[5]                 # kembali ke kondisi itu agar bisa difoto ulang
    write_manifest()


# ---- input nama + warna: DIKETIK LANGSUNG di jendela kamera (tanpa popup) ----
COLORS = "cola / merah / oranye / kuning / coklat / putih / hijau / ungu / bening"
edit = None


def start_edit(mode):
    """mode: 'first' = botol pertama, 'new' = botol berikutnya, 'edit' = koreksi botol yang sedang difoto."""
    global edit
    edit = {"mode": mode, "field": 0, "vals": ["", ""], "buf": "",
            "defaults": [brand if mode == "edit" else "", liquid]}


def rename_bottle(b, new_brand, new_liquid):
    """Ganti nama/warna SEMUA foto botol b (nama file + manifest). Kembalikan jumlah file yang diganti nama."""
    n_ren = 0
    for r in rows:
        if r[1] != str(b):
            continue
        r[2], r[3] = new_brand, new_liquid
        if r[0] != "SKIP" and r[4] != "ref":
            ext = os.path.splitext(r[0])[1]
            new = f"b{b:02d}_{slug(new_brand) or 'x'}_{new_liquid}_{r[4]}_cap{r[5]}_{int(r[6]):02d}{ext}"
            if new != r[0] and os.path.exists(os.path.join(a.out, r[0])):
                os.replace(os.path.join(a.out, r[0]), os.path.join(a.out, new)); r[0] = new; n_ren += 1
    write_manifest()
    return n_ren


def finish_edit():
    global edit, bottle, brand, liquid, msg
    nm = edit["vals"][0].strip()
    lq = slug(edit["vals"][1]) or "cola"
    mode = edit["mode"]
    edit = None
    if mode == "new":
        brands[bottle] = (brand, liquid)
        bottle += 1
        brand, liquid = nm, lq
        goto_next_step()
        msg = f"botol {bottle:02d} dimulai. {step_text()}"
    elif mode == "edit":
        n = rename_bottle(bottle, nm, lq)
        brand, liquid = nm, lq
        msg = f"info botol {bottle:02d} diubah ({n} file diganti nama)"
    else:
        brand, liquid = nm, lq
        goto_next_step()
        msg = f"botol {bottle:02d}. {step_text()}"
    brands[bottle] = (brand, liquid)


def handle_edit_key(k):
    global edit, msg
    if k == 13:                                              # Enter: kosong = pakai default
        f = edit["field"]
        edit["vals"][f] = edit["buf"].strip() or edit["defaults"][f]
        if f == 0:
            edit["field"], edit["buf"] = 1, ""
        else:
            finish_edit()
    elif k == 27:
        edit = None; msg = "input dibatalkan"
    elif k in (8, 127):
        edit["buf"] = edit["buf"][:-1]
    elif 32 <= k <= 126:
        edit["buf"] += chr(k)


def draw_edit(view):
    h, w = view.shape[:2]
    y0 = h // 2 - 110
    cv2.rectangle(view, (30, y0), (w - 30, y0 + 220), (25, 25, 25), -1)
    cv2.rectangle(view, (30, y0), (w - 30, y0 + 220), (0, 255, 255), 2)
    title = {"first": "BOTOL 01 (awal)", "new": f"BOTOL BARU {bottle + 1:02d}", "edit": f"KOREKSI botol {bottle:02d}"}[edit["mode"]]
    cv2.putText(view, title, (50, y0 + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    f = edit["field"]
    lab = "NAMA minuman/merek" if f == 0 else "WARNA cairan"
    d = edit["defaults"][f]
    cv2.putText(view, f"{lab}:  {edit['buf']}_", (50, y0 + 88), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
    cv2.putText(view, f"Enter kosong = {d if d else '(kosong)'}", (50, y0 + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 180), 1)
    if f == 1:
        cv2.putText(view, "warna: " + COLORS, (50, y0 + 152), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
    cv2.putText(view, "ketik langsung | Enter = lanjut/simpan | Backspace = hapus | Esc = batal", (50, y0 + 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)


def checklist():
    parts = []
    for lv, cp in STEPS:
        key = (str(bottle), lv, cp)
        tag = "SKIP" if key in skipped else f"{min(counts[key], SIDES)}/{SIDES}"
        t = f"{LV_SHORT[lv]}+{'cap' if cp == 'on' else 'nocap'} {tag}"
        parts.append(f"[{t}]" if (lv, cp) == (level, cap_state) else t)
    return "  ".join(parts)


# ---- kamera -----------------------------------------------------------------
cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)
if not cap.isOpened():
    sys.exit(f"kamera index {a.cam} tidak bisa dibuka (dipakai program lain?)")
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
for _ in range(15):
    cap.read(); time.sleep(0.03)
_ok, _f = cap.read()
aw, ah = (_f.shape[1], _f.shape[0]) if _ok else (0, 0)
print(f"resolusi aktual: {aw} x {ah}", flush=True)
if (aw, ah) != (W, H) and not a.selftest:
    cap.release()
    sys.exit(f"STOP: kamera memberi {aw}x{ah}, diminta {W}x{H}. Jangan lanjut agar resolusi dataset tidak campur.\n"
             "Jalankan dengan --list untuk memilih index kamera yang benar (C922), atau cabut-colok C922.")


def grab():
    cap.read()
    ok, f = cap.read()
    return f if ok else None


def save(frame, name, rec):
    p = os.path.join(a.out, name)
    if name.endswith(".png"):
        cv2.imwrite(p, frame, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    else:
        cv2.imwrite(p, frame, [cv2.IMWRITE_JPEG_QUALITY, 100])
    rows.append(pad(rec)); write_manifest()


if a.selftest:
    f = grab()
    if f is None:
        sys.exit("selftest: tidak ada frame")
    save(f, f"selftest.{a.fmt}", ["selftest." + a.fmt, "0", "test", "bening", "kosong", "off", "1", now(), "-"])
    print("selftest OK", f.shape, "->", a.out); cap.release(); sys.exit(0)

WIN = "capture CAIRAN BERWARNA  [SPACE simpan | k lewati | c cap | x nocap | b botol baru | n koreksi | v balik | e belt kosong | z undo | q keluar]"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
if not rows:
    start_edit("first")
else:
    goto_next_step()
while True:
    ok, frame = cap.read()
    if not ok:
        continue
    fh, fw = frame.shape[:2]
    view = cv2.resize(frame, (960, int(960 * fh / fw)))
    h, w = view.shape[:2]
    n_here = counts[(str(bottle), level, cap_state)]
    si = min(n_here, SIDES - 1)
    tx = int(w * POSX[si])
    cv2.rectangle(view, (0, 0), (w, 100), (0, 0, 0), -1)
    cv2.line(view, (tx, 106), (tx, h), (0, 255, 255), 2)                     # target posisi TENGAH botol (kuning)
    col = (0, 200, 0) if level == "kosong" else (0, 165, 255)
    cv2.putText(view, f"botol {bottle:02d} {brand or '-'} [{liquid}]   {LV_SHORT[level].upper()} + TUTUP {cap_state.upper()}   sisi {min(n_here + 1, SIDES)}/{SIDES}",
                (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    cv2.putText(view, checklist(), (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    cv2.putText(view, f"{step_text()}   |   {SIDE_TXT[si]}", (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 230, 255), 1)
    cv2.putText(view, f"foto {sum(1 for r in rows if r[0] != 'SKIP')}   {msg}", (8, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
    cv2.putText(view, f"LEHER KE {DIRS[si].upper()}", (w - 300, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    if qc_msg:                                                              # bilah QC di bawah: hijau OK / kuning periksa / merah salah
        qcol = {0: (0, 160, 0), 1: (0, 200, 255), 2: (0, 0, 230)}[qc_lvl]
        cv2.rectangle(view, (0, h - 34), (w, h), qcol, -1)
        cv2.putText(view, qc_msg[:120], (8, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0) if qc_lvl == 1 else (255, 255, 255), 1)
    if edit is not None:
        draw_edit(view)
    cv2.imshow(WIN, view)
    k = cv2.waitKey(1) & 0xFF
    if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break
    if k == 255:
        continue
    if edit is not None:                                     # sedang mengetik nama/warna: tombol lain dinonaktifkan
        handle_edit_key(k)
        continue
    if k in (ord("q"), 27):
        break
    elif k in (32, ord("s")):
        f = grab()
        if f is not None:
            commit_photo(f)
    elif k == ord("k"):
        skip_step()
    elif k == ord("c"):
        cap_state = "on"
    elif k == ord("x"):
        cap_state = "off"
    elif k == ord("t"):
        cap_state = "off" if cap_state == "on" else "on"
    elif k in (ord("1"), ord("2"), ord("3"), ord("4")):
        level = LEVELS[int(chr(k))]
    elif k == ord("b"):
        start_edit("new")
    elif k == ord("n"):
        start_edit("edit")
    elif k == ord("v") and bottle > 1:
        brands[bottle] = (brand, liquid)
        bottle -= 1
        brand, liquid = brands.get(bottle, ("", "cola"))
        goto_next_step()
        msg = f"kembali ke botol {bottle:02d} {brand}. {step_text()}"
    elif k == ord("e"):                                     # referensi belt kosong (pantau drift cahaya)
        f = grab()
        if f is not None:
            m = sum(1 for r in rows if r[1] == "0" and r[4] == "ref") + 1
            name = f"ref_belt_{m:02d}.{a.fmt}"
            save(f, name, [name, "0", "", "-", "ref", "-", str(m), now(), "-"])
            msg = f"disimpan {name}"
    elif k == ord("z"):
        undo_last(); qc_msg, qc_lvl = "", 0

cap.release(); cv2.destroyAllWindows()
print(f"selesai: {sum(1 for r in rows if r[0] != 'SKIP')} foto di {a.out}")
