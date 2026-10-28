"""Pengaturan aplikasi kiosk RVM (poin, aturan sesi, server). Semua bisa ditimpa lewat argumen `python -m app ...`.
Ambang model (cairan, batas ukuran) TIDAK diatur di sini: itu milik rvm_kp/config.py (dikalibrasi saat pemasangan).
Kode harus tetap jalan di Python 3.8 (Docker JetPack 4 di Jetson Nano)."""
import os

APP_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(APP_DIR)
DATA_DIR = os.path.join(APP_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "rvm_app.db")
STATIC_DIR = os.path.join(APP_DIR, "static")

# ---- server ----
HOST = "0.0.0.0"
PORT = 8000
# Alamat yang dipakai di QR klaim (HP harus bisa membukanya: satu jaringan Wi-Fi/LAN dengan mesin).
# Kosong = deteksi otomatis IP LAN mesin. Isi manual jika salah, mis. "http://192.168.0.20:8000".
PUBLIC_BASE_URL = os.environ.get("RVM_PUBLIC_URL", "")

# ---- kamera & analisis ----
CAMERA_INDEX = 1             # C922 biasanya index 1 di laptop (0 = webcam bawaan); di Nano biasanya 0
CAMERA_W, CAMERA_H = 1280, 720
BACKEND = "pytorch"          # Nano: "tensorrt"; PC Intel: "openvino"
PROFILE = "nano"
PREVIEW_WIDTH = 640          # lebar gambar pratinjau yang dikirim ke layar (hemat CPU Nano)
PREVIEW_FPS = 10

# ---- pelacakan & keputusan per botol ----
ZONE_X = (0.25, 0.75)        # zona periksa (fraksi lebar frame); suara hanya dihitung saat pusat botol di zona ini
MIN_VOTES = 2                # minimal frame di zona sebelum botol diputuskan
LOST_SEC = 1.2               # botol dianggap keluar bila tidak terlihat selama ini
NOCAP_FRAC = 0.6             # botol dianggap tanpa tutup bila >= 60% deteksi leher = "tanpa_tutup"
# Ambang cairan diambil dari rvm_kp.config.LIQUID_THR (median p_liquid semua frame di zona).

# ---- berhenti-lalu-pindai (stop-and-scan, 22 Okt) ----
# Saat pusat botol masuk STOP_ZONE_X: konveyor dimatikan (PLC), tunggu STOP_SETTLE_SEC sampai botol diam, ambil STOP_FRAMES
# frame DIAM, putuskan dari frame diam saja, lalu konveyor jalan lagi. Alasan: semua data latih = foto diam (tanpa blur gerak).
# Matikan dengan `python -m app ... --terus` (konveyor jalan terus, voting dari frame bergerak seperti sebelumnya).
STOP_SCAN = True
STOP_ZONE_X = (0.40, 0.60)   # zona pemicu berhenti (fraksi lebar frame), di dalam ZONE_X
STOP_SETTLE_SEC = 0.5        # jeda setelah perintah berhenti (belt melambat, botol berhenti menggelinding)
STOP_FRAMES = 3              # frame diam yang dianalisis lalu di-voting
STOP_MAX_SEC = 4.0           # pengaman: konveyor dijalankan lagi paling lambat setelah ini
# Simpan frame diam tiap botol (diagnosis + data nyata belt untuk latih ulang). Kosong = tidak menyimpan.
SCAN_SAVE_DIR = os.path.join(REPO, "datasets", "1_foto_mentah", "{tanggal}_scan_app")

# ---- benda BUKAN botol (26 Okt) ----
# Kotak kelas non_bottle dari detektor (ver9c) yang masuk zona & tidak menumpang kotak botol = benda asing.
# Uji 22 Okt: muncul di 41/86 foto non-botol, 0/391 foto botol (4 sesi). Berhenti-lalu-pindai: belt berhenti, frame diam
# disimpan, keputusan "tolak: bukan botol". Belt TETAP berhenti sampai bendanya diambil (selama belum ada jalur tolak servo).
FOREIGN_ENABLE = True
FOREIGN_IOU_BOTTLE = 0.3     # kotak non_bottle yang menumpang kotak botol (IoU >= ini) diabaikan
# 28 Okt (kebijakan user: tolak yang benar > terima yang salah, KECUALI tolak karena noise): saat botol dipindai, bila ada
# benda LAIN yang meyakinkan (terlihat di >= MULTI_MIN_FRAMES frame diam pada posisi sama) dan DEKAT botol (celah < MULTI_GAP x
# panjang botol), botol DITOLAK 'lebih_dari_satu' (penyapu tak bisa memisahkan benda yang berdempetan).
MULTI_ENABLE = True
# 28 Okt (keluhan user di demo: botol 400 ml terbaca kecil): aturan ukuran AKTIF = 'panjang' (CATATAN §11: 25/26 vs luas 24/26).
# Botol 400 ml luas ~221k (< batas 228k) tetapi panjang ~850 (>= 817) -> tanggung. Hari ini 16 botol lain sama dengan 'luas'.
SIZE_RULE = "panjang"
MULTI_MIN_FRAMES = 2
MULTI_GAP = 0.25
FOREIGN_WAIT_CLEAR = False   # 27 Okt (user): belt TETAP jalan agar benda terbawa ke pembuangan; True = tunggu benda diambil tangan
FOREIGN_CONF = 0.60          # 27 Okt: noda bekas selotip di belt terbaca non_bottle 0,50 -> ambang 0,60 (data 22 Okt: 50/53 benda nyata >= 0,60)
HOLD_MAX_SEC = 15.0          # pengaman mode tunggu: benda yang tetap terlihat > ini (mis. noda belt) -> belt jalan lagi
HOLD_CLEAR_SEC = 1.5         # benda dianggap sudah diambil bila tidak terlihat di zona selama ini

# ---- deteksi "ada benda" dari patokan belt kosong (26 Okt) -> app/pencatat_benda.py ----
# "catat" = hanya mencatat kejadian (BENDA.csv + gambar), TIDAK menghentikan belt; "mati" = nonaktif.
# Mulai tiap sesi dengan belt KOSONG beberapa detik agar patokan terbentuk.
PRESENCE_MODE = "catat"

# ---- PENYAPU V di ujung belt (servo tim, 26 Okt) -> app/sortir.py ----
# Gagal-aman tanpa sensor tambahan: sapuan BUANG berkala + sapuan TERIMA terjadwal untuk botol yang diterima.
# Semua dalam DETIK BELT BERJALAN (berhenti tidak dihitung). KALIBRASI di mesin dengan SORTIR.csv:
#   SORTIR_TRAVEL_SEC  = waktu dari keputusan sampai botol tepat di bawah V (ukur: stopwatch dari belt jalan lagi setelah pindai)
#   SORTIR_BUANG_EVERY = harus < (panjang V searah belt / kecepatan belt), agar tiap benda pasti kena satu sapuan
SORTIR_ENABLE = True
SORTIR_TRAVEL_SEC = 3.0      # 28 Okt (user, setelah uji di mesin; 6,0 -> 5,2 -> 3,8 -> 3,0): botol diam di talang ~3 s belt jalan setelah pindai
SORTIR_BUANG_EVERY = 1.5
SORTIR_BUANG_PERIODIK = False  # 28 Okt: alat tim = TALANG tilting (benda diam di talang) -> sapuan berkala tak perlu; diganti SORTIR_KOSONGKAN
SORTIR_KOSONGKAN = False       # 28 Okt (user): MATI -- gerak BUANG sebelum TERIMA dianggap membingungkan. True = kosongkan talang ke BUANG
                               # saat keputusan botol diterima (benda tak terdeteksi yang tertinggal di talang tidak ikut ke TERIMA)
SORTIR_BUANG_TIKET = True    # benda yang ditolak juga dijadwalkan disapu ke BUANG saat tiba
SORTIR_GUARD_SEC = 1.4       # tidak ada sapuan buang berkala dalam +-ini dari jadwal botol diterima (28 Okt: >= siklus servo 1,2 s)
SORTIR_SWEEP_SEC = 1.3       # 28 Okt: siklus firmware ESP32 TERUKUR 1,2 s (BUSY->DONE) + cadangan; selama ini tidak ada perintah baru
SORTIR_LOG_BERKALA = False   # catat juga tiap sapuan buang berkala (konsol + SORTIR.csv); bawaan: hanya sapuan bertiket, berkala dihitung di panel
SORTIR_SERIAL = os.environ.get("RVM_SORTIR_SERIAL", "COM7")   # 28 Okt: ESP32 tilting (CP2102) di laptop = COM7; tak ada -> hanya dicatat   # mis. "COM5" (laptop) / "/dev/ttyUSB0" (Nano); kosong = hanya dicatat
SORTIR_BAUD = 115200

# Folder PER SESI (27 Okt, permintaan user): <SCAN_SAVE_DIR tanggal>/sesi_<id>_<jamMulai>/ berisi frame + semua CSV sesi itu.
SESI_DIR = ""                # diisi Machine.start_session; kosong = folder tanggal (mis. sebelum sesi pertama)


def scan_dir():
    """Folder simpan aktif: folder sesi yang sedang/terakhir berjalan, atau folder tanggal bila belum ada sesi."""
    import datetime
    d = SESI_DIR or SCAN_SAVE_DIR.replace("{tanggal}", datetime.date.today().isoformat())
    os.makedirs(d, exist_ok=True)
    return d


# ---- model KANDIDAT bayangan (sesi pembuktian, 23 Okt) ----
# Bila aktif (`--kandidat`), setiap botol yang dipindai saat berhenti juga dinilai model kandidat pada FRAME DIAM YANG SAMA.
# Keputusan di layar & poin TETAP dari model aktif; keputusan kandidat hanya dicatat ke <scan>/SCAN_KANDIDAT.csv untuk dibandingkan.
# Hanya backend pytorch (berkas .pt kandidat). Menambah waktu berhenti (~2x analisis).
SHADOW = False
_M = os.path.join(REPO, "datasets", "3_model_final")
# 28 Okt (persetujuan user): detektor ver12 & merek v2 (latih + data belt 27 Okt); cairan tetap s35 (v2 dinilai offline)
KANDIDAT = dict(bottle=os.path.join(_M, "botol_ver12_kandidat.pt"), liquid=os.path.join(_M, "cairan_berisi_kandidat.pt"), size_rule="panjang",
                # merek (24 Okt): coca_cola vs non_coca_cola, resep A (foto belt saja), dinilai pada kotak botol model AKTIF; bonus tetap 0
                brand=os.path.join(_M, "merek_coca_v2_kandidat.pt"))

# ---- mode KIOSK CAPTURER / ambil data (24 Okt, `--ambil-data`) ----
# Mesin dipakai seperti biasa (berhenti-lalu-pindai, poin tampil) sambil mengumpulkan data belt nyata.
# Operator memilih LABEL KELOMPOK di /operator ("semua botol berikutnya: air bening, tanpa tutup, Aqua 600"); label itu
# ikut tercatat per botol di SCAN.csv dan dibandingkan dengan keputusan model -> cukup periksa botol yang BEDA (lembar_cek --beda).
# Sesi tidak berakhir karena diam 30 s, dan TIDAK membuat kode klaim (database poin/akun tidak terisi data percobaan).
DATA_MODE = False
DATA_IDLE_TIMEOUT = 3600     # sesi capturer berakhir sendiri hanya setelah 1 jam tanpa botol
# 29 Okt: tingkat isi air bening (uji batas deteksi per tingkat isi; sesi 27 Okt diisi acak tanpa label tingkat)
KELOMPOK_ISI = ("kosong", "air_1-6", "air_1-3", "air_1-2", "air_penuh", "air_bening", "berwarna", "campur", "bukan_botol")
KELOMPOK_TUTUP = ("tanpa", "bertutup", "campur")


def harapan_kelompok(isi, tutup):
    """Keputusan yang SEHARUSNYA untuk label kelompok (kebijakan 23 Okt: berisi apa pun, termasuk air bening, ditolak).
    '' = tidak bisa diharapkan (campur). bukan_botol: seharusnya tidak terpindai sama sekali."""
    if isi == "bukan_botol":
        return "bukan_botol"                                   # seharusnya ditolak dengan alasan 'bukan_botol' 
    if tutup == "bertutup" or isi in ("air_bening", "berwarna") or isi.startswith("air_"):
        return "tolak"
    if isi == "kosong" and tutup == "tanpa":
        return "terima"
    return ""


# ---- sesi ----
IDLE_TIMEOUT = 30            # detik tanpa botol terlihat -> sesi berakhir (sama dengan alur mesin)
SUMMARY_SEC = 75             # layar ringkasan kembali ke layar awal setelah ini
CLAIM_HOURS = 7 * 24         # kode klaim berlaku 7 hari; baru setelah itu poin yang tidak diklaim didonasikan ke kampanye default
# Waktu habis di layar ringkasan TIDAK memindahkan poin: poin tetap di kode klaim sampai diklaim/didonasikan pengguna.
CLAIM_LINK_SEC = 180         # setelah HP membuka tautan QR, kiosk menunggu selama ini (proses di HP boleh lebih lama)
RECOVER_IDLE_SEC = 120       # sesi tanpa aktivitas selama ini yang belum ditutup (server mati/putus) dipulihkan jadi kode klaim

# ---- poin (pengembangan: belum ada nilai rupiah asli) ----
POINTS_BY_SIZE = {"kecil": 10, "tanggung": 15, "besar": 25}
POINTS_UNKNOWN_SIZE = 10     # ukuran tak terbaca (botol terpotong tepi frame) -> poin terkecil
BRAND_BONUS = 10             # 28 Okt (keputusan user, demo): +10 poin untuk botol DITERIMA yang terbaca produk Coca-Cola Company
BRAND_MODEL = os.path.join(REPO, "datasets", "3_model_final", "merek_coca_v2_kandidat.pt")   # CV 104/107, salah bonus ~1/75 non-Coca
# Perkiraan berat botol PET kosong per kategori (gram) untuk angka "plastik terselamatkan".
GRAMS_BY_SIZE = {"kecil": 14, "tanggung": 20, "besar": 32, "?": 16}

# ---- akun ----
PIN_LEN = 6
PIN_MAX_FAIL = 5             # salah PIN 5x -> akun dikunci sementara
PIN_LOCK_SEC = 300
AUTH_DAYS = 30

# ---- PLC (konveyor) ----
# Mati secara default. Nyalakan dengan --plc HANYA setelah area belt dipastikan aman.
PLC_IP = "192.168.0.1"
PLC_BYTE_RUN, PLC_BIT_RUN = 10, 0   # MB10.0 (sama dengan server PLC lama)
