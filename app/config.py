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

# ---- model KANDIDAT bayangan (sesi pembuktian, 23 Okt) ----
# Bila aktif (`--kandidat`), setiap botol yang dipindai saat berhenti juga dinilai model kandidat pada FRAME DIAM YANG SAMA.
# Keputusan di layar & poin TETAP dari model aktif; keputusan kandidat hanya dicatat ke <scan>/SCAN_KANDIDAT.csv untuk dibandingkan.
# Hanya backend pytorch (berkas .pt kandidat). Menambah waktu berhenti (~2x analisis).
SHADOW = False
_M = os.path.join(REPO, "datasets", "3_model_final")
KANDIDAT = dict(bottle=os.path.join(_M, "botol_ver10_kandidat.pt"), liquid=os.path.join(_M, "cairan_berisi_kandidat.pt"), size_rule="panjang",
                # merek (24 Okt): coca_cola vs non_coca_cola, resep A (foto belt saja), dinilai pada kotak botol model AKTIF; bonus tetap 0
                brand=os.path.join(_M, "merek_coca_kandidat.pt"))

# ---- mode KIOSK CAPTURER / ambil data (24 Okt, `--ambil-data`) ----
# Mesin dipakai seperti biasa (berhenti-lalu-pindai, poin tampil) sambil mengumpulkan data belt nyata.
# Operator memilih LABEL KELOMPOK di /operator ("semua botol berikutnya: air bening, tanpa tutup, Aqua 600"); label itu
# ikut tercatat per botol di SCAN.csv dan dibandingkan dengan keputusan model -> cukup periksa botol yang BEDA (lembar_cek --beda).
# Sesi tidak berakhir karena diam 30 s, dan TIDAK membuat kode klaim (database poin/akun tidak terisi data percobaan).
DATA_MODE = False
DATA_IDLE_TIMEOUT = 3600     # sesi capturer berakhir sendiri hanya setelah 1 jam tanpa botol
KELOMPOK_ISI = ("kosong", "air_bening", "berwarna", "campur", "bukan_botol")
KELOMPOK_TUTUP = ("tanpa", "bertutup", "campur")


def harapan_kelompok(isi, tutup):
    """Keputusan yang SEHARUSNYA untuk label kelompok (kebijakan 23 Okt: berisi apa pun, termasuk air bening, ditolak).
    '' = tidak bisa diharapkan (campur). bukan_botol: seharusnya tidak terpindai sama sekali."""
    if isi == "bukan_botol":
        return "tidak_terdeteksi"
    if tutup == "bertutup" or isi in ("air_bening", "berwarna"):
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
BRAND_BONUS = 0              # bonus merek (whitelist Coca-Cola/Fanta/Sprite) BELUM aktif: model merek belum dikerjakan ulang
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
