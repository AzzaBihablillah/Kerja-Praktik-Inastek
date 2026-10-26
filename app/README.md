# app/ — aplikasi kiosk RVM "Tukar Botol" (KP PT Inastek)

Satu server FastAPI untuk tiga layar:

| alamat | untuk | isi |
|---|---|---|
| `/` | layar sentuh di mesin | layar awal (aturan, poin per ukuran, statistik, kampanye), sesi berjalan (kamera + hasil tiap botol), ringkasan + **QR klaim** |
| `/k/<kode>` | HP pengguna (hasil pindai QR) | simpan poin ke akun **atau** donasikan |
| `/k` | HP pengguna | ketik kode klaim bila QR gagal dipindai |
| `/akun` | HP pengguna | saldo, tukar hadiah (simulasi), donasi dari saldo, riwayat |
| `/operator` | petugas/peneliti | status mesin, waktu per tahap, **data kalibrasi per botol** (p_cairan, area_1920, suara leher), tombol simulasi |

Analisis: `rvm_kp.RVMAnalyzer` (profil `nano`) + kamera `rvm_kp.kamera.Kamera`. Folder `open_vino/` tidak dipakai.
Tanpa build step Node; frontend = HTML/CSS/JS biasa di `static/`. Dependensi tambahan hanya `fastapi`, `uvicorn`, `segno` (QR).

## Menjalankan (dari folder proyek, induk `app/`)
```bash
# coba antarmuka tanpa kamera/model/belt (belt virtual):
venv/Scripts/python.exe -m app --sim
# laptop + kamera belt C922 (index 1), model .pt:
venv/Scripts/python.exe -m app --backend pytorch --cam 1
# Jetson Nano (di dalam Docker, setelah engine TensorRT dibuat; lihat rvm_kp/JETSON-NANO.md):
python -m app --backend tensorrt --cam 0
```
Opsi lain: `--profil laptop`, `--port 8000`, `--url http://192.168.0.20:8000` (alamat di QR bila deteksi IP otomatis salah), `--plc`.
**`--plc` menyalakan/mematikan konveyor (MB10.0). Pakai hanya setelah area belt dipastikan aman.** Tanpa `--plc` konveyor tidak disentuh.

Mode simulasi: tombol di bar atas layar sesi, atau tombol keyboard `1` kecil, `2` tanggung, `3` besar, `4` bertutup, `5` berisi, `A` otomatis.

## Benda bukan botol & jalur terima/tolak (26 Okt)
- Detektor punya kelas `non_bottle`. Kotak `non_bottle` yang masuk zona dan tidak menumpang kotak botol = **benda asing**: belt berhenti, frame diam disimpan (SCAN.csv alasan `bukan_botol`), layar "Bukan botol PET – ambil kembali bendanya". Belt **ditahan** sampai benda tidak terlihat lagi di zona selama `HOLD_CLEAR_SEC` (default 1,5 s), lalu jalan lagi (`app/config.py`: `FOREIGN_*`).
- **Penyapu V (servo tim) — `app/sortir.py`.** V terbalik di atas ujung belt melempar benda ke wadah TERIMA/BUANG. Tanpa sensor tambahan, jadi GAGAL-AMAN: (1) **sapuan BUANG berkala** tiap `SORTIR_BUANG_EVERY` detik belt jalan -> benda apa pun di bawah V (termasuk yang tak terdeteksi kamera) terbuang; (2) **sapuan TERIMA terjadwal** untuk botol yang diterima, `SORTIR_TRAVEL_SEC` detik belt jalan setelah keputusannya; sapuan berkala ditunda dalam +-`SORTIR_GUARD_SEC` dari botol itu; (3) benda ditolak juga dijadwalkan ke BUANG. Waktu dihitung sebagai **detik belt berjalan** (saat berhenti untuk pindai/menunggu, jam ikut berhenti). Panel `/operator` → "Penyapu V" (posisi, antrean, jumlah sapuan); tiap sapuan bertiket dicatat di `SORTIR.csv`. Perintah servo di `Sortir._kirim()`: kini dicatat; isi `SORTIR_SERIAL` (env `RVM_SORTIR_SERIAL`, mis. `COM5`) untuk mengirim protokol USULAN `T`/`B`/`N` + newline (sesuaikan firmware tim). Setelah penyapu terpasang, set `FOREIGN_WAIT_CLEAR = False`.
- **Kalibrasi di mesin:** `SORTIR_TRAVEL_SEC` (stopwatch: dari belt jalan lagi setelah pindai sampai botol tepat di bawah V); `SORTIR_BUANG_EVERY` < panjang V searah belt ÷ kecepatan belt; dua benda harus berjarak > `SORTIR_GUARD_SEC` di belt.
- Uji offline (foto 22 Okt, `rvm_kp/analyzer.last_nonbottle` + ver9c): 86 non-botol -> 41 ditolak 'bukan botol' (belt ditahan, jalan lagi setelah diambil), 30 dikira botol tetapi tertolak di pemeriksaan tutup/isi (sebagian besar 'leher'), 15 tidak memicu apa pun; 94 foto botol: 0 salah ditahan. Detektor kandidat ver10 TIDAK mengeluarkan `non_bottle` (dilatih sebagai latar) -> jalur ini hanya bekerja dengan ver9c.
- Simulasi: tombol / tombol keyboard `6` "Bukan botol".

## Kiosk capturer (`--ambil-data`): demo sekaligus pengumpul data
```bash
venv/Scripts/python.exe -m app --backend pytorch --cam 1 --plc --jeda 1.0 --kandidat --ambil-data
```
Mesin bekerja seperti biasa (berhenti-lalu-pindai, poin tampil di kiosk), dengan tanda **KIOSK CAPTURER** di kiosk dan `/operator`. Bedanya:
- Di `/operator` (bisa dari HP) pilih **label kelompok** untuk semua botol berikutnya: isi (kosong / air bening / berwarna / campur / bukan botol), tutup (tanpa / bertutup / campur), produk, ml, catatan → **Mulai kelompok ini**. Ganti label setiap kali jenis botol yang dimasukkan berganti.
- Tiap botol: frame diam (PNG) + baris `SCAN.csv` berisi label kelompok, keputusan yang **seharusnya** (kebijakan 23 Okt: berisi apa pun ditolak) dan `cocok` ya/tidak; `KELOMPOK.csv` mencatat tiap kelompok; dengan `--kandidat` juga `cocok_kandidat` di `SCAN_KANDIDAT.csv`.
- Panel menampilkan penghitung langsung (terpindai, model aktif sesuai, kandidat sesuai) dan daftar botol yang **beda**.
- Sesi tidak berakhir karena diam 30 s (baru setelah 1 jam), dan **tidak membuat kode klaim** (sesi ditandai `capturer_*`; pemulihan juga tanpa klaim).
- Sesudahnya cukup periksa yang beda: `python scripts/alat/lembar_cek.py --dir 1_foto_mentah/<tanggal>_scan_app --beda --acak 5` → `LEMBAR-BEDA.jpg` (botol beda + 5 sampel acak yang sesuai, untuk memastikan label kelompoknya benar).
- Label kelompok = label lemah: satu botol yang salah masuk kelompok ikut salah label. Sampel acak `--acak` ada untuk menangkap ini.

HP harus **satu jaringan Wi-Fi/LAN** dengan mesin agar QR bisa dibuka. Pastikan firewall mengizinkan port 8000.

## Alur pengguna
1. Sentuh **Mulai setor botol** → sesi mulai (konveyor jalan bila `--plc`).
2. Tiap botol yang lewat zona periksa diputuskan **sekali** (voting beberapa frame, `tracker.py`):
   - leher tidak pernah terlihat → **tolak** "Leher botol tidak terlihat" (tidak dianggap tanpa tutup, sama seperti server PLC lama);
   - < 60% deteksi leher = tanpa tutup → **tolak** "Tutup masih terpasang" (cincin di leher = sah);
   - median p_cairan ≥ `rvm_kp.config.LIQUID_THR` → **tolak** "Masih ada isinya";
   - selain itu **terima**, poin menurut ukuran: kecil 10, tanggung 15, besar 25 (`config.POINTS_BY_SIZE`).
3. Sesi selesai bila pengguna menekan **Selesai** atau 30 detik tanpa botol terlihat.
4. Layar ringkasan menampilkan total poin lalu **dua pilihan besar**:
   - **Ambil poin untuk saya** → QR + kode 6 karakter (`/k/<kode>`) → di HP: simpan ke akun (nomor HP + PIN 6 angka);
   - **Donasikan ke yayasan** → pilih salah satu program (kartu besar dengan progres) → konfirmasi → layar terima kasih + progres yayasan bertambah.
   **Poin tidak pernah berpindah karena waktu habis.** Donasi hanya terjadi bila pengguna menekan tombol konfirmasi (di kiosk atau di HP).
   - Hitung mundur 75 detik tertunda setiap kali layar disentuh. Bila habis tanpa pilihan, layar menampilkan "Poinmu tetap aman" + kode klaim (25 detik), lalu kembali ke awal. Poin tetap di kode itu.
   - Begitu HP membuka tautan QR, kiosk pindah ke "Lanjutkan di HP" dan menunggu 180 detik (`CLAIM_LINK_SEC`). Proses di HP (daftar, masuk, pilih) boleh lebih lama: klaim tetap terbuka walau kiosk sudah kembali ke awal.
5. Kode klaim berlaku **7 hari** (`CLAIM_HOURS`). Baru setelah itu poin yang tidak diklaim didonasikan ke kampanye default (tercatat sumber `kedaluwarsa`).
6. **Pemulihan sesi:** bila server mati / kamera putus di tengah sesi, sesi yang tidak tertutup dan diam ≥ 120 detik (`RECOVER_IDLE_SEC`) ditutup otomatis dengan alasan `pemulihan`; poinnya dibuatkan kode klaim (terlihat di `/operator`, tabel "Klaim terakhir"), dicek saat server mulai dan tiap menit.
7. Di `/akun`: tukar hadiah (kode `DEMO-…`, bukan hadiah asli), donasi dari saldo, riwayat.

## Data (SQLite, `app/data/rvm_app.db`, tidak ikut git)
`sessions`, `bottles` (keputusan + p_liquid, area, suara leher per botol — bahan kalibrasi & laporan TA), `claims`, `users` (PIN di-hash PBKDF2, kunci 5 menit setelah 5 kali salah), `auth`, `ledger` (buku poin; saldo = jumlah mutasi), `campaigns`, `donations`, `rewards`, `redemptions`.
Kampanye & hadiah contoh ada di `db.py` (`CAMPAIGNS`, `REWARDS`) dan berlabel simulasi. Hapus file database untuk mulai bersih.

## Yang perlu disesuaikan saat pemasangan
| apa | di mana |
|---|---|
| ambang cairan, batas luas ukuran | `rvm_kp/config.py` (pakai tabel "Botol terakhir" di `/operator`) |
| zona periksa, minimal suara, ambang tanpa tutup | `app/config.py` (`ZONE_X`, `MIN_VOTES`, `NOCAP_FRAC`) |
| poin per ukuran, masa berlaku klaim, batas diam sesi | `app/config.py` |
| kampanye donasi & katalog hadiah | `app/db.py` (baris baru masuk saat server mulai; yang sudah ada tidak ditimpa) |

## Keterbatasan & catatan
- **Bonus merek belum aktif** (`BRAND_BONUS = 0`): model merek belum dikerjakan ulang (lihat `docs/00_STATUS_PROYEK.md`).
- Kelemahan model ikut terbawa: tutup hitam (Coca-Cola Zero, Powerade), botol kosong berlabel penuh berwarna terbaca berisi, botol 400 ml terbaca kecil.
- Akun hanya nomor HP + PIN, tanpa OTP; cukup untuk pengembangan, **bukan** untuk poin bernilai uang.
- Font (Bricolage Grotesque, Geist) dimuat dari Google Fonts; tanpa internet layar memakai font cadangan sistem (tampilan tetap utuh). Untuk kiosk luring penuh, unduh file `.woff2` ke `static/fonts/` dan ganti `<link>` dengan `@font-face`.
- `python -m rvm_kp.buat_paket_nano` ikut menyalin `app/` ke paket Nano (tanpa `app/data/`, jadi database di Nano mulai kosong). Di Nano, dari `~/rvm`: `pip install -r app/requirements.txt` lalu `python -m app --backend tensorrt --cam 0`.
- Di Jetson Nano: tampilkan kiosk dengan Chromium layar penuh di luar Docker, mis. `chromium-browser --kiosk http://localhost:8000`. Pratinjau kamera sengaja kecil (640 px, 10 fps) agar CPU Nano tidak habis untuk JPEG.

## Berkas
```
app/
  __main__.py   argumen baris perintah
  config.py     poin, aturan sesi, zona, PLC
  server.py     FastAPI: halaman, WebSocket /ws, REST /api/*, MJPEG /video_feed
  engine.py     thread kamera + RVMAnalyzer + status sesi (analisis hanya saat sesi aktif)
  tracker.py    pelacak antar-frame + keputusan per botol
  simulator.py  belt virtual untuk --sim
  db.py         SQLite: sesi, botol, klaim, akun, ledger, kampanye, hadiah
  plc.py        konveyor via snap7 (opsional)
  static/       kiosk.html/.css/.js, m.html/.css/.js (HP), operator.html, base.css, icons.js
  DESIGN.md     arah visual & token desain
```
