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
