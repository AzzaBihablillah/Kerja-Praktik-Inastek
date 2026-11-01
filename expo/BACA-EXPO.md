# RVM di Capstone Expo: cara menyalakan

Semua sudah terpasang di Jetson Nano (disiapkan 1 Nov 2026). Besok cukup colok perangkat, lalu klik ikon.

## Langkah

1. **Daya Jetson:** pakai adaptor barrel **5V 4A** dengan jumper J48 terpasang. Adaptor 5V 2A membuat Jetson restart sendiri saat model bekerja (sudah terjadi 24 Okt).
2. **Colok:** kamera C922 (USB), ESP32 talang (USB), kabel LAN ke jaringan yang sama dengan PLC, monitor, keyboard dan mouse.
3. **Tunggu 5 detik**, lalu di desktop Jetson klik dua kali:
   - **1 Mulai RVM (belt jalan)**: konveyor dikendalikan PLC. Pakai **hanya bila area belt sudah aman**.
   - **2 Uji RVM (tanpa belt)**: untuk mengecek kamera dan layar tanpa menggerakkan konveyor.
   - **3 Stop RVM**: menghentikan aplikasi dengan aman (konveyor ikut berhenti).
4. Layar kiosk terbuka penuh dalam ±10–60 detik (model dimuat dulu). Keluar dari layar penuh: **Alt+F4**.
5. **Halaman operator** (dari laptop atau HP di jaringan yang sama): `http://<IP Jetson>:8000/operator`. IP muncul di log, atau ketik `hostname -I` di terminal Jetson. Di lab: `http://192.168.0.149:8000/operator`.

Kamera dan port ESP32 dicari otomatis setiap kali ikon diklik. Kalau ESP32 tidak tercolok, aplikasi tetap jalan dan keputusan talang hanya dicatat.

## Cara memastikan aman: ikon "4 Cek RVM"

Klik setelah ikon 1 atau 2 dan layar kiosk muncul. Aman bila:

| Baris | Harus terbaca |
|---|---|
| APLIKASI | `OK` (status siap). `TUNGGU` = model masih dimuat, cek lagi 30 detik kemudian |
| KAMERA | `OK`, lebih dari 5 gambar/detik |
| PLC | `OK` saat ikon 1 dipakai (`MATI` wajar di ikon 2) |
| TALANG ESP32 | `OK /dev/ttyUSB0` bila talang dipakai |
| DAYA | Jetson tidak pernah restart sendiri |

Lalu uji 3 botol sebelum pengunjung datang: botol kosong tanpa tutup harus **diterima**, botol bertutup **ditolak**, botol berisi air **ditolak**.

**Catatan daya:** Jetson mencatat peringatan arus (OC) saat beban berat, sekitar 600 kali dalam 50 menit uji 1 Nov, hampir semuanya saat model bekerja. Jetson memperlambat diri sebentar, tidak mati, dan uji tetap lancar (±1,2 detik per botol). Bila Jetson restart sendiri saat expo, pindah ke cadangan laptop.

## Kalau ada masalah

| Gejala | Yang dilakukan |
|---|---|
| Muncul "Kamera belum tercolok" | Colok ulang kamera, tunggu 5 detik, klik ikon lagi |
| Layar kiosk tidak muncul dalam 2 menit | Klik **3 Stop RVM**, lalu ikon 1 lagi. Log: `~/rvm/expo/log/` |
| Jetson mati atau restart sendiri | Hampir pasti adaptor kurang kuat: ganti ke 5V 4A |
| Konveyor tidak bergerak | Cek kabel LAN dan PLC menyala; PLC di `192.168.0.1` |
| Jetson tetap bermasalah | **Cadangan laptop:** klik dua kali `expo\Mulai-RVM-laptop.bat` di folder proyek laptop (kamera C922 = index 1, ESP32 = COM7) |

## Yang sudah diuji (1 Nov)

- Model aktif (detektor ver9c, tutup v3, isi v3, merek v2) berjalan sebagai engine TensorRT FP16 di Jetson.
- Uji aplikasi utuh di Jetson (kamera tiruan memutar 84 botol foto belt 29 Okt, PLC dan talang mati): 84/84 tercatat dan sama dengan simulasi di laptop, berhenti per botol ±1,2 detik, model dimuat 47 detik. Ini uji fungsi, bukan akurasi (model final sudah melihat foto itu). Hasil: `~/rvm/datasets/4_eksperimen/uji_app/`.
- Ikon desktop, pesan "Kamera belum tercolok", layar kiosk Chromium, dan Stop sudah dicoba di Jetson.
- Chromium di Jetson lambat: jendela kiosk bisa baru muncul 30–60 detik setelah ikon diklik. Tunggu, jangan klik ikon berkali-kali.
- RAM Jetson hampir penuh saat aplikasi + Chromium jalan (swap 8 GB menolong). Jangan buka aplikasi lain di Jetson saat expo; buka halaman operator dari laptop atau HP.
- Belum diuji: kamera sungguhan, ESP32, dan PLC dari Jetson (baru tersedia besok). Lakukan uji singkat dengan ikon **2 Uji RVM** sebelum pengunjung datang, lalu baru ikon **1 Mulai RVM** setelah area belt aman.

## Isi folder

- `mulai_rvm.sh`, `stop_rvm.sh`, `cek_rvm.sh`: dipanggil ikon desktop (salinan ikon: `rvm_*.desktop`).
- Container Docker `rvm_expo`: berisi aplikasi dan paketnya. Container lama `rvm` tidak dipakai. Cadangannya image `rvm_expo:siap_expo`; bila container hilang, buat ulang di terminal Jetson:
  `docker run -d --name rvm_expo --runtime nvidia --network host --privileged -v /dev:/dev -v /home/jetson/rvm:/rvm -v /etc/nv_tegra_release:/etc/nv_tegra_release:ro -v /proc/device-tree/model:/etc/rvm_board_model:ro -w /rvm rvm_expo:siap_expo sleep infinity`
- Kode versi sebelumnya di Jetson diarsipkan di `~/rvm/_arsip_kode/`.
