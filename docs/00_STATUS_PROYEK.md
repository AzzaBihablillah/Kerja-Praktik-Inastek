# Status Proyek RVM — KP PT Inastek (dibaca pertama di room/sesi baru)

Terakhir diperbarui: **21 Okt 2026** (sore: aplikasi kiosk `app/`). Ringkasan konteks, keputusan, status, lokasi file, dan cara melanjutkan.
Rincian percobaan dan angka: `docs/CATATAN-PERCOBAAN-TA.md`. Riwayat dan keputusan: `docs/CATATAN-PROYEK.md`.

## 1. Konteks dan tujuan
- **Proyek:** Reverse Vending Machine (RVM) botol PET, Kerja Praktik di PT Inastek (DTETI UGM), juga bahan **tugas akhir**. **Laporan KP sudah selesai** (`docs/laporan_kp/`); yang dikumpulkan sekarang adalah bahan TA.
- **Alur mesin:** kamera di atas konveyor memeriksa tiap botol:
  1. botol PET atau bukan;
  2. bertutup atau tidak (bertutup ditolak; botol **berring** tanpa tutup = sah);
  3. berisi **cairan berwarna** atau kosong;
  4. **kategori ukuran** untuk poin;
  5. **merek** sebagai bonus poin.
- **Tujuan mitra:** kamera **menggantikan load cell**; sistem tidak boleh bergantung pada load cell.
- **Lingkup deteksi cairan:** hanya **cairan berwarna**. **Air bening di luar lingkup** (disepakati dengan mitra 16 Okt); ditulis sebagai keterbatasan, bukan pekerjaan. Jangan mengusulkan percobaan air bening.
- **Penerimaan:** semua botol PET diterima. Merek hanya bonus: whitelist Coca-Cola, Fanta, Sprite (termasuk Nipis Mint); lainnya poin default.
- **Ukuran (keputusan user):** kecil ≤ 350 ml, tanggung 351–800 ml (botol ritel 390 ml = tanggung), besar > 800 ml.
- **Target perangkat:** **Jetson Nano Developer Kit B01, 4 GB** (dipastikan 21 Okt dari label P3450 / 180-13449-DAAF-B01 / 945-13450-0000-100): JetPack 4.6.x maksimal (Ubuntu 18.04, Python 3.6, CUDA 10.2, TensorRT 8.2), GPU Maxwell 128 core, FP16 (tanpa INT8). Ultralytics modern hanya lewat Docker JetPack 4. Daya wajib barrel 5V 4A + jumper J48. Jika Jetson tidak bisa dipakai, sistem dijalankan di laptop (RTX 4050) dengan kode yang sama; angka kecepatan dilaporkan sesuai perangkat yang benar-benar diuji.

## 2. Aturan kerja
- Jangan menghapus data. Tanya dulu sebelum memindah data. Pindah ke `_arsip/`, bukan hapus.
- Jangan menyalakan belt/PLC tanpa konfirmasi user bahwa area belt aman.
- `open_vino/` dan isi `testing/` (kecuali `testing/cap version test/`) milik tim (Capstone 1, hanya deteksi botol/non-botol). Jangan diubah.
- Satuan evaluasi = **botol**, bukan foto. Botol isi ulang dengan cairan sama diuji bersama (per kelompok cairan).
- Data tambahan harus foto belt nyata, **botol berisi dan kosong di sesi yang sama**. Data internet/luar sudah dicoba 10+ cara dan selalu memperburuk.

## 3. Status model (semua di `datasets/3_model_final/`)
| modul | model | hasil utama | status |
|---|---|---|---|
| botol vs non-botol | `botol_ver8b.pt` (ver7 + foto belt); kandidat `botol_ver9b_kandidat.pt` (+ negatif TACO-tempel & botol ber-alpha) | ver8b: botol belt 100%, tapi menerima 19,5% sampah TACO-tempel; ver9b: 3,0%, tanpa regresi (CATATAN 4b) | ver8b dipakai; ver9b menunggu uji negatif NYATA di belt |
| tutup | `cap_belt_final.pt` | CV 5-fold per botol 98,3% (CI 95,2–99,4) di sesi 13 Okt; **uji independen sesi 16/20 Okt 71%**: gagal di **tutup hitam** (Coke Zero, Powerade), lainnya ±94% | perlu data tutup hitam; server masih `cap_v4_640.pt` |
| cairan berwarna | `cairan_RH.pt` (utama), `cairan_RA.pt` | validasi per kelompok cairan (21 botol, 13 cairan): **AUC 0,992**; biru dikenali walau tidak dilatih; putih/pucat dikenali | siap; demo live |
| ukuran | luas kotak botol (batas 228k / 357k px², 1080p) | 21/22 botol benar (botol 400 ml terbaca kecil) | perlu kalibrasi per pemasangan |
| merek | `export_model/brand_best_ver5` (tim) | menerima BIG Cola/BIG Strawberry/Vit sebagai coca_cola; Sprite Nipis Mint luput | belum dikerjakan ulang |

**Kelemahan yang masih ada:**
- (a) Botol **kosong berlabel penuh berwarna** (Minute Maid, Isoplus) terbaca berisi.
- (b) Ambang cairan perlu dikalibrasi saat pemasangan (skor RH jenuh mendekati 1).
- (c) Data uji kecil (21 botol berwarna, 2 sesi).
- (d) Cairan merah murni belum ada di data.

## 4. Peta direktori (sejak 21 Okt; lama → baru di `datasets/PINDAH-LOG.csv`)
```
datasets/                       (tidak ikut git; ±10 GB; BACKUP manual!)
  1_foto_mentah/                foto asli kamera (tak tergantikan): 2026-10-13_conveyor_v1 (cap), 2026-10-16_water_v1, 2026-10-20_water_v1 (cairan), shoot studio Sep
  2_dataset_cap/                dataset tutup turunan (cap_belt_*, cap_v4, _rf_export, shoot_*)
  3_model_final/                model siap pakai (+ dasar/ = bobot awal yolov8*)
  4_eksperimen/cap/             run training & CV model tutup
  4_eksperimen/cairan/          SEMUA keluaran percobaan cairan (crop, indeks, laporan *.txt, runs, final*/), log/ = log & flag
  4_eksperimen/cairan_pilot/    pilot 15 Okt (tidak dapat disimpulkan)
  5_data_luar/                  unduhan & data warisan (Roboflow, RF100, Open Food Facts, TACO, data tim lama)
  _arsip/                       eksperimen cap lama, gambar debug, foto cek kamera
scripts/                        (ikut git)
  alat/                         alat lapangan: capture_water, capture_free, capture_conveyor, live_liquid_demo, qc_water_set, water_qc
  pipeline_cap/                 pipeline model tutup (prelabel → build → cv → final), cap_decider (dipakai server PLC)
  pipeline_cairan/              pipeline cairan s00–s30 + common.py (urutan & fungsi: scripts/README.md)
  _arsip/                       skrip lama (path lama, tidak diperbarui)
docs/                           00_STATUS_PROYEK (ini), CATATAN-PERCOBAAN-TA, CATATAN-PROYEK, PETA-DATASET, DAFTAR-POTRET, laporan_kp/, _arsip/
testing/cap version test/server.py   server PLC (FastAPI + snap7, bit MB10.0); path sudah diperbarui
rvm_kp/                         sistem analisis KP (satu kode, backend pytorch/onnx/openvino/tensorrt), lihat rvm_kp/README.md
app/                            aplikasi kiosk "Tukar Botol" (FastAPI + HTML biasa): layar mesin, QR klaim poin, akun/donasi/hadiah simulasi, panel operator; lihat app/README.md
open_vino/, export_model/       milik tim (aplikasi Capstone 1, model bottle ver7 & brand ver5)
```

## 5. Cara menjalankan (dari folder proyek, venv)
```bash
venv/Scripts/python.exe scripts/alat/capture_water.py --list
venv/Scripts/python.exe scripts/alat/capture_water.py --cam 1
venv/Scripts/python.exe scripts/alat/live_liquid_demo.py --cam 1
venv/Scripts/python.exe scripts/alat/qc_water_set.py --dir 1_foto_mentah/<folder>
venv/Scripts/python.exe scripts/pipeline_cairan/s28_cv_liquidgroup.py --recipe RH
venv/Scripts/python.exe scripts/pipeline_cairan/s28b_report.py RA RH
```
Aplikasi kiosk: `venv/Scripts/python.exe -m app --sim` (tanpa kamera/model) atau `-m app --backend pytorch --cam 1`; buka http://localhost:8000 (kiosk), /operator, dan QR klaim dari HP satu jaringan.
Kamera belt (C922) biasanya index 1 setelah dicolok (index 0 = webcam laptop). Alat potret menolak jalan jika bukan 1080p.

## 6. Pekerjaan terbuka (urut prioritas)
0. **Tutup hitam** (Coca-Cola Zero = produk mitra, Powerade): keputusan user 21 Okt = pakai foto sesi 16 & 20 Okt yang sudah ada. **Pra-label SIAP** (`scripts/pipeline_cap/prelabel_water.py` → `datasets/2_dataset_cap/cap_water_v1/`: 217 foto, 215 kotak model + 2 PLACEHOLDER, 60 foto tutup hitam wajib dicek di `PERLU-DITINJAU.txt`, zip unggah `cap_water_v1_roboflow.zip` 112 MB). MENUNGGU: user koreksi kotak di Roboflow → ekspor → latih ulang model tutup (13 Okt + 16/20 Okt) + CV per botol (`docs/CATATAN-PERCOBAAN-TA.md` C-9).
1. Foto **botol kosong berlabel penuh** (botol bekas) di sesi yang sama dengan botol berisi; tambah **cairan merah** (sirup di botol bekas).
2. **(sebagian selesai 21 Okt)** `rvm_kp/`: kode sistem multi-backend, requirements laptop/Jetson, ekspor ONNX & OpenVINO (`datasets/3_model_final/ekspor/`), benchmark awal (`datasets/4_eksperimen/benchmark/`). **21 Okt sore:** profil `nano` (720p, tutup hanya di area botol, FP16) diuji: keputusan tidak berubah, komputasi −36%; ONNX opset 12 untuk TensorRT 8.2 terverifikasi; panduan `rvm_kp/JETSON-NANO.md`; paket salin `datasets/_paket_nano/` (171 MB). Sisa: engine TensorRT & benchmark DI Nano.
3. Uji live di belt berjalan (butuh konfirmasi keamanan belt).
4. Model merek `produk_mitra` vs lain (relabel `brand2`, ±12% `unknown` = produk Coca-Cola).
5. Negatif non-botol: TACO-tempel diuji 21 Okt (ver9 ditolak: jalan pintas tempelan; ver9b kandidat, CATATAN 4b); tetap perlu foto NYATA di belt: plastik bening diremas, kantong, gelas plastik, kaleng.
5b. **Uji iseng penguji (spoof):** foto botol di layar HP, gambar botol dicetak di kertas/ditempel ke busa. Kamera tunggal dari atas tidak bisa membedakan 2D vs 3D secara pasti. Rencana: potret 20-30 contoh tiap jenis di belt (HP berbagai kecerahan, cetak doff/glossy, busa bergambar) sebagai `non_bottle` + uji per jenis; isyarat yang bisa dipelajari: bingkai HP, moire layar, kertas tanpa transparansi/kilap PET. Pertahanan berlapis: model + keputusan multi-frame; opsi sensor murah non-load-cell (ToF/ultrasonik tinggi objek) bila mitra mengizinkan.
6. Jetson Nano B01 4 GB (belum diservis): pasang JetPack 4.6.x + Docker `ultralytics/ultralytics:latest-jetson-jetpack4`, buat engine TensorRT FP16 di Nano, benchmark. Jika gagal, sistem di laptop.
7. **Aplikasi kiosk `app/` (21 Okt, versi pengembangan):** alur sesi → keputusan per botol (voting antar-frame) → poin per ukuran → QR klaim → simpan ke akun / donasi; diuji di mode simulasi dan dengan model asli pada foto belt. Belum: uji live di belt, kalibrasi zona periksa, bonus merek, salin `app/` ke paket Nano.

## 7. Riwayat singkat
| tanggal | peristiwa |
|---|---|
| Sep | data studio, model cap v1–v4 (v4 terbaik) |
| 13 Okt | dataset belt 30 botol; `cap_belt_final` (CV 98,3%) |
| 15 Okt | pilot cairan (confound) |
| 16 Okt | rapat mitra: cairan berwarna saja; set 16 Okt (5 botol) |
| 17–18 Okt | `ver8`; dua tahap > detektor 3 kelas |
| 19–20 Okt | data luar & preprocessing gagal; kurva belajar; cek jalan pintas sesi |
| 20 Okt | sesi 20 Okt (16 botol); validasi per kelompok cairan |
| 21 Okt | geser hue (RH) terbaik, AUC 0,992; model final; direktori dirapikan |
