# Status Proyek RVM — KP PT Inastek (dibaca pertama di room/sesi baru)

Terakhir diperbarui: **30 Okt 2026** (butir 16 aktif, butir 17). Ringkasan konteks, keputusan, status, lokasi file, dan cara melanjutkan.
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
- **Usulan perubahan lingkup (user, 23 Okt):** botol berisi **air bening juga ditolak** (definisi 'berisi apa pun' vs 'kosong'), dipicu temuan uji live (air bening di botol air mineral terdeteksi 5/5). Sedang diuji: `scripts/pipeline_cairan/s34_berisi_bening.py`. **Perlu dikabarkan ke mitra** sebagai tambahan; di TA ditulis sebagai temuan, bukan janji.
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
| botol vs non-botol | **`botol_ver9c.pt`** (22 Okt; ver7 + belt + TACO & TrashNet tempel, botol ber-alpha); `botol_ver8b.pt` cadangan | uji NYATA belt: non-botol diterima 7/32 (ver8b 13/32), botol PET 64/64; plastik bening diremas masih lolos (CATATAN 4b) | dipakai `rvm_kp` |
| tutup | **`cap_belt_v2.pt`** (21 Okt; 13 Okt + 16/20 Okt) | CV 5 fold per merek **99,0%** (CI 97,4–99,6), tutup hitam 95% (v1 `cap_belt_final`: 88,2% / 31,7%); mode ROI nano 416/½ = 99,0% | dipakai `rvm_kp`; server PLC lama masih `cap_v4_640.pt` |
| cairan berwarna | `cairan_RH.pt` (utama), `cairan_RA.pt` | validasi per kelompok cairan (21 botol, 13 cairan): **AUC 0,992**; biru dikenali walau tidak dilatih; putih/pucat dikenali | siap; demo live |
| ukuran | luas kotak botol (batas 228k / 357k px², 1080p) | 21/22 botol benar (botol 400 ml terbaca kecil) | perlu kalibrasi per pemasangan |
| merek | **kandidat `merek_coca_kandidat.pt` (24 Okt, bayangan di app `--kandidat`; belum aktif, bonus 0).** Keputusan user 23 Okt: hanya 2 kelas **coca_cola** (semua produk Coca-Cola Company) vs **non_coca_cola**; potongan botol utuh 256 (`m02.brand_crop`), YOLOv8n-cls. Dasar = foto belt saja 551/37 produk (`scripts/pipeline_merek/m01..m06`) | CV per botol (fold sama, uji hanya belt): **A belt saja** coca 222/261, non salah terima 11; **D** +brand2+OFF teraudit: 220/261, salah 26; **E** D dgn belt x6: 224/261, salah 14. Produk tak dikenal: A 158, D 154, E 154 /261 (hanya Sprite terbantu 13→31/36). Laporan `datasets/4_eksperimen/merek/cv_campur_report.txt` | data tambahan TIDAK membantu berarti → pakai A; perbaikan = tambah botol fisik (Nutriboost 1 botol 0/16, Powerade 2 botol, Sprite, Frestea). Audit selesai: `merek/audit_final.csv` (halaman https://claude.ai/artifact/R8XTKDdWJt2NmiFohE7RPC). B (brand2 saja) 218/261, salah 14; C (OFF saja) 222/261, salah 21 → OFF penyebab utama salah terima (Sunbay, Vit); tak ada varian mengalahkan A (CATATAN §5) |

**Kelemahan yang masih ada:**
- (0) **Botol berlabel penuh**: jenis dengan celah bening → diperbaiki dengan data (sesi pembuktian bagian C); jenis TANPA celah → batasan fisik kamera, ditulis di TA sebagai batasan (perlu sensor berat/ditolak) HANYA bila latih ulang tetap gagal (CATATAN bagian 8).
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
0. ~~Tutup hitam~~ **SELESAI 21 Okt**: `cap_belt_v2` (CATATAN C-10/C-11). Sisa: uji live di belt; tambah produk bertutup hitam lain bila ada.
1. Foto **botol kosong berlabel penuh** (botol bekas) di sesi yang sama dengan botol berisi; tambah **cairan merah** (sirup di botol bekas).
2. **(sebagian selesai 21 Okt)** `rvm_kp/`: kode sistem multi-backend, requirements laptop/Jetson, ekspor ONNX & OpenVINO (`datasets/3_model_final/ekspor/`), benchmark awal (`datasets/4_eksperimen/benchmark/`). **21 Okt sore:** profil `nano` (720p, tutup hanya di area botol, FP16) diuji: keputusan tidak berubah, komputasi −36%; ONNX opset 12 untuk TensorRT 8.2 terverifikasi; panduan `rvm_kp/JETSON-NANO.md`; paket salin `datasets/_paket_nano/` (171 MB). **24 Okt:** engine TensorRT FP16 5 model dibuat DI Nano (±11 menit) dan benchmark: **156 ms/frame ≈ 6,4 FPS**, merek 16,6 ms/potongan (CATATAN L-22). Sisa: benchmark PyTorch Nano & kesesuaian keputusan, kamera di Nano, cek daya (Nano restart sekali saat beban puncak).
3. ~~Uji live di belt~~ **dilakukan 22 Okt** (CATATAN L-17): aplikasi + PLC, mode berhenti-lalu-pindai; tutup 30/30, kosong 16/16, berisi 10/14 (luput warna pucat). Data scan untuk latih ulang di `1_foto_mentah/2026-10-22_scan_app/`.
3b. **Selesai 23 Okt dini hari** (CATATAN L-18–L-20): `botol_ver10_kandidat` (CV per benda: non-botol 0/79, botol 154/154, tanpa regresi), `cairan_RN` + opsi gabungan RH+RN (40 vs 45 salah; botol kosong berlabel penuh masih gagal), ukuran berbasis panjang (24/26 vs 22/26). **Belum diaktifkan** → uji di sesi pembuktian: `docs/DAFTAR-POTRET-SESI-PEMBUKTIAN.md`.
3c. Data luar (Open Food Facts) dilabel ulang Claude: menunggu validasi user (`4_eksperimen/cairan/RELABEL-LUAR-*.jpg`), lalu jalankan `s33_luar_ketat.py cv`.
4. **Model merek (23 Okt):** baseline & uji data tambahan selesai (lihat tabel §3 dan CATATAN §5). Sisa: (a) hasil B/C SUDAH dicatat di CATATAN §5 (tabel A–E); (b) sesi Senin: potret botol fisik produk mitra tambahan (≥4–5 botol per produk, prioritas Nutriboost, Powerade, Sprite, Frestea) — SUDAH ditambahkan sebagai bagian G di `DAFTAR-POTRET-SESI-PEMBUKTIAN.md`; (c) SUDAH 24 Okt: model final kandidat `3_model_final/merek_coca_kandidat.pt` (resep A, 551 foto belt, `m07_final.py`, kelas coca_cola/non_coca_cola), modul `rvm_kp/merek.py` (brand_crop satu sumber dengan pipeline latih), dipasang di app `--kandidat` (bayangan: kolom `p_coca_kandidat`/`merek_kandidat` di SCAN_KANDIDAT.csv, baris MEREK di lembar cek; bonus tetap 0). Diuji offline pada frame 22 Okt (frame itu ikut data latih → hanya uji pipa; p jalan = p potongan latih, selisih 0). (d) setelah Senin: nilai merek pada botol baru, latih ulang dengan botol bagian G, lalu (bila disetujui) aktifkan bonus `BRAND_BONUS`.
5. **Daftar potret: `docs/DAFTAR-POTRET-22-OKT.md`** (negatif nyata + uji iseng + kontrol PET, ±140 foto). Negatif non-botol: TACO-tempel diuji 21 Okt (ver9 ditolak: jalan pintas tempelan; ver9b jalan pintas sumber asing; ver9c kandidat terbaik, CATATAN 4b); tetap perlu foto NYATA di belt: plastik bening diremas, kantong, gelas plastik, kaleng.
5b. **Uji iseng penguji (spoof):** foto botol di layar HP, gambar botol dicetak di kertas/ditempel ke busa. Kamera tunggal dari atas tidak bisa membedakan 2D vs 3D secara pasti. Rencana: potret 20-30 contoh tiap jenis di belt (HP berbagai kecerahan, cetak doff/glossy, busa bergambar) sebagai `non_bottle` + uji per jenis; isyarat yang bisa dipelajari: bingkai HP, moire layar, kertas tanpa transparansi/kilap PET. Pertahanan berlapis: model + keputusan multi-frame; opsi sensor murah non-load-cell (ToF/ultrasonik tinggi objek) bila mitra mengizinkan.
6. Jetson Nano B01 4 GB (belum diservis): pasang JetPack 4.6.x + Docker `ultralytics/ultralytics:latest-jetson-jetpack4`, buat engine TensorRT FP16 di Nano, benchmark. Jika gagal, sistem di laptop.
7. **Aplikasi kiosk `app/` (21 Okt, versi pengembangan):** alur sesi → keputusan per botol (voting antar-frame) → poin per ukuran → QR klaim → simpan ke akun / donasi; diuji di mode simulasi dan dengan model asli pada foto belt. Belum: uji live di belt, kalibrasi zona periksa, bonus merek, salin `app/` ke paket Nano.

17. **30 Okt:** (a) tutup v4w warna buatan (audit user, latih Colab A100 akun Azza) TIDAK LULUS kriteria per hari (27 Okt salah tolak +1) -> tutup v3 tetap aktif (CATATAN L-33); (b) uji aplikasi utuh dengan kamera tiruan 469 botol: tanpa galat, identik dengan simulasi (L-34); (c) sesi belt berikutnya siap: kandidat bayangan = tutup v4w, label tutup_asli/biru/tukar, protokol `docs/PROTOKOL-SESI-BELT-BERIKUTNYA.md`.
16. **Latih ulang isi/tutup/merek dengan data belt 27 + 29 Okt (29 Okt malam, CATATAN L-32):** validasi per hari (latih tanpa satu hari, uji di hari itu). Isi `cairan_isi_v3_kandidat.pt`: 29 Okt kosong 149/150 (RH aktif 122/150), ±1/6 66/75 (RH 60), penuh 53/62 (RH 52). Tutup `cap_belt_v3_kandidat.pt`. Merek v3 = v2 -> tetap v2. **Simulasi keputusan** (`scripts/alat/replay_keputusan.py`): aktif 449/469 (salah terima 12, salah tolak 8) -> semua baru 461/469 (7, 1). Semua kandidat diekspor ONNX/OpenVINO. Bug kandidat bayangan (kotak bergerak) diperbaiki di `app/engine.py`; angka bayangan langsung 27–29 Okt tidak sahih. **AKTIF sejak 30 Okt (persetujuan user): tutup v3 + isi v3** di `rvm_kp/config.py MODEL_SETS` (detektor ver9c & merek v2 tetap); engine TensorRT v3 belum dibuat di Nano.
15. **Latih ulang dengan data belt 27 Okt (28 Okt dini hari, CATATAN L-27/L-28):** (a) cairan: `scripts/pipeline_cairan/s36_berisi_27okt.py`, CV per sesi 27 Okt -> berisi_v2 TIDAK lebih baik dari kandidat s35 pada sesi tak dilatih (kosong salah 3/72 = sama; air 24 vs 27/35, tetapi fold air tak pernah melihat air belt). Final `3_model_final/cairan_berisi_v2_kandidat.pt` disimpan, BELUM dipakai; nilai pada sesi air bening berikutnya (offline). (b) merek: audit merek botol air sesi5 selesai (10 Coca-Cola Co./25 bukan; model benar 30/35) -> `scripts/pipeline_merek/m08_belt27.py` (CV 2 fold + final `merek_coca_v2_kandidat.pt`). (c) detektor ver12: audit kotak 132 benda selesai (3 dibuang) -> `scripts/pipeline_botol/ver12_belt27.py` (CV 2 fold per pasangan sesi + final `botol_ver12_kandidat.pt`). Model aktif TIDAK diubah. **Hasil CV:** merek v2 104/107 vs m07 101/107 (non-Coca salah bonus 1/75 vs 3/75; Frestea sesi1 tetap gagal; McNemar p=0,38); ver12 = ver11 pada deteksi (bias acuan) tetapi alarm palsu 21 -> 4/387 frame. Sisa: uji pipa tutup dengan kotak ver12, usul ganti kandidat bayangan app ke ver12 + merek v2 (+cairan v2 offline) untuk sesi berikutnya (perlu persetujuan user), tutup biru (perlu data), sesi air bening tanpa tutup, keputusan aturan cairan (user/mitra). Cek ambang Belum di-commit. jejak data tak-teraudit: CATATAN L-29 (ambang bukan penyebab utama; pemilahan Claude tidak masuk model aktif). Belum di-commit.
14. **Sesi belt 27 Okt sore (CATATAN L-25):** 72 botol kosong + 22 bukan botol + 35 air, 6 sesi logis di `2026-10-27_scan_app/sesi1..6_*` (BACA.txt). Kandidat lebih baik: keputusan 70/72 vs 67/72, cairan 70/72 vs 60/72; bukan botol 22/22 ditolak. Perubahan app: benda asing tidak ditahan (ikut ke pembuangan), ambang non_bottle 0,60, kandidat di thread terpisah, folder per sesi. Sesi air 053-059: tutup diaudit (34 bertutup) tetapi ISI tidak pasti -> tidak dipakai menilai cairan. SISA: sesi air bening BERSIH (diisi sendiri, tanpa tutup, per level isi) sebelum memutuskan aktivasi cairan_berisi; setel pencatat 'ada benda' (noisy); tutup biru muda lolos (Alfamart 1,5 L) -> data latih tutup; merek Frestea/Vit/Fitri; usul aktivasi cairan_berisi (perlu persetujuan user).
13. **Detektor ver11 kandidat (27 Okt, `datasets/3_model_final/botol_ver11_kandidat.pt`, CATATAN L-24):** non-botol nyata diberi kotak non_bottle (audit user) + bobot seimbang per jenis benda. CV foto nyata: non-botol diterima 33->5/81, ditandai non_bottle 50->78/81, botol 154/154, botol salah non_bottle 24->2/154; data lain (TrashNet/TACO) setara atau lebih baik, tidak bias ke plastik; old_test recall botol 0,99, botol tempelan belt 95%. BELUM aktif & belum dijadikan kandidat bayangan di app (`app/config.py KANDIDAT` masih ver10) -> usul: ganti kandidat bayangan ke ver11 untuk sesi belt berikutnya (perlu persetujuan user).
12. **"Ada benda" mode catat + audit kotak non_bottle (26 Okt malam):** `app/pencatat_benda.py` (patokan belt kosong per piksel, BENDA.csv; diuji dengan belt kosong tiruan: kosong 0 salah alarm, botol/benda tak dikenal/kamera tergeser tercatat benar) -> butuh sesi belt (mulai sesi dengan belt kosong beberapa detik). Audit kotak non_bottle 86 foto 22 Okt (usulan ver9c, `datasets/4_eksperimen/nonbotol_audit/usulan.csv`): https://claude.ai/artifact/HWEc2Tc8bkcMBLuvBmVgKs (koleksi db `nonbotol_audit`) -> setelah audit: latih detektor ver11 = resep ver10 + kotak non_bottle nyata, CV per benda. Keputusan user 27 Okt: botol plastik NON-minuman (semprot, Listerine) = bukan botol, mesin menolak (bahan beda, bisa merusak) -> tetap contoh non_bottle.
11. **Penyapu V terima/buang (26 Okt, alat servo tim belum terpasang):** penjadwal gagal-aman di `app/sortir.py` (sapuan BUANG berkala + TERIMA terjadwal, jam = detik belt berjalan; uji logika deterministik 7/7 lulus, uji simulasi OK). Sisa: kalibrasi `SORTIR_TRAVEL_SEC` & `SORTIR_BUANG_EVERY` di mesin, protokol kendali servo dari tim (usulan serial T/B/N).
10. **Benda bukan botol (26 Okt, laporan user: non-botol tidak tersingkir):** app dulu MENGABAIKAN kelas `non_bottle` dan tidak punya perintah penyingkiran (PLC hanya MB10.0 jalan/berhenti). Kini: `non_bottle` (ver9c) di zona -> berhenti, tolak 'bukan botol', belt ditahan sampai diambil; kaitan servo `app/sortir.py` (jalur terima/tolak buatan tim, belum terpasang). Offline 22 Okt: non-botol 71/86 tertolak (41 'bukan botol' + 30 lewat tutup/isi), 15/86 tak terdeteksi sama sekali, botol 0/94 salah ditahan. Sisa: (a) deteksi "ada benda" berbasis belt kosong (`rvm_kp/benda.py`, prototipe) HARUS dikalibrasi di belt (data lama tak punya foto belt kosong; perbandingan per-baris gagal karena sabuk miring + pantulan); (b) ver10 tidak punya kelas non_bottle -> bila ver10 diaktifkan, perlu ver9c untuk non_bottle atau latih ulang ver10 dengan kelas non_bottle.
9. **Kiosk capturer (24 Okt, `app --ambil-data`):** app untuk demo sekaligus pengumpul data: label kelompok di `/operator` (isi/tutup/produk) dicatat per botol di `SCAN.csv` + `KELOMPOK.csv`, dibandingkan dengan keputusan model aktif & kandidat (penghitung langsung, daftar beda), tanpa kode klaim; `lembar_cek.py --beda --acak N`. Diuji offline (frame 22 Okt) & simulasi. Belum dipakai di belt nyata.
8. **Ketahanan medan & warisan (rinci di CATATAN §8):** (a) SUDAH 24 Okt (`k01_uji_ketahanan.py`): tutup tahan gangguan ringan (58–64/64) tapi bias di studio (tanpa tutup 60/175); cairan gagal di medan lain (studio kosong benar 9/350); ukuran rusak bila jarak kamera berubah (37/64); merek peka noise/blur/warna cahaya; (b) dokumen batas operasi (ukur kamera/cahaya/belt saat sesi Senin); (c) panduan "pindah medan" (kalibrasi ukuran → kumpul 100–200 foto/model via app → label → fine-tune → CV + syarat lolos). Rekomendasi: ruang pindai tertutup + lampu sendiri.

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
