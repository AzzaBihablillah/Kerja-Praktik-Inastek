# Catatan Percobaan untuk Bahan Tugas Akhir

> **Lokasi file berubah 21 Okt 2026** (direktori dirapikan). Peta lokasi baru: `docs/00_STATUS_PROYEK.md`; daftar lengkap lama→baru: `datasets/PINDAH-LOG.csv`.


Disusun 20 Okt 2026 dari catatan kerja Sep–Okt 2026. Ini **catatan mentah yang bisa diacu**, belum berupa naskah TA. Penjelasan dan narasi menyusul saat TA dikerjakan.
Sumber rinci: `CATATAN-PROYEK.md`, `_arsip/HASIL-EKSPERIMEN-CAIRAN-18-OKT.md`, `PETA-DATASET-19-OKT.md`, `laporan_kp/laporan-deteksi-cap-nocap.md` (laporan KP, sampai v3), skrip di `datasets/` dan `scripts/pipeline_cairan/`.

## 0. Cara membaca
- **Satuan evaluasi = botol**, bukan foto. Foto dari botol yang sama hampir kembar, jadi hasil "40/40 foto" setara "5/5 botol". Selang kepercayaan per foto terlalu optimis.
- Kode data: **W16** = set cairan 16 Okt (5 botol, 100 foto); **C13** = belt 13 Okt (30 botol kosong, 180 foto); **P15** = sesi pilot 15 Okt (76 foto, tidak pernah dilatih, uji lintas sesi); **old_test** = 15% data warisan `bottle_detection` untuk regresi.
- Semua model berbasis YOLOv8n (deteksi: `yolov8n`; klasifikasi: `yolov8n-cls`). Bobot epoch terakhir dipakai (tanpa memilih epoch terbaik).

## 1. Lingkungan dan reprodusibilitas
| item | nilai |
|---|---|
| Python | 3.13 (venv) |
| Ultralytics | 8.4.54 |
| PyTorch | 2.11.0 + CUDA 12.8 |
| GPU | NVIDIA GeForce RTX 4050 Laptop 6 GB |
| OpenCV | 4.13.0 |
| Seed | 0, `deterministic=True`, `workers=0` (Windows) |
| Classifier | `yolov8n-cls`, 224 px, batch 32, 30 epoch (kurva belajar 15), AdamW otomatis, augmentasi: hsv_h=0 (warna cairan = label), hsv_s 0,2, hsv_v 0,3, fliplr/flipud 0,5, erasing 0,1, tanpa auto_augment |
| Detektor | dari bobot `bottle_best_ver7.pt`, 640 px, batch 16, 10 epoch, mosaic dengan `close_mosaic=3`, hsv_h=0 |
| Penyeimbangan kelas | kelas minoritas digandakan (maks 6×) di data latih |
| Data | di luar git: `datasets/` (±10 GB), `Downloads/bottle_detection`, `Downloads/brand2` |

Urutan skrip cairan: `s00` salin warisan → `s01` indeks belt dan kotak → `s02` crop → `s03` CV classifier → `s04/s04b/s04c` evaluasi → `s05–s06` data detektor → `s07–s08` detektor + evaluasi ujung-ke-ujung → `s09` ambang → `s10`, `s23` model final → `s11–s16` data luar dan preprocessing → `s20` registri → `s22` kurva belajar. Semua dijalankan dari `datasets/` dengan `../venv/Scripts/python.exe`.

## 2. Garis waktu ringkas
| tanggal | peristiwa |
|---|---|
| 2–6 Sep | data primer studio (13+15 botol), model cap v1–v3 (sintetis) |
| ≈Sep | v4 (jangkar data real, val/test 100% real, split per botol) = model terbaik |
| 13 Okt | dataset belt 1080p (30 botol), pelabelan Roboflow, CV 5-fold, model `cap_belt_final` |
| 15 Okt | pilot isi cairan (tidak bisa disimpulkan, confound) |
| 16 Okt | rapat mitra: lingkup deteksi cairan = hanya cairan berwarna; set cairan 5 botol |
| 17–18 Okt | eksperimen malam: `ver8` + classifier vs detektor 3 kelas |
| 19–20 Okt | data luar, preprocessing, peta dataset, kurva belajar |

## 3. Percobaan model cap (cap / no_cap)
Rincian: `laporan_kp/laporan-deteksi-cap-nocap.md` dan `CATATAN-PROYEK.md`.

| kode | tujuan | hasil | keputusan |
|---|---|---|---|
| C-1 | model dari data sintetis + konveyor lama (v1) | baik di metrik, goyah live | lanjut |
| C-2 | v2, degradasi gambar hanya pada no_cap | gagal: model belajar "tajam = cap" | **degradasi harus simetris** |
| C-3 | v3 degradasi simetris | tidak lebih baik | |
| C-4 | v4: jangkar data real, val/test 100% real, split per botol | test mAP50 0,988 (79 foto) | model terbaik live |
| C-5 | v4-8s (YOLOv8s), v5 (hard negative dasar botol) | tidak mengalahkan v4 | v4 dipakai |
| C-6 | fine-tune di belt 13 Okt (180 foto, 30 botol, CV 5 fold) | **98,3%** (CI 95,2–99,4) di 640 dan 960 | `cap_belt_final` |
| C-7 | baseline `cap_v4` di belt | 97,8% di 640; **92,8% di 960; ±76% di 1280** | keunggulan fine-tune = ketahanan terhadap skala input, bukan lonjakan akurasi di skala deploy |
| C-8 | pelabelan | box dari model lalu dikoreksi manual: 180/180 tepat 1 box, 124 sama, 52 digeser kecil | box ketat leher/cincin |
| C-9 | **uji independen** `cap_belt_final` di sesi 16 & 20 Okt (217 foto, 21 botol, tidak pernah dilatih; 21 Okt, `rvm_kp/uji_profil.py`) | **71%** (154/217): bertutup 124/181, tanpa tutup 30/36, 36 foto tutup tidak terdeteksi. Tanpa Coca-Cola Zero & Powerade: ±94% (148/157) | kesalahan terkumpul di **tutup hitam** (Coke Zero b02/b05/b13: 37 salah; Powerade b01/b12: 17 salah), tutup hitam menempel di lajur plastik hitam belt; data latih 13 Okt tidak punya tutup hitam. **Angka 98,3% (C-6) hanya berlaku untuk jenis tutup yang ada di data latih.** Perlu foto botol bertutup hitam (bertutup + tanpa tutup) lalu latih ulang dan CV per botol |
| C-10 | **model tutup v2** (`cap_belt_v2`, 21 Okt): + 217 foto sesi 16/20 Okt (kotak pra-label model, dikoreksi user di Roboflow v2; 7 kotak diubah). CV 5 fold **per merek** (32 merek; botol isi ulang bisa botol fisik yang sama), resep sama C-6, dua model per fold: hanya 13 Okt vs gabung (`scripts/pipeline_cap/cv_cap_v2.py`) | full-frame 640: **88,2% → 99,0%** (393/397, CI 97,4–99,6); **tutup hitam 31,7% → 95,0%** (57/60); sesi 13 Okt tetap 99–100%. Coke Zero yang TIDAK dilatih dikenali setelah melihat tutup hitam Powerade (fold 0: 50/76 → 75/76) | `cap_belt_v2` dipakai `rvm_kp` |
| C-11 | setelan mode ROI (profil nano) untuk v2, diuji ulang dengan model fold yang sama (`cv_cap_v2_roi.py`) | kanvas 320 skala 1/3 conf 0,25: 380/397 (Powerade 10/20, 4 tak terdeteksi); **kanvas 416 skala 1/2 conf 0,03: 393/397 = full-frame** (Powerade 17/20, Coke Zero 40/40, 0 tak terdeteksi) | ROI 416/½/0,03 dipakai. Catatan: setelan dipilih dari 7 kandidat pada data CV yang sama (bias seleksi kecil); GFLOPs tutup 3,5 (vs 8,2 full-frame) |

Kesalahan sisa model final (3/180): tutup biru berpita merah → no_cap; leher putih berulir → cap (0,86); leher bening + cincin kuning → cap (0,54).

## 4. Percobaan deteksi cairan berwarna
**Target (rapat mitra 16 Okt):** hanya cairan berwarna; **air bening secara eksplisit di luar lingkup** (keterbatasan yang dicatat, bukan pekerjaan). Concern mitra (20 Okt): kamera menggantikan load cell, sehingga rancangan dan klaim tidak bergantung pada load cell dalam lingkup ini. Kebijakan 20 Okt: semua botol PET diterima, merek hanya bonus poin (whitelist Coca-Cola, Fanta, Sprite), tiga kategori ukuran (diubah user 20 Okt: kecil ≤350, tanggung 351–800, besar >800 ml; sebelumnya kecil <450).

### L-1 Pilot 15 Okt
Air putih vs kosong tampak 100%, tetapi area belt kosong saja memisahkan kedua kelompok 100% (beda sesi/pencahayaan) dan tutup berbeda (air = tanpa tutup, kosong = bertutup). **Tidak dipakai sebagai klaim.**

### L-2 Baseline fitur warna (M0)
Selisih Lab dan saturasi badan botol terhadap belt di sekitarnya, regresi logistik, uji tahan-botol: 87,6% (satu fitur), akurasi seimbang 0,78 pada fold yang sama dengan CNN. Kegagalan: Fanta kuning 0/8, Sprite hijau pucat 1/8. Kurva belajar fitur warna (k botol latih = 1–4): 0,69 / 0,82 / 0,89 / 0,91 (semua level).

### L-3 Detektor botol di belt
`ver7`: botol berisi di W16 hanya **20/40** terdeteksi (conf 0,35, kelas teratas = bottle); botol kosong C13 170/180; pilot 150/160. `ver8` (ver7 + data warisan terpilih + W16 + C13): **100%** di semua kelompok belt, regresi `old_test` 99,8% (ver7 100%, kemungkinan in-sample).

### L-4 Classifier crop (CV 5 fold per botol)
Fold k: uji = botol W16 ke-k + 6 botol C13; P15 diuji oleh kelima model. Hasil (ambang 0,5):

| varian | data latih tambahan | berisi→berwarna (W16, /40) | kosong ditolak benar (W16, /40) | C13 kosong diterima (/180) | pilot berisi (/32) |
|---|---|---|---|---|---|
| M0 fitur warna | – | 25 | 37 | 166 | 19 |
| V1 crop pad | W16 + C13 | 26 | 40 | 178 | 21 |
| **V2 crop lipat** | W16 + C13 | 33 | 40 | 178 | 23 |
| V3 | + Fanta web (540) | 38 | 40 | 176 | 18 |
| **V4** | W16 saja | **40** | 40 | 159 | **32** |
| V5 | + Fanta web + warisan | 32 | 40 | 174 | 20 |
| V6 | + warisan | 28 | 40 | 172 | 21 |

Catatan: "crop lipat" = botol dipotong dua sepanjang sumbu panjang dan ditumpuk, sehingga resolusi badan botol berlipat dua. Level `seperempat` tidak dilatih (tidak konsisten) dan hanya dilaporkan.

### L-5 Detektor 3 kelas vs dua tahap
Detektor 3 kelas (botol / botol_berwarna / non_bottle): berisi→berwarna 24/40 (Fanta 0/8, Sprite 3/8); regresi `old_test` 99,2% dan 1,1% non-botol salah diterima. Dua tahap `ver8` + classifier V4: 40/40 berisi, pilot 154/160. **Dipilih dua tahap.**

### L-6 Sapuan ambang (ver8 + classifier)
| konfigurasi | berisi W16 | berisi pilot | kosong + air diterima | C13 kosong diterima |
|---|---|---|---|---|
| V2 @0,5 | 78% | 68% | 100% | 100% |
| V2 @0,2 | 90% | 76% | 100% | 98% |
| V4 @0,7 | 100% | 89% | 100% | 93% |
| V4 @0,5 | 100% | 96% | 100% | 91% |
| V4 @0,3 | 100% | 100% | 100% | 85% |

### L-7 Data luar dan preprocessing (semua dengan uji per botol di belt)
| varian | perubahan | berisi (/40) | C13 (/180) | pilot (/32) | kesimpulan |
|---|---|---|---|---|---|
| G2 | V2 + kilau sintetis | 24 | 180 | 20 | tidak membantu |
| G4 | V4 + kilau | 40 | 156 | 32 | tidak berubah |
| S2 | V2 + RF100 soda-bottles | 30 | 180 | 20 | tidak membantu |
| S4 | V4 + RF100 | 40 | **104** | 32 | merusak |
| SG2 | V2 + RF100 + kilau | 29 | 180 | 21 | tidak membantu |
| **P1** | normalisasi warna belt | 31 | **180** | **26** | sedikit membantu; produk whitelist 24/24 |
| P2 | CLAHE | 32 | 180 | 17 | pilot lebih buruk |
| P3 (V2) | supresi latar | 28 | 179 | 21 | tidak membantu |
| P3 (V4) | supresi latar | 40 | 152 | 32 | merusak C13 |
| E2 | V2 + Open Food Facts (1577 crop) | 27 (kosong 37/40) | 172 | 21 | merusak (AUC berisi 1,000 → 0,912) |
| E4 | V4 + Open Food Facts | 32 | 162 | 22 | merusak (Sprite NM 7/8 → 0/8 pada E2) |

**Pola:** positif dari domain lain (foto kemasan, lemari pendingin) memberi model jalan pintas ("bukan belt" = berwarna) karena semua negatif dari belt. Tambahan harus membawa positif dan negatif dari domain yang sama dan diuji hanya di belt.

### L-8 Kurva belajar classifier CNN
35 latihan (15 epoch, resep V2): untuk tiap botol uji, k = 1–4 botol berwarna latih.

| k | recall botol uji | recall pilot |
|---|---|---|
| 1 | 0,25 | 0,14 |
| 2 | 0,30 | 0,19 |
| 3 | 0,71 | 0,52 |
| 4 | 0,70 | 0,60 |

Botol dengan "kerabat warna" di data latih cepat 100% (cola hitam ↔ cola kecil, hijau pucat); warna tanpa kerabat (Fanta kuning, Frestea) tidak pernah dikenali. Kebutuhan perkiraan ±3 botol per kelompok warna, ±15 botol untuk 5 kelompok. Batasan: 15 epoch (pesimis), 5–10 latihan per k.

**Koreksi analisis ulang (20 Okt) — lihat L-8b untuk hasil yang benar:** kurva ini **terkonfound oleh ketidakseimbangan kelas**. Penggandaan positif dibatasi 6×, sehingga rasio positif:negatif di data latih = 1:3,2 (k=1), 1:1,7 (k=2), 1:1,2 (k=3), 1,1:1 (k=4). Lonjakan di k=3 bertepatan dengan data latih menjadi seimbang, dan spesifisitas 1,00 di semua k menunjukkan model condong menolak. Selain itu b04 (Fanta kuning) 0/8 di k=4 dengan 15 epoch, padahal 7/8 dengan botol latih yang sama pada resep 30 epoch (V2). Jadi kesimpulan "±3 botol per kelompok warna" dan "warna tanpa kerabat tidak pernah dikenali" **belum terbukti**; perlu diulang dengan kelas seimbang, 30 epoch, dan metrik tanpa ambang (AUC).

### L-8b Kurva belajar DIULANG dengan benar (20 Okt, menggantikan kesimpulan L-8)
Perbaikan: kelas seimbang di semua k (positif digandakan ceil(neg/pos)), 30 epoch, probabilitas disimpan → AUC. Subset botol sama dengan L-8 (35 latihan). Skrip `s22c_curve_balanced.py`, ringkasan `s22d_curve_bal_report.py`.

| k botol berwarna latih | AUC botol uji [rentang] | recall @0,5 | spesifisitas (W16 / C13) | AUC sesi pilot | recall pilot @0,5 |
|---|---|---|---|---|---|
| 1 | 0,802 [0,18–1,00] | 0,14 | 1,00 / 1,00 | 0,741 | 0,08 |
| 2 | 0,915 [0,51–1,00] | 0,35 | 1,00 / 1,00 | 0,813 | 0,28 |
| 3 | 0,976 [0,79–1,00] | 0,62 | 1,00 / 1,00 | 0,919 | 0,47 |
| 4 | 0,998 [0,99–1,00] | 0,82 | 1,00 / 0,99 | 0,997 | 0,68 |

Per botol uji, AUC k=1/2/3/4: Sprite NM 0,76/0,97/1,00/0,99; cola hitam 0,91/1,00/1,00/1,00; Frestea 0,93/0,89/0,99/1,00; botol berlabel Fanta 0,40/0,75/0,89/1,00; cola kecil 1,00/0,96/1,00/1,00.

**Kesimpulan yang dapat dipegang:**
- Kemampuan memisahkan (AUC) naik **bertahap** tiap tambahan botol, tanpa lonjakan; di 4 botol sudah ±0,998 (sesi sama) dan ±0,997 (sesi pilot). Belum bisa diekstrapolasi di atas 4 botol (hanya 5 botol berwarna).
- Botol berlabel Fanta **dapat** dikenali (AUC 1,00, recall 0,88 di k=4): klaim lama "warna tanpa kerabat tidak pernah dikenali" **salah**.
- **Masalah utama di ambang, bukan di pemisahan:** spesifisitas 1,00 di semua k sedangkan recall @0,5 rendah. Contoh Frestea: AUC 1,00 tetapi recall 0,38. Model mengurutkan dengan benar tetapi skornya terlalu rendah, sehingga **ambang perlu dikalibrasi** pada botol yang tidak ikut latih sebelum dipakai.
- Angka "±15 botol" dari L-8 **tidak didukung**. Yang didukung: tiap botol berwarna tambahan (sampai 4) memperbaiki hasil secara bermakna, terutama lintas sesi dan kalibrasi.

### L-10 Cek jalan pintas sesi (20 Okt)
Classifier dilatih membedakan 13 Okt vs 16 Okt (CV per botol, kelas seimbang). **Potongan belt tanpa botol: AUC 1,000 (220/220 benar)**; potongan botol kosong: AUC 0,966 (sebagian karena beda merek antar set). Artinya pencahayaan/kondisi belt antarsesi berbeda nyata. Konsekuensi: perbandingan berisi vs kosong di dalam set 16 Okt sah (satu sesi), tetapi angka pada botol kosong 13 Okt (C13) mencampur efek label berwarna dan beda sesi. **Aturan data:** positif dan negatif harus dipotret dalam sesi yang sama.

### L-11 Pralatih data luar lalu fine-tune di belt (20 Okt)
Pralatih classifier di Open Food Facts (1577 crop, positif dan negatif dari sumber sama), lalu fine-tune per fold di belt (resep V2 dan V4). Hasil: V2 33→**25**/40 berisi (AUC 1,000→0,911); V4 40→**31**/40 (AUC 0,967), dan 3/35 air putih sesi pilot keliru disebut berwarna. Sprite Nipis Mint **0/8** pada keduanya: negatif Sprite/7up (botol hijau, cairan bening) di data luar mengajari "hijau = tidak berwarna". **Cara pralatih juga merusak**; kesimpulan "data luar tidak membantu" kini berlaku untuk dicampur maupun pralatih.

### L-12 Sesi 20 Okt: botol dan warna baru (data `datasets/1_foto_mentah/2026-10-20_water_v1/`, 117 foto, 16 botol)
**Data.** Dibeli 5 botol: Powerade (biru), Fanta anggur (ungu), Sunbay soybean (putih keruh), Minute Maid Pulpy (oranye), Coke Zero vanila (cola), dipotret lengkap (penuh, setengah, kosong tanpa tutup, kosong bertutup; soybean hanya penuh dan setengah). Ditambah 11 botol **isi ulang** (hanya penuh): cairan dari botol lain atau buatan sendiri, dituang ke botol bekas merek lain. Keterangan cairan (user): oranye = Nutrisari jeruk peras; hijau pucat = Nutrisari anggur hijau; putih (b09) = soybean + Nutrisari sirsak; hijau sabun = sabun Sunlight; b07 = Powerade; b08 = Fanta anggur; b06 dan b14 diasumsikan Coke Zero vanila dan Powerade. Kosong untuk botol isi ulang ada di sesi 13 Okt atau hari yang sama.

**Uji model lama tanpa latih ulang (warna baru, paling jujur).** V4 (dilatih set 16 Okt): AUC 0,92; berisi 69/84; kosong diterima 24/33. Dikenali: ungu, cola, putih, oranye, hijau pucat Nutrisari. **Gagal: biru** (Powerade 1/8, Vica isi biru 0/4) **dan sangat pucat** (air sabun 1/4). **Minute Maid kosong dibaca berwarna 0/9** (label oranye menutupi botol). V2 (+ botol kosong 13 Okt): kosong 33/33, tetapi melewatkan putih, oranye, biru.

**Latih ulang 13 botol (resep RA, per botol).** AUC 0,987; berisi 92/92; kosong 60/73 (@0,5) dan 64/73 (ambang terkalibrasi ±0,94); Minute Maid kosong tetap 0/9; sesi pilot AUC 0,975 tetapi ambang hari ini terlalu tinggi untuk sesi lain (berisi 19/32). **Hasil ini terlalu optimis:** botol isi ulang berbagi cairan dengan botol aslinya, sehingga cairan uji sudah terlihat saat latih. Diulang per kelompok cairan (L-13).

**Ukuran.** Botol 390 ml jatuh tepat di batas luas kotak (Coke Zero vanila 240k vs 242k; Fanta anggur 244k vs 239k), dan luas kotak sedikit bergeser antarsesi. Batas kecil/tanggung perlu digeser (±255k) dan **ukuran perlu dikalibrasi per pemasangan kamera** dengan botol yang ml-nya diketahui. ml botol sesi ini belum dicatat.

### L-13 Validasi per kelompok cairan (berjalan 20 Okt malam)
21 botol berwarna, 13 kelompok cairan (botol dengan cairan sama diuji bersamaan); resep RA (dua sesi, tanpa C13) dan RB (+ C13). Skrip `s28_cv_liquidgroup.py`, ringkasan `s28b_report.py` (`out/cv_lg_RA_report.txt`, `cv_lg_RB_report.txt`).

| ukuran | RA (W16 + 20 Okt) | RB (+ kosong 13 Okt) |
|---|---|---|
| AUC gabungan (out-of-fold, jujur) | **0,971** | **0,973** |
| berisi → berwarna @0,5 | **124/124** | 104/124 |
| kosong diterima @0,5 | 58/73 | **69/73** |
| botol kosong merek lain 13 Okt diterima @0,5 | 86% | **95%** |
| sesi pilot (AUC; berisi @ambang median) | 0,987; 27/32 | 0,979; 27/32 |
| gagal utama | kosong berlabel berwarna penuh (Minute Maid 0/9, Powerade 2/8) | putih (soy 1/12) dan sangat pucat (Sunlight 0/4) |

**AUC di dalam botol yang sama (berisi vs kosong botol itu sendiri):** 7 dari 9 kelompok yang punya pasangan **1,000 di kedua resep**, termasuk warna yang tidak pernah dilatih (cola vanila, Fanta anggur ungu). Gagal: **Minute Maid** (RA 0,78; RB 0,83) dan **Powerade** (RA 0,97; RB 0,83): botol dengan **label/kemasan berwarna penuh** membuat botol kosong tampak berwarna.

**Kalibrasi:** ambang terkalibrasi tidak stabil antarfold (RA ±0,94, RB 0,01–0,61), jadi satu ambang tetap belum bisa dipegang; ambang harus dikalibrasi per pemasangan dan/atau probabilitas dikalibrasi.

**Kesimpulan:** dalam lingkup cairan berwarna, kamera memisahkan berisi vs kosong dengan baik pada botol yang tidak ikut latih (AUC ±0,97, 7/9 botol sempurna). Dua kelemahan nyata: (1) botol berlabel penuh berwarna saat kosong; (2) cairan putih dan sangat pucat (tergantung resep). Data paling bernilai berikutnya: **botol kosong berlabel penuh berwarna, dipotret di sesi yang sama dengan botol berisi** (bisa botol bekas, gratis).

### L-14 Resep tambahan: RC (kosong berlabel penuh 13 Okt) dan RH (geser hue piksel botol) — 20/21 Okt
- **RC** = RA + hanya 12 botol kosong 13 Okt berlabel penuh/berwarna. AUC 0,971 (= RA). Minute Maid membaik (dalam botol 0,78→0,94) tetapi Powerade memburuk (0,97→0,71) dan soy gagal (1/12). **Tidak lebih baik dari RA.**
- **RH** = RA + 3 salinan per foto dengan **hue piksel botol diputar** (cairan, label, tutup; belt tidak diubah), diterapkan **simetris** ke berisi dan kosong (`s29_hueaug.py`). Ide: dari warna yang sudah ada, model belajar warna lain (usul user).

| | RA | RB | RC | **RH** | ansambel RA+RH |
|---|---|---|---|---|---|
| AUC gabungan (per kelompok cairan) | 0,971 | 0,973 | 0,971 | **0,992** | 0,979 |
| berisi → berwarna @0,5 | 124/124 | 104/124 | 104/124 | **124/124** | 124/124 |
| kosong diterima @0,5 | 58/73 | **69/73** | 67/73 | 60/73 | 59/73 |
| AUC dalam botol Powerade (biru) | 0,969 | 0,828 | 0,711 | **1,000** | 1,000 |
| AUC dalam botol Minute Maid | 0,778 | 0,833 | **0,944** | 0,889 | 0,868 |
| soy / Sunlight @0,5 | 12/12, 4/4 | 1/12, 0/4 | 1/12, 0/4 | **12/12, 4/4** | 12/12, 4/4 |
| sesi pilot: AUC; berisi @0,5; kosong+air @0,5 | **0,987**; 32/32; 38/44 | 0,979 | 0,977 | 0,936; 31/32; 39/44 | 0,972 |

**Bacaan:** geser hue pada botol saja **berhasil**: biru (kelompok Powerade disisihkan seluruhnya) dipisahkan sempurna, putih dan sangat pucat tetap dikenali, AUC naik 0,971 → 0,992. Kelemahan tersisa: (1) **Minute Maid kosong** (label oranye membungkus penuh) masih terbaca berwarna; (2) skor jenuh mendekati 1 sehingga ambang terkalibrasi ≈ 1,00 dan tidak informatif: pakai 0,5 lalu kalibrasi per pemasangan; (3) urutan skor lintas sesi (pilot) sedikit lebih buruk dari RA (AUC 0,936 vs 0,987), walau pada ambang 0,5 hasilnya setara (31/32 dan 39/44). Ansambel RA+RH adalah kompromi (0,979; pilot 0,972).

### L-15 Sistem multi-backend dan benchmark awal di laptop (21 Okt)
Kode `rvm_kp/` (botol → tutup → cairan → ukuran) dengan backend dipilih dari file model; model diekspor ke ONNX dan OpenVINO. Benchmark 40 frame 1920×1080 (sesi 20 Okt), laptop Intel Core 5 210H + RTX 4050, terhubung listrik:

| backend | total per frame | ≈ FPS | botol / tutup / cairan (ms) |
|---|---|---|---|
| PyTorch GPU (CUDA) | 210 ms (p95 283) | 4,7 | 85 / 82 / 41 |
| **OpenVINO CPU** (ulang, setelah onnxruntime-gpu dicopot) | **87 ms (p95 124)** | **11,5** | 35 / 36 / 15 |
| OpenVINO CPU (run pertama) | 321 ms (p95 516) | 3,1 | 154 / 147 / 20 |
| PyTorch CPU | 450 ms (p95 558) | 2,2 | 202 / 202 / 45 |
| ONNX Runtime CPU via Ultralytics | 4.227 ms | 0,2 | 1.929 / 1.925 / 365 |

**Temuan ONNX:** satu model ONNX sendiri hanya 47 ms, tetapi tiga sesi ONNX Runtime bergantian menjadi ±3,7 s per frame karena **thread pool tiap sesi berputar (spinning) dan saling berebut CPU**. Dengan `session.intra_op.allow_spinning = 0` turun ke ±0,4 s (diuji langsung dengan onnxruntime). Ultralytics tidak membuka opsi ini, jadi untuk CPU dipakai OpenVINO. Paket `onnxruntime-gpu` yang dipasang otomatis oleh Ultralytics saat ekspor dicopot (bentrok dengan `onnxruntime`), dan `onnxruntime` 1.29 dipasang ulang.

**Variasi antar-run besar** (OpenVINO 321 vs 87 ms): ulangi benchmark beberapa kali, laptop terhubung listrik dan tanpa beban lain, sebelum angka dipakai di laporan.

Keputusan cairan OpenVINO = ONNX pada 40/40 frame. **Catatan jujur:** angka ini termasuk pra-proses frame 1080p dan dua detektor penuh-frame per frame; ini angka awal tanpa optimasi (deteksi tutup hanya di area botol, FP16, TensorRT), dan **hanya berlaku untuk laptop ini**, bukan Jetson. Dengan pemrosesan dipicu per botol (bukan tiap frame), ±200–300 ms per botol masih memadai untuk konveyor lambat.

### L-16 Profil Jetson Nano ditiru di laptop (21 Okt)
**Perangkat target dipastikan:** Jetson Nano Developer Kit B01 4 GB (P3450) → JetPack 4.6.x, Python 3.6, CUDA 10.2, TensorRT 8.2, GPU Maxwell 128 core. Ultralytics modern hanya lewat Docker `ultralytics/ultralytics:latest-jetson-jetpack4`. Panduan: `rvm_kp/JETSON-NANO.md`.

**Profil `nano`** (`rvm_kp/config.py`): kamera 720p, deteksi tutup **hanya di area tiap botol** (kanvas 320 px, skala potongan = skala full-frame 640, margin 8%), FP16, 4 thread CPU. Profil `laptop` = cara lama (1080p, tutup full-frame 640, FP32). Di laptop profil ini meniru **batasan** Nano, bukan **kecepatannya**.

**Uji di 217 foto belt (sesi 16 & 20 Okt, 21 botol), PyTorch GPU, terhadap acuan profil laptop** (`datasets/4_eksperimen/benchmark/uji_profil_2026-10-21.txt` + `.csv` per foto):

| konfigurasi | tutup benar | cairan sama acuan | ukuran sama acuan | GFLOPs / frame (1 botol) |
|---|---|---|---|---|
| acuan 1080p, tutup full-frame, FP32 | 154/217 (71%) | – | – | 16,8 |
| 1080p, tutup ROI 320 | 157/217 (72%) | 217/217 | 217/217 | 10,7 |
| 720p, tutup full-frame | 159/217 (73%) | 217/217 | 216/217 | 16,8 |
| 540p, tutup ROI 320 | 155/217 (71%) | 217/217 | 213/217 | 10,7 |
| **profil nano: 720p, ROI 320, FP16** | 154/217 (71%) | **217/217** | **216/217** | **10,7** |
| 1080p full-frame FP16 | 154/217 (71%) | 217/217 | 217/217 | 16,8 |

Kesimpulan: resolusi 720p, deteksi tutup di area botol, dan FP16 **tidak mengubah keputusan** (selisih tutup ±1–5 foto, dalam derau). Komputasi per frame turun **36%** (16,8 → 10,7 GFLOPs). Kanvas ROI 416 tidak lebih baik dan menambah "tidak terdeteksi". Akurasi tutup 71% berasal dari model (C-9), bukan dari profil. Model cairan RH dilatih dengan sesi ini, jadi untuk cairan yang dilaporkan hanya kesepakatan, bukan akurasi.

**Kecepatan di laptop** (40 frame, `benchmark_2026-10-21_laptop_*.txt`): PyTorch GPU profil laptop 218 ms → profil nano 149 ms; OpenVINO CPU profil nano 107 ms; keputusan OpenVINO = PyTorch 40/40 (cairan dan tutup). Di GPU laptop waktu didominasi overhead per panggilan Ultralytics, bukan komputasi, sehingga penurunan GFLOPs lebih berarti di Nano (terbatas komputasi). **Kecepatan Nano belum diukur.**

**Ekspor untuk TensorRT 8.2:** ONNX opset 12, IR 7, input statis (botol 640, tutup full 640, tutup ROI 320, cairan 224). Hanya 14 jenis operasi standar (Conv, Sigmoid, Mul, Split, Add, Concat, MaxPool, Resize, Reshape, Transpose, Softmax, Slice, Sub, Div; classifier + GlobalAveragePool, Flatten, Gemm). Keluaran ONNX = `.pt` (selisih relatif maks 2,4×10⁻⁶). Kode `rvm_kp/` lolos cek sintaks Python 3.8. Engine `.engine` tetap harus dibuat di Nano.

### L-17 Uji LIVE di belt berjalan dengan aplikasi kiosk + PLC (22 Okt sore)
Aplikasi `app/` (`python -m app --backend pytorch --cam 1 --plc`), profil nano (720p), model aktif: `botol_ver9c`, `cap_belt_v2`, `cairan_RH`. Area belt dinyatakan aman oleh user.
- **Belt jalan terus (cara awal):** suara tutup antar-frame bercampur (mis. 10 tanpa tutup / 5 bertutup / 5 tak terbaca dalam satu botol); user: cairan "tidak terlihat bekerja" — sebagian karena layar hanya menampilkan alasan pertama (tutup). Diperbaiki: alasan gabungan "bertutup dan masih ada isinya".
- **Berhenti-lalu-pindai (usul user):** pusat botol masuk zona 40–60% lebar → PLC MB10.0 OFF → jeda → 3 frame DIAM → voting → ON. Berhenti total 0,9–1,3 s (jeda 0,5 s) dan 1,5–1,8 s (jeda 1,0 s); PLC selalu merespons. Frame diam disimpan (`datasets/1_foto_mentah/2026-10-22_scan_app/`, `SCAN.csv` + label user). Jeda 1,0 s: frame tajam (tidak ada blur).
- **Hasil 30 botol berlabel user** (jeda 1,0 s): tutup **30/30** benar (termasuk tutup hitam Coca-Cola), selalu bulat 3/3 frame. Cairan: kosong **16/16** benar (0 salah alarm); berisi **10/14** — luput 3× Fanta berisi minuman oren yang di kamera tampak kuning PUCAT (p 0,00–0,29) dan 1× Coca-Cola berisi air bening (p 0,49). **Di luar dugaan: air bening di botol air mineral (Le Minerale, Cheers) terdeteksi 5/5 (p 1,00)** — tetapi semua bertutup, n kecil, dan air bening tetap di luar lingkup resmi; jangan diklaim tanpa uji tanpa tutup & botol lain.
- Ukuran: botol 400 ml ramping (#17) terbaca "kecil" (luas kotak kecil) → batas berbasis luas lemah untuk botol ramping; pertimbangkan panjang botol.
- Kesimpulan: stop-and-scan dijadikan mode default aplikasi (`app/config.py` STOP_SCAN, `--terus` untuk mematikan, `--jeda` detik). Kelemahan tersisa = warna pucat & botol berlabel penuh (latih ulang dengan sesi 22 Okt + frame scan).

### L-18 Latih ulang malam 22→23 Okt: cairan RN dan detektor ver10
**Cairan (`scripts/pipeline_cairan/s31_data_22okt.py`, `s32_cv_rn.py`).** Data baru: V22 (sesi capture 22 Okt, 5 botol berlabel penuh; air bening berisi dibuang, Mizone kosong dipakai) + S22 (90 frame diam aplikasi, label user) = 128 potongan, 18 kelompok botol fisik. CV 5 fold per kelompok (31 kelompok, 325 foto), RH (data lama) vs RN (+ data 22 Okt), resep sama:

| salah (ambang 0,5) | RH | RN | rata-rata RH+RN |
|---|---|---|---|
| total 325 foto | 45 | 42 | **40** |
| data lama 16/20 Okt | 15 | 10 | 10 |
| V22 berlabel penuh (AUC) | 0,59 | **0,94** | 0,80 |
| S22 frame belt | 0 | 9 | 3 |

Temuan: (1) **ketidakstabilan pada warna pucat** — Fanta isi oren pucat: model RH final (aktif) luput di uji live, model RH dari fold CV mendeteksi 9/9, RN luput 9/9 → kasusnya di batas keputusan, bergantung variasi latih; (2) gabungan RH+RN paling stabil (40 salah); (3) **botol KOSONG berlabel penuh** (Minute Maid, Mizone) tetap salah di semua resep (0/8); Isoplus kosong membaik 1/8 → 7/8 dengan RN. Keputusan: model aktif tetap RH; `cairan_RN.pt` + opsi gabungan (`rvm_kp/config.py` LIQUID_EXTRA, mati) diuji di sesi pembuktian.

**Detektor botol ver10 (`scripts/pipeline_botol/ver10_nyata.py`).** Resep ver9c + foto NYATA 22 Okt: 79 non-botol (plastik kresek 37, tangan 9, foto botol di HP 8, tisu 6, HP 5, dll.) sebagai gambar tanpa botol (×5) + 154 botol hari yang sama (×2; kotak dari ver9c). CV 5 fold per BENDA (kresek dibagi 6 kantong):

| | ver9c | **ver10** |
|---|---|---|
| non-botol nyata diterima (79) | 29 | **0** |
| plastik kresek (37) | 26 | **0** |
| botol terdeteksi (154) | 154 | **154** |

Regresi model final: TACO tempel 8,8% → 7,9%; TrashNet non-botol 7% → 5% (kaca 16 → 10/108), PET TrashNet 33 → 30/44; botol tempelan studio/belt 94%/97%; sesi 20 Okt 117/117; old_test sama; kotak vs ver9c IoU median 0,976, keputusan tutup & cairan identik 167/167. Keterbatasan: negatif & positif uji dari SATU sesi (22 Okt) → perlu sesi pembuktian di hari lain dengan benda baru (`docs/DAFTAR-POTRET-SESI-PEMBUKTIAN.md`). Kandidat: `3_model_final/botol_ver10_kandidat.pt` (+ ONNX/OpenVINO `ekspor/botol_ver10*`).

### L-19 Kenapa data luar tidak membantu (analisis 23 Okt, usul user)
- Model cairan memakai potongan kotak botol (bukan foto utuh), tetapi **botol PET tembus pandang**: latar (belt / meja putih) terlihat MELALUI botol → sumber foto tetap terbaca dari dalam kotak. Menghapus latar otomatis (rembg) **menghapus badan botol bening** (hanya tutup/label tersisa) → tidak dapat dipakai.
- Saringan kualitas Open Food Facts (botol utuh, tajam, berbentuk botol): hanya 92/1577 lolos (1271 menempel tepi foto: foto produk dibingkai mepet).
- **Label 'kosong' Open Food Facts = botol PENUH air bening**, bukan kosong; foto produk hampir tidak pernah botol kosong. Pelabelan ulang visual (Claude, menunggu validasi user): 34 berwarna, 40 air bening penuh, 18 buang, **0 kosong**. Data luar praktis hanya membawa satu kelas → jalan pintas sumber. Percobaan setelah validasi user (`s33_luar_ketat.py`, potongan ketat, tanpa geser-hue, CV 5 fold per kelompok sama dgn s32; data luar hanya di latih):

| resep | AUC | salah /325 | V22 berlabel penuh: AUC, kosong benar | S22 belt: berisi |
|---|---|---|---|---|
| T0 data kita | 0,937 | 50 | 0,80 · 6/32 | 22/24 |
| T1 + luar 14 berwarna + 4 kosong | 0,948 | 50 | 0,81 · 7/32 | 19/24 |
| T2 = T1 + 15 air bening sebagai 'tidak berwarna' | **0,969** | **43** | **0,91 · 11/32** | **24/24** |

Data luar TERVALIDASI tidak merusak (T1 = T0), dan menambah botol berisi air bening sebagai 'tidak berwarna' (T2) memperbaiki justru kelemahan utama (botol berlabel penuh; dugaan: model belajar memisahkan "ada isi" dari "isinya berwarna"). Efek kecil-sedang (7 foto dari 325), satu kali CV. Catatan: T2 mengajari model bahwa air bening = tidak berwarna (sesuai lingkup resmi), padahal di uji live model sempat menolak botol air bening — konsekuensi kebijakan perlu disepakati. Pembanding resep penuh (dengan geser-hue): RH 45, gabungan RH+RN 40 salah.
- Pelajaran: data luar membantu hanya bila membawa positif DAN negatif dari sumber yang sama (TrashNet untuk detektor botol), atau berupa variasi data kita sendiri (geser warna RH).

### L-20 Ukuran dari panjang botol (23 Okt, `scripts/alat/analisis_ukuran.py`)
26 botol ber-ml (13/16/20 Okt), uji tinggalkan-satu-botol, setelah koreksi user (Sprite Nipis Mint 16 Okt = 390 ml, bukan 250): luas kotak 24/26 (salah: Pristine 400, Isoplus 350), **panjang 25/26** (salah: Cleo galon 1000 yang pendek-gemuk; ditutup aturan luas ≥ 360k = besar), lebar 20/26. Aturan panjang + "luas ≥ 360k = besar" (menutup galon gemuk) tersedia di `rvm_kp` (`SIZE_RULE = "panjang"`, bawaan masih "luas"). Kalibrasi tetap per pemasangan.

### L-21 Definisi baru: berisi APA PUN (air bening juga ditolak) — keputusan user 23 Okt
Dipicu uji live L-17 (air bening di botol air mineral terdeteksi 5/5). Lingkup 16 Okt (hanya berwarna) diperluas → perlu dikabarkan ke mitra; di TA sebagai temuan. `scripts/pipeline_cairan/s34_berisi_bening.py` (CV 5 fold per kelompok, 351 foto, 33 kelompok, potongan ketat tanpa geser-hue; 26 foto air bening belt dari 5 botol: Mizone, Le Minerale ×2, Cheers ×2, Coca-Cola isi air):

| | B0 data kita (+ air bening belt = berisi) | B1 + data luar tervalidasi |
|---|---|---|
| air bening belt terdeteksi (botol ditahan) | **26/26** | 24/26 |
| cairan berwarna lama | 122/124 | 120/124 |
| kosong benar | 115/153 | 114/153 |
| total salah | **42/351** | 47/351 |

Menambah air bening sebagai 'berisi' tidak menurunkan deteksi warna dan tidak menambah salah alarm kosong; data luar tidak membantu pada definisi ini. Keterbatasan: air bening hanya 5 botol, satu sesi, mayoritas bertutup → uji wajib di sesi pembuktian dengan botol air bening TANPA tutup merek lain. Kandidat: `3_model_final/cairan_berisi_kandidat.pt` (`s35_final_berisi.py`, resep RN + air bening, geser-hue). Botol kosong berlabel penuh tetap belum terpecahkan (5/32).

### L-9 Model final
`datasets/4_eksperimen/cairan/final/` (belt W16 + C13) dan `final2/` (+ pilot dan 28 foto tambahan). Keduanya tidak punya uji bersih; akurasi = hasil CV di atas. Belum dipasang di server (lingkup Capstone 1).

## 4b. Percobaan detektor botol vs non-botol dengan negatif TACO-tempel (21 Okt)
Kode: `scripts/pipeline_botol/` (`taco_paste.py`, `ver9_nonbottle.py`, `cek_jalan_pintas.py`). Hasil: `datasets/4_eksperimen/botol/`.
Data negatif lama (data tim, `old_*`) ternyata sudah foto NYATA di belt rig lama (sabun, kaleng, karton, kemasan); yang belum ada: plastik film, kantong, gelas, plastik diremas. Ditemukan juga 1 contoh botol Aqua berlabel `non_bottle` di data lama (kemungkinan salah label, belum diaudit).

**Metode.** Objek TACO (CC BY 4.0) dipotong dengan poligon segmentasi asli dan ditempel ke foto belt (rotasi, skala 15–40% lebar, tepi dihaluskan, kecerahan disesuaikan). Split per gambar TACO (id % 5 == 0 = uji). Latar latih = foto belt yang sudah ada di data ver8b; latar uji = sesi 20 Okt (tak pernah dilatih detektor). Resep latih = resep ver8b (dari ver7, 10 epoch, data ver8b + 672 foto sintetis).

| ukuran (ambang sistem: bottle conf ≥ 0,35) | ver7 | ver8b (dipakai) | ver9 | **ver9b** |
|---|---|---|---|---|
| A. sampah TACO-tempel diterima sebagai botol (328 objek uji) | 11,0% | 19,5% (kaleng 10/23, kantong 7/16) | 0,0% | **3,0%** |
| B/C. botol asli sesi 20 Okt terdeteksi (117) | 98/117 | 117/117 | 117/117 | **117/117** |
| D. old_test 284 foto nyata: recall bottle / non_bottle | 0,999 / 1,000 | 0,990 / 0,974 | 1,000 / 0,977 | **1,000 / 0,986** |
| S1. botol studio (tak dilatih) ditempel dengan cara sama: diterima | 74,8% | 41,1% | **1,9%** | 76,6% |
| S2. botol belt 20 Okt ditempel (mask elips): diterima | 60,8% | 48,0% | **5,9%** | 97,1% |
| E. plastik bening diremas, foto nyata alas abu (3 foto) | 3/3 diterima | 1/3 | 3/3 | 3/3 |

**Temuan 1 – jalan pintas tempelan.** ver9 (hanya negatif yang ditempel) menolak 100% sampah tempelan, tetapi juga menolak 94–98% **botol asli** yang ditempel dengan cara sama: ia belajar "bekas tempelan = non_bottle", bukan bentuk sampah. **ver9 ditolak.** Uji kontrol S1/S2 ini wajib untuk setiap data sintetis tempel.
**Temuan 2 – mitigasi.** ver9b menambahkan botol asli ber-alpha (372 potongan: botol belt latih bermask elips + botol studio bernomor genap) dengan cara tempel yang sama, jumlah seimbang dengan negatif. Hasil: sampah tempelan diterima turun 19,5% → 3,0%, botol tempelan tetap diterima (S1 77% vs ver8b 41%), tidak ada regresi pada botol asli maupun old_test (non_bottle recall naik 0,974 → 0,986).
**Ancaman validitas.** (a) Semua perbaikan diukur pada data **sintetis**; sisa jalan pintas lain (tekstur foto Flickr TACO vs belt, bentuk mask poligon vs elips) belum bisa disingkirkan; S2 memakai mask elips yang sama dengan latih sehingga mungkin optimistis, S1 (mask rembg, botol tak dilatih) lebih jujur. (b) Satu-satunya negatif sulit NYATA (3 foto plastik diremas, alas abu, bukan belt) tetap diterima ver9b (ver8b 1/3); n terlalu kecil untuk disimpulkan, tetapi bukan bukti perbaikan.
**Keputusan.** `botol_ver8b` tetap model utama. `datasets/3_model_final/botol_ver9b_kandidat.pt` = kandidat, dipakai hanya bila lolos uji foto NYATA di belt: plastik diremas, kantong, gelas, kaleng, serta uji iseng (foto botol di layar HP, gambar botol dicetak/ditempel busa) — status item 5/5b.

**Lanjutan 22 Okt – ver9c (+ TrashNet).** TrashNet (MIT, Thung & Yang 2016; 2.527 foto objek tunggal di alas putih) dipotong dengan rembg (u2net lokal), hanya objek UTUH (794). Kelas plastic dipilah visual oleh Claude (belum diverifikasi manusia): 86 botol PET = `bottle`, 102 plastik lain = `non_bottle`, 49 ragu dibuang (jerigen HDPE, botol isi ulang, tampak ujung, toples); kaca, logam, kertas, kardus, sampah = `non_bottle`. Tujuan: positif DAN negatif dari sumber luar yang SAMA (pelajaran L-7/peta dataset), agar "sumber asing = non_bottle" tidak bisa dipakai sebagai jalan pintas. Split per nomor foto (nomor % 5 == 0 = uji). ver9c = resep ver9b + 50% negatif dari TrashNet + 30% tempelan botol dari PET TrashNet. Skrip: `scripts/pipeline_botol/trashnet_prep.py`, `uji_trashnet.py`.

| ukuran | ver8b (dipakai) | ver9b | **ver9c** |
|---|---|---|---|
| A. sampah TACO-tempel diterima (328) | 19,5% | 3,0% | 8,8% |
| F. non-botol TrashNet ditahan diterima (382 tempelan) | 21% (kaca 21/108, kaleng 30/106) | 3% | **7%** (kaca 16/108, kaleng 2/106) |
| F. botol PET TrashNet ditahan diterima (44) — kontrol | 64% | **7%** ✗ | **75%** |
| S1/S2. botol studio/belt ditempel diterima — kontrol | 41% / 48% | 77% / 97% | **93,5% / 100%** |
| C. botol asli 20 Okt | 117/117 | 117/117 | 117/117 |
| D. old_test recall bottle / non_bottle | 0,990 / 0,974 | 1,000 / 0,986 | 1,000 / 0,985 |
| E. plastik diremas nyata (3) diterima | 1/3 | 3/3 | 3/3 |

**Temuan 3 – jalan pintas "sumber asing".** ver9b tampak terbaik pada negatif luar (3%), tetapi juga menolak 93% botol PET dari sumber luar yang sama: ia belajar "objek bergaya foto luar = non_bottle". ver9c, yang melihat botol PET dari sumber yang sama, menerima 75% PET TrashNet sambil hanya menerima 7% non-botol TrashNet: pemisahan berdasarkan jenis objek, bukan sumber. **ver9c = kandidat terbaik** (`3_model_final/botol_ver9c_kandidat.pt`), menggantikan ver9b sebagai kandidat. Negatif tersulit: botol kaca (15% masih diterima). Tetap sintetis; plastik diremas nyata (E) belum membaik; keputusan ganti ver8b menunggu foto negatif NYATA di belt.

**Uji NYATA di belt (22 Okt, sesi `2026-10-22_nonbottle_v1`, 49 foto, belt mati) — penentu.** Kontrol positif = 64 foto botol PET sesi cairan hari yang sama (`2026-10-22_water_v1`). Diterima = ada kotak `bottle` conf ≥ 0,35 (`datasets/4_eksperimen/botol/uji_nyata_2026-10-22.txt`).

| | ver7 | ver8b | ver9b | **ver9c** |
|---|---|---|---|---|
| non-botol nyata diterima (32 foto: kotak kacamata, gunting, HP, kardus, mouse, plastik bening diremas, bungkus jajan, tangan) | 3 | 13 | 8 | **7** |
| kontrol botol PET terdeteksi (64) | 47 | 64 | 64 | **64** |
| uji iseng: foto botol di layar HP (8) | 0 | 0 | 0 | 0 |
| plastik bening diremas (4 foto, 1 benda) | 2 | 4 | 4 | 4 |
| botol bukan minuman: hand sanitizer (3) / Listerine (4) | 0 / 0 | 3 / 4 | 2 / 3 | 3 / 4 |

ver9c menurunkan salah terima non-botol nyata dari 13/32 (41%) ke 7/32 (22%) tanpa kehilangan botol (64/64), arah sama dengan uji sintetis. ver9b gagal di mouse (3/3) dan tangan tidak, ver8b menerima tangan 2/9 dan HP 2/5. Dengan ver9c, kotak botol nyaris sama dengan ver8b (IoU median 0,966 pada 103 foto botol), keputusan tutup 103/103 sama, cairan 69 vs 68 benar dari 87. **Keputusan 22 Okt: `botol_ver9c` menjadi detektor aktif `rvm_kp`** (ver8b disimpan). Keterbatasan: n kecil (32 foto dari ±8 benda, satu sesi); plastik bening diremas masih selalu lolos; foto HP bergambar botol ditolak semua model (bagus, tetapi baru 8 foto dari 1 HP); botol non-minuman (hand sanitizer, Listerine) diterima — perlu keputusan mitra apakah botol PET non-minuman diterima.

## 5. Percobaan model merek dan data lain
- **Model merek lama** `brand_best_ver5` (aqua, coca_cola, le_mineral, unknown): `coca_cola` = produk perusahaan Coca-Cola (mitra), jadi Fanta/Sprite yang terbaca `coca_cola` benar. Pada 180 foto belt, produk non-Coca-Cola yang diterima sebagai `coca_cola`: BIG Cola 6/6, BIG Strawberry 6/6, Vit 6/6, Superindo merah 5/6, Aquviva 5/6. Sprite Nipis Mint → `unknown` 20/20 (luput).
- **Perkiraan label salah di `brand2`:** dari contoh 48 kotak `unknown`, 8 adalah produk Coca-Cola Company (≈12%, ±140 kotak) — perkiraan, belum diverifikasi penuh.
  **Terverifikasi 23 Okt (audit penuh user):** 2216 kotak unik (1675 dari 3265 gambar adalah duplikat persis, dibuang dengan md5). `unknown` → 134/828 ternyata produk Coca-Cola Company (16%; Fanta, Sprite, Minute Maid, Ades, dll.); kelas `coca_cola` lama hampir bersih (2 salah, 2 buang). Usulan model dasar (belt saja) untuk data lama: 177 ditebak Coca-Cola, 106 di antaranya ditolak user → model belt saja tidak bisa dipakai sebagai pelabel otomatis data internet.
- **Model merek baru (23 Okt, keputusan user: hanya 2 kelas `coca_cola` vs `non_coca_cola`; merek lain = pekerjaan lain).** Potongan botol utuh (tidak dilipat), YOLOv8n-cls 256. Data: foto belt 551 (37 produk, 8 produk Coca-Cola Company, label produk divalidasi user). Model dasar, CV 5 fold per botol: Coca-Cola terdeteksi 222/261, non benar 279/290; per produk tak dikenal: 158/261 (Coca-Cola/Zero tetap 100%, Frestea 5/29, Powerade 4/20, Sprite 13/36 → model menghafal produk, bukan "ciri Coca-Cola Company"). Data tambahan teraudit: brand2 (500 coca / 1702 non) dan Open Food Facts (346 coca / 406 non pesaing mirip: Pepsi, 7up, Dr Pepper, Mirinda, Nestea, Lipton; 21 buang). Perbandingan (uji tetap hanya foto belt): `datasets/4_eksperimen/merek/cv_campur_report.txt`.
  **Hasil D (belt + data lama + OFF, 12 epoch):** TIDAK lebih baik secara keseluruhan. Produk dikenal: coca 220/261 (dasar 222), non benar 264/290 (dasar 279; salah terima naik 11→26, mis. Sunbay 7, Mizone 3, Vit 2). Produk tak dikenal: coca 154/261 (dasar 158). Hanya Sprite yang jelas terbantu (tak dikenal 13→32/36; banyak foto Sprite di internet); Powerade, Minute Maid, Fanta tak dikenal justru turun. Dugaan: data tambahan 6x lebih banyak dari foto belt sehingga model belajar "gaya foto internet". Uji berikut: E = D dengan foto belt diulang 6x.
  **Hasil E (belt diulang 6x):** produk dikenal coca 224/261, non benar 276/290 (salah terima 14) ≈ dasar (222/261, 279/290) — selisih dalam batas acak. Produk tak dikenal coca 154/261 (dasar 158): Sprite naik 13→31/36, tetapi Minute Maid 0/33, Frestea 3/29, Powerade 0/20 (dasar 11, 5, 4). **Kesimpulan: data tambahan teraudit tidak memperbaiki model secara berarti**; hanya produk yang banyak fotonya di internet (Sprite) terbantu. Kelemahan model ada pada produk dengan sedikit botol fisik di belt (Nutriboost 1 botol 0/16, Powerade 2 botol 8–11/20) → solusi: tambah botol fisik per produk mitra, bukan data internet. Karena produk mitra adalah daftar tetap, skenario F (produk dikenal) adalah ukuran yang relevan untuk mesin.
  **Hasil B (belt + brand2 saja) dan C (belt + OFF saja), skema F, 23 Okt malam** — untuk melihat sumber mana yang berpengaruh:

  | varian (skema F, ambang 0,5) | Coca-Cola terdeteksi | non benar | salah terima (produk terbanyak) | Nutriboost | Powerade | Sprite |
  |---|---|---|---|---|---|---|
  | A belt saja (551 foto) | 222/261 | 279/290 | 11 (Pristine 4, Aqua 2, Big Nipis Madu 2) | 0/16 | 11/20 | 27/36 |
  | B + brand2 (2202) | 218/261 | 276/290 | 14 (Pristine 4, Superindo 3, Sunbay 3) | 0/16 | 6/20 | 30/36 |
  | C + OFF (752) | 222/261 | 269/290 | 21 (Sunbay 6, Vit 3, Natsbee 3) | 3/16 | 8/20 | 30/36 |
  | D + brand2 + OFF | 220/261 | 264/290 | 26 (Sunbay 7, Aqua 4, Pristine/Mizone/Big Nipis Madu 3) | 4/16 | 4/20 | 32/36 |
  | E = D, belt ×6 | 224/261 | 276/290 | 14 (Sunbay 4, Superindo 3, Pristine 2) | 0/16 | 8/20 | 30/36 |

  Tafsiran: kedua sumber menaikkan Sprite (27→30–32/36) tetapi menurunkan Powerade (11→4–8/20). **OFF adalah penyebab utama naiknya salah terima** (Sunbay, Vit, Natsbee — botol berwarna yang mirip produk Coca-Cola di foto internet), brand2 hampir netral. Tidak ada varian yang lebih baik dari A secara keseluruhan (selisih coca ±4 = dalam batas acak satu kali latih). Keputusan: model final = A (belt saja); perbaikan lewat botol fisik tambahan (sesi pembuktian bagian G).
- **Metode validasi model merek (bahan subbab "Metode Validasi" TA).**
  1. *Metode:* validasi silang 5-fold berkelompok (*grouped k-fold cross-validation*, setara GroupKFold). Data uji = 551 foto belt mesin sendiri (261 Coca-Cola Company, 290 non; 37 produk, label produk divalidasi manusia). Setiap foto diuji tepat satu kali oleh model yang tidak melihat kelompoknya saat latih; hasil 5 fold dijumlahkan.
  2. *Alasan berkelompok, bukan acak per foto:* satu botol fisik difoto berkali-kali; pembagian acak per foto menaruh foto botol yang sama di latih dan uji (*data leakage*) sehingga akurasi tampak terlalu tinggi.
  3. *Dua skema kelompok:*
     - **F (ukuran utama)** — kelompok = satu botol fisik dalam satu sesi (foto katalog/scan: per produk). Menjawab: apakah botol BARU dari produk yang sudah dikenal dikenali? Relevan untuk mesin karena daftar produk bonus tetap. Hasil A: Coca-Cola 222/261, non benar 279/290 (salah terima 11).
     - **P (tambahan)** — kelompok = produk. Menjawab: apakah produk yang belum pernah dilihat dikenali? Hasil A: 158/261 → model menghafal tampilan produk, bukan "ciri Coca-Cola Company"; produk baru harus dilatih dulu.
  4. *Pembagian fold:* kelompok dibagi rata (serakah menurut jumlah foto), kelompok produk Coca-Cola dibagikan lebih dulu agar setiap fold memuat kedua kelas.
  5. *Pencegahan bocornya data uji ke pelatihan:* epoch tetap (30 dasar, 12 varian data tambahan), dipakai bobot epoch terakhir tanpa *early stopping* pada data uji; ambang tetap 0,5, tidak disetel; penyeimbangan kelas (duplikasi) dan data tambahan hanya di bagian latih; data uji selalu foto belt asli.
  6. *Metrik:* jumlah benar per kelas (Coca-Cola terdeteksi, non ditolak benar), **jumlah salah terima** (non lolos sebagai Coca-Cola = bonus salah, paling penting bagi mitra), dan rincian per produk.
  7. *Perbandingan varian A–E:* fold dan data uji identik untuk semua varian (perbandingan berpasangan); data tambahan teraudit manusia dan duplikat md5 dibuang.
  8. *Keterbatasan:* satu kali latih per fold (satu seed), tanpa selang kepercayaan → selisih ±4 botol dianggap dalam batas acak; kelompok F = botol per sesi, sehingga botol fisik yang sama yang difoto di sesi lain bisa jatuh di fold berbeda (angka F mungkin sedikit optimis); validasi silang hanya memakai data yang sudah ada. **Bukti akhir yang sah = uji terpisah di hari lain dengan botol baru (sesi pembuktian, bagian G `DAFTAR-POTRET-SESI-PEMBUKTIAN.md`).**
- **Pemilahan data warisan `bottle_detection`** (1092 kotak `bottle`): berwarna 143, bukan 723, ragu 226. Dilakukan secara visual oleh Claude, **belum diverifikasi manusia**.
- **TACO** (812 gambar, 1177 kotak negatif non_bottle): foto sampah luar ruang dengan objek kecil, nilai rendah sebagai gambar penuh; rencana: tempel potongan objek ke foto belt. Belum dikerjakan.

## 6. Dataset yang dinilai
Peta lengkap dengan proporsi per model: `PETA-DATASET-19-OKT.md`. Temuan utama: **yang jomplang adalah domain**, bukan jumlah per kelas — `cap_v4` train 70% sintetis, 22% web, 9% studio, 0 belt; kotak botol berwarna hanya 13% di data bottle; cairan belt 40 positif : 220 negatif. Dataset luar yang ditinjau: Roboflow Detection_of_Bottle (ditolak), Water Detection, RF100 soda-bottles, Open Food Facts (CC BY-SA), TACO (CC BY 4.0), ZeroWaste (dilewati: ±30 GB, berkata sandi), Bottled Goods HF (CC BY-NC-SA, tidak dipakai).

## 7. Kesalahan dan koreksi (penting untuk kejujuran ilmiah)
1. Klaim "`cap_v4` lemah di no_cap belt (76%)" hanya berlaku di imgsz 1280; di 640 ia 97,8%.
2. `best.pt` epoch 1 adalah artefak early-stopping pada val yang jenuh (24 foto); dipakai epoch terakhir.
3. Pilot air putih "100%" invalid (confound sesi dan tutup).
4. Nama file set 16 Okt: `capon/capoff` terbalik untuk `kosong` b01, b02, b04, b05 dan `seperempat`; token arah leher salah. Diperbaiki 18 Okt (cadangan: `MANIFEST.backup-sebelum-perbaikan.csv`, `RENAME-LOG.csv`).
5. Pernyataan awal "model merek salah membaca Fanta sebagai coca_cola" keliru: definisi kelas = produk Coca-Cola Company.
6. Klaim "cola gelap adalah target utama deteksi cairan" dilebih-lebihkan.
7. Bug pengunduhan TACO: file 0 byte dari `open(..., 'wb')` sebelum `urlopen`; dihapus dan diunduh ulang.
8. Data warisan dan data luar semula diharapkan membantu; hasilnya memperburuk (L-7), dan tercatat apa adanya.
9. **(23 Okt, ditemukan bersama user)** Percobaan data luar Open Food Facts (s14, L-7) memakai label OTOMATIS dari kategori produk tanpa audit manusia: label 'kosong' ternyata = botol PENUH air bening, ada pula teh/minuman berwarna berlabel 'kosong'. Sebagian alasan "data luar memperburuk" adalah label yang salah, bukan hanya beda sumber foto. Audit visual 92 potongan bersih oleh user (halaman validasi, 2 putaran tinjau): 14 berwarna, 15 air bening, 4 kosong (semua berlabel/berbahan berwarna), 59 buang. Pelajaran audit: (a) posisi botol menentukan (botol REBAH berisi sebagian = realistis seperti belt; botol BERDIRI berisi sebagian = garis cairan melintang, tidak realistis); (b) botol berbahan plastik berwarna (Mizone, botol teh hijau) — warna dari plastik, bukan cairan; (c) yang ragu dibuang agar tidak bias.

## 8. Keterbatasan
- W16 hanya 5 botol, satu sesi, satu pencahayaan, foto statis; Fanta Grape belum ada; pemilahan data warisan belum divalidasi manusia.
- P15 punya confound; hanya dipakai sebagai uji, tidak pernah dilatih.
- Level isi `seperempat` tidak konsisten (sebagian hampir penuh, sebagian hampir kosong).
- Belum diuji pada belt bergerak dan di Jetson.
- **Botol berlabel penuh (rencana batasan, keputusan user 23 Okt — dipakai di TA HANYA bila setelah latih ulang dengan data tambahan masih gagal):** dibedakan dua jenis. (1) Label penuh dengan celah bening (leher/bahu/pantat; Minute Maid, Isoplus, Mizone): cairan masih terlihat lewat celah → diperbaiki dengan data berpasangan berisi/kosong botol yang sama di belt (Isoplus kosong 1/8 → 7/8 dengan data 22 Okt). (2) Label penuh TANPA celah (mis. Oi Ocha): isi secara fisik tidak terlihat kamera mana pun → rumusan batasan: "kamera tidak dapat menentukan isi botol berlabel penuh tanpa celah; kategori ini memerlukan sensor berat (load cell) atau ditolak". Opsi sistem hibrida (kamera utama, sensor berat hanya saat kamera ragu) = keputusan mitra. Botol berlabel penuh TIDAK ditolak semua karena banyak produk mitra (Minute Maid, Nutriboost, Fanta) termasuk jenis (1). Untuk demo, mitra memilih botol.
- Model tutup: tutup hitam sudah dilatih (C-10, v2) tetapi baru 2 produk bertutup hitam (Coke Zero, Powerade); Powerade masih 17/20 (tutup sport lebar). Semua uji = foto statis.
- Hasil "tidak membantu" untuk data luar berlaku untuk resep dan data yang dicoba, bukan bukti umum.

## 9. Pertanyaan yang masih terbuka
Botol berring tanpa tutup valid atau tidak; ukuran botol yang diterima; Fanta Grape ikut atau tidak; definisi "mampu" (usul: ≥95% pada botol tak ikut latih, batas bawah CI ≥90%, per warna dan per tingkat isi); tipe Jetson dan kecepatan (TensorRT); relabel `brand2`; plastik bening diremas di belt (belum ada).

## 10. Peta ke bab TA (usulan, bisa diubah)
| bagian catatan | cocok untuk |
|---|---|
| 1, 2 | Metodologi: lingkungan, alur kerja |
| 3 | Hasil: model cap |
| 4 (L-1 s.d. L-9) | Hasil: deteksi cairan, perbandingan pendekatan, ablasi data |
| 5, 6 | Dataset dan label |
| 7, 8 | Pembahasan: ancaman validitas dan keterbatasan |
| 9 | Penutup: pekerjaan lanjutan |

## 11. Uji kelayakan kategori ukuran dari kamera (20 Okt)
Data: C13, 22 botol dengan volume diketahui dari nama; 84 foto yang kotaknya terpotong tepi frame dibuang. Nearest-centroid, leave-one-bottle-out. Hasil: **luas kotak 22/22 botol (76/76 foto) benar**; panjang kotak 19/22 (400 ml → tanggung; 1000 dan 1500 ml → tanggung). Batas dipilih sebelum uji, di celah ukuran pasar. Batasan: 22 botol, kotak sejajar-sumbu, 1500 ml hanya 1 foto, botol harus utuh dalam frame. **Catatan analisis ulang:** 84 dari 180 foto (47%) dibuang karena terpotong tepi frame, dan dua botol 1500 ml hilang seluruhnya; batas ukuran dipilih setelah melihat tabel panjang (bias ringan). Kategori besar praktis baru bertumpu pada 5 botol.

**Pembaruan 20 Okt (batas baru dari user: kecil ≤350 ml):** uji yang sama memberi **21/22 botol** benar (luas kotak); salah: Pristine 400 ml (luas 219k ≈ botol 350 ml 218k). Botol 350 ml vs 390–400 ml hampir tak terbedakan dari luas kotak; batas luas dipasang 228k. Batas lama (<450 ml) memberi 22/22 karena jatuh di celah 400|500 ml.
