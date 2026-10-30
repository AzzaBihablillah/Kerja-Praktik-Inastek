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

### L-22 Jetson Nano: engine TensorRT & kecepatan NYATA di Nano (24 Okt dini hari)
Perangkat: Jetson Nano Developer Kit B01 4 GB, L4T R32.7.4 (JetPack 4.6.4), mode MAXN + `jetson_clocks`, LAN. Docker `ultralytics/ultralytics:latest-jetson-jetpack4` (Ultralytics 8.4.184, Python 3.8, torch 1.11 CUDA, TensorRT 8.2.0.6, numpy 1.24.4).
- **Engine TensorRT FP16** (`python3 -m rvm_kp.export_models --format engine --half --workspace 1`, di Nano): 5 model dalam **±11 menit** (botol 640, tutup 640, tutup ROI 416, cairan 224, merek 256), jauh lebih cepat dari perkiraan 10–20 menit/model.
- **Kecepatan** (40 frame belt 1080p sesi 20 Okt, profil nano = 720p + tutup di area botol + FP16; `datasets/4_eksperimen/benchmark/jetson_2026-10-24/benchmark_2026-10-23_jetson_nano_auto_trt.txt`): **total 156 ms/frame (p95 171 ms) ≈ 6,4 FPS**; pra-proses 23, botol 72, tutup 44, cairan 14 ms; **merek 16,6 ms per potongan** (sekali per botol, di luar total); RAM proses 2,65 GB dari 4 GB. Untuk mode berhenti-lalu-pindai (3 frame diam per botol): ±0,5 s analisis + ±0,05 s merek per botol → memadai.
- **Kendala yang ditemui & solusinya** (untuk bab implementasi):
  1. Di dalam container `python` = Python 2.7 → wajib `python3`.
  2. TensorRT 8.2 memanggil `np.bool` yang dihapus di numpy 1.24 → engine gagal dimuat; diperbaiki dengan alias `np.bool = bool` di `rvm_kp/__init__.py` (identik dengan bool bawaan).
  3. Model klasifikasi dari `.engine` baru membawa nama kelas setelah prediksi pertama → `rvm_kp/merek.py` membaca nama kelas setelah pemanasan.
  4. Container tidak melihat info board → `/etc/nv_tegra_release` dan `/proc/device-tree/model` dipasang read-only ke container (`JETSON-NANO.md`).
  5. **Jetson restart sendiri satu kali** (01.12 WIB) tepat saat benchmark TensorRT+PyTorch mulai, setelah semua engine jadi; log sebelum restart tidak tersimpan (journal tidak persisten). Laptop tidak mencatat gangguan listrik saat itu. Dugaan: daya kurang saat beban GPU/CPU puncak (perlu dicek: adaptor barrel 5V 4A + jumper J48). Benchmark TensorRT saja sesudahnya berjalan normal. **Restart kedua** (02.01 WIB) saat uji suhu 10 menit baru memuat engine TensorRT: `tegrastats` 38 detik terakhir menunjukkan suhu CPU 29–31 °C, GPU 28–29 °C (bukan panas), RAM naik 1,1 → 3,36 GB dari 3,96 GB (swap mulai dipakai), clock CPU naik-turun 614–1479 MHz (`jetson_clocks` hilang setelah restart). Kedua restart terjadi di fase MEMUAT model → dugaan: lonjakan daya (adaptor/jumper J48) dan/atau memori nyaris penuh (desktop GUI ±1 GB). Uji suhu jangka panjang BELUM dapat dilakukan; ulang setelah cek daya & GUI dimatikan. **Penyebab dipastikan user (24 Okt): adaptor yang dipakai hanya 5V 2A** (≤10 W), sedangkan mode MAXN + `jetson_clocks` butuh 5V 4A lewat barrel + jumper J48 → tegangan turun saat beban puncak → board reset. Angka benchmark 156 ms tetap sah (dari run yang selesai), tetapi uji beban/suhu harus diulang dengan adaptor 5V 4A. Log: `datasets/4_eksperimen/benchmark/jetson_2026-10-24/` (salin `suhu_tegrastats_2026-10-24.log` dari Nano).
- **Belum:** benchmark PyTorch di Nano (pembanding) dan kesesuaian keputusan TensorRT vs PyTorch (dulu ditunda karena restart); uji dengan kamera C922 langsung di Nano; app/PLC di Nano (fastapi/uvicorn/snap7 belum dipasang); pemantauan suhu jangka panjang (`tegrastats`).

### L-23 Benda bukan botol tidak tersingkir (laporan user 26 Okt) — deteksi & penahanan belt
**Masalah:** (1) app hanya bereaksi pada kelas `bottle`; kelas `non_bottle` detektor dibuang -> benda asing tidak memicu berhenti/pesan dan terbawa belt; (2) PLC hanya MB10.0 (jalan/berhenti), tidak ada aktuator jalur tolak (tim akan membuat jalur terima/tolak dengan servo).
**Ukuran detektor pada foto belt 22 Okt** (`2026-10-22_nonbottle_v1` 86 foto = uji ver9c, bukan latih): ver9c -> dikira botol 36, `non_bottle` 49, tak terlihat 1; ver10 kandidat -> dikira botol 5, `non_bottle` **0**, tak terlihat 81 (ver10 dilatih dengan non-botol sebagai latar, jadi tidak pernah menandai benda asing). Presisi `non_bottle` ver9c: muncul di zona tanpa menumpang botol pada 41/86 foto non-botol dan **0/391** foto botol (sesi 13, 20, 22 Okt).
**Perbaikan app:** `non_bottle` di zona = benda asing -> berhenti-lalu-pindai -> tolak 'bukan_botol' (frame + CSV disimpan) -> belt ditahan sampai benda hilang dari zona 1,5 s; kaitan jalur `app/sortir.py`. **Uji offline alur** (Machine dengan kamera tiruan, foto diam, zona pemicu diperlebar meniru benda lewat): non-botol 41/86 ditolak 'bukan botol' (41/41 belt jalan lagi setelah benda diambil), 30/86 dikira botol tetapi semua tertolak di tahap tutup/isi (leher 23, tutup_cairan 5, cairan 2), 15/86 tanpa reaksi; botol 0/94 salah ditahan.
**Belum / keterbatasan:** 15/86 benda (17%) tak terlihat detektor sama sekali -> butuh deteksi "ada benda" yang tidak bergantung kelas. Prototipe `rvm_kp/benda.py` (bandingkan tiap piksel dengan warna barisnya) GAGAL offline: sabuk atas/bawah tidak sejajar baris gambar + pantulan cahaya -> salah alarm 48/64 foto botol. Cara yang tepat = patokan belt kosong per piksel diambil saat sesi mulai, tetapi tidak dapat diuji dengan data lama (tidak ada foto belt kosong; median sesi tercemar benda di tengah & kamera sempat bergeser) -> uji di belt. Alasan tolak untuk benda yang dikira botol menyesatkan ('leher tidak terlihat' alih-alih 'bukan botol'), tetapi tetap ditolak.

### L-24 Detektor ver11: benda bukan botol NYATA diberi kotak non_bottle + penyeimbangan per jenis benda (27 Okt)
**Masalah:** ver10 dilatih dengan foto non-botol nyata sebagai latar kosong -> tidak pernah menandai non_bottle pada benda nyata di belt (0/86), padahal app menolak benda asing lewat kelas non_bottle. Pada benda TEMPELAN (TrashNet) ver10 tetap menandai 97% -> ia belajar MENGABAIKAN benda nyata di belt kita, bukan tidak mampu.
**Resep ver11** (`scripts/pipeline_botol/ver11_nonbotol.py`): resep ver10; 81 foto non-botol nyata 22 Okt diberi kotak non_bottle = usulan ver9c yang DIAUDIT user (5 dari 86 kotak tidak pas -> tidak dipakai; https://claude.ai/artifact/HWEc2Tc8bkcMBLuvBmVgKs); botol semprot & Listerine ikut sebagai non-botol (keputusan user 27 Okt). **Anti-bias (permintaan user):** plastik remas = 35/81 (43%) -> bobot per JENIS benda disamakan (plastik 35 salinan, jenis lain 24-36; batas 12 per foto) -> porsi plastik 9%.
**CV 5 fold per benda (foto nyata 22 Okt, ambang 0,35):**

| | ver9c | ver11 |
|---|---|---|
| non-botol DITERIMA sebagai botol | 33/81 | **5/81** (4 = Listerine) |
| non-botol DITANDAI non_bottle | 50/81 | **78/81** (semua 12 jenis benda tertandai) |
| botol terdeteksi | 154/154 | 154/154 |
| botol salah ditandai non_bottle | 24/154 | **2/154** |

ver10 (CV-nya sendiri, tanpa semprot/Listerine): non-botol diterima 0/74 vs ver11 1/74, tetapi ver10 tidak pernah menandai non_bottle.
**Uji di data LAIN yang tak dilatih** (`scripts/pipeline_botol/ver11_uji_lain.py`, `datasets/4_eksperimen/botol/ver11/uji_lain.txt`) — ver9c / ver10 / ver11:
- TrashNet ditahan ditempel ke belt (382 non-PET: kardus, kaca, logam, kertas, plastik lain, sampah): diterima 7% / 5% / **4%**; ditandai non_bottle 96% / 97% / **98%** (ver11 tertinggi di setiap kategori non-PET) -> kelas non_bottle TIDAK menyempit menjadi 'plastik kresek'.
- TACO tempel (328): diterima 8,8% / 7,9% / **7,0%**; dikenali non_bottle 313 / 318 / 314.
- Botol: sesi 20 Okt 117/117 semua; old_test recall botol 1,000 / 1,000 / **0,990** (turun ~3 dari 284), mAP50 0,995/0,995/0,993.
- Cek jalan pintas (botol asli ditempel): studio 93,5% / 93,5% / **95,3%**; belt **100% / 97,1% / 95,1%** (ver11 menolak 5/102 botol tempelan belt, 10 dicap non_bottle) -> sedikit lebih condong menandai non_bottle pada botol tempelan.
**Kesimpulan:** ver11 menutup celah benda asing nyata (ditandai 50 -> 78/81) tanpa bias ke plastik dan tanpa kehilangan botol nyata (154/154, sesi 20 Okt 117/117); kekurangan kecil: recall old_test 0,99 dan botol tempelan belt 95%. **Belum aktif**: uji di belt nyata (Kiosk Capturer, botol & non-botol BARU) sebelum menggantikan ver9c. Listerine masih diterima 4/4 (bentuknya botol) -> butuh lebih banyak contoh botol non-minuman.

### L-25 Sesi pembuktian di belt nyata 27 Okt sore (Kiosk Capturer, PLC, kandidat bayangan ver11 + cairan_berisi + merek)
Data: `datasets/1_foto_mentah/2026-10-27_scan_app/` — **6 sesi logis** (folder potongan akibat sesi terputus saat conveyor diperbaiki digabung 27 Okt malam; daftar di BACA.txt): sesi1 kosong tanpa tutup Coca-Cola, sesi2 kosong bertutup Coca-Cola, sesi3 kosong bertutup non, sesi4 kosong tanpa tutup non, sesi5 berisi air bertutup (merek campur; 1 botol tanpa tutup dari teman, tak diketahui yang mana -> label tutup +-1), sesi6 bukan botol; sesi0 = percobaan awal saat baterai (tidak dinilai). Label = label KELOMPOK dari user (lemah: satu botol salah kelompok = salah label). Sesi awal (folder tanggal, 14 botol) dijalankan saat laptop memakai BATERAI -> deteksi botol ~250 ms/frame (normal ~14 ms), berhenti ~3 s, video tersendat -> diulang setelah dicas; model kandidat lalu dipindah ke thread terpisah (berhenti 1,5-2,4 s).
**Botol kosong (sesi 049-052, 72 botol: Coca-Cola Co./bukan x bertutup/tanpa tutup):**

| | model aktif (ver9c, cap_belt_v2, cairan_RH) | kandidat (ver11, cairan_berisi, ukuran panjang) |
|---|---|---|
| keputusan akhir sesuai label | 67/72 | **70/72** |
| cairan: botol kosong terbaca kosong | 60/72 | **70/72** |
| tutup benar | **71/72** | 70/72 |

Kesalahan aktif: botol kosong berlabel penuh/berwarna/semi-buram terbaca berisi (Fanta, Ichitan pisang, Big PET hijau, botol semi-buram, Frestea). Tutup LOLOS 1x (aktif & kandidat): Alfamart air mineral 1,5 L tutup BIRU MUDA (0/3 frame melihat tutup) -> botol bertutup diterima. Kandidat salah 'bertutup' 2x (kotak ver11 menggeser area leher). ver11 tidak mendeteksi botol #1 sesi 049 (satu kali).
**Merek (model merek kandidat, coca_cola vs bukan):** 64/72 sesuai label kelompok; salah: Frestea (2x, p 0,00-0,01 -> data Frestea kurang), Fanta 1,5 L (0,45), Vit merah (2x terbaca Coca-Cola), Fitri (0,54; 0,97), botol bening berlabel kartun (0,90).
**Benda bukan botol (sesi 060, 22 benda, user memantau: semua terlihat):** 22/22 DITOLAK, 0 diterima: 18 langsung 'bukan botol' (kelas non_bottle ver9c), 4 dikira botol oleh ver9c tetapi tertolak di tahap tutup/isi; ver11 tidak mendeteksi botol pada keempatnya. Masalah yang muncul & diperbaiki saat sesi: noda bekas selotip di belt terbaca non_bottle 0,50 dan belt MACET menunggu benda 'diambil' -> ambang benda asing 0,60 (data 22 Okt: 50/53 benda nyata >= 0,60), belt TIDAK lagi ditahan (keputusan user: benda asing ikut terbawa ke pembuangan), batas tahan 15 s sebagai pengaman.
**Air bening (sesi 053-059, 35 botol):** 34/35 ditolak; label merek DIREVISI user menjadi campuran, label tutup belum pasti (sebagian mungkin tanpa tutup) -> angka cairan (aktif 28/35, kandidat 24/35 terbaca berisi) BELUM sah sampai user mengaudit tutup. **Audit user 27 Okt malam** (https://claude.ai/artifact/NEtxoetGHqam8V1YLP9daL, `datasets/4_eksperimen/air_tutup_audit/usulan.csv`): 34 bertutup dikonfirmasi, 1 dibuang; user RAGU soal ISI (botol dimasukkan teman; beberapa tampak kosong; air bening di botol bening tak terbaca dari foto) -> sesi air 053-059 TIDAK dipakai menilai model cairan (hanya contoh botol bertutup). Penilaian air bening & keputusan aktivasi cairan_berisi **Koreksi user (27 Okt malam):** 95% botol dimasukkan user sendiri sesuai label; SEMUA botol air DIISI user (penuh/sedikit acak) dan bertutup -> sesi air SAH untuk menilai cairan. Botol S53B195 (Nestlé Pure Life, tutup BIRU) = bertutup + berisi air, tetapi DITERIMA: model tutup 0/3 frame melihat tutup DAN cairan aktif 0,01 / kandidat 0,00 -> kesalahan ganda. Bersama Alfamart 1,5 L (tutup biru muda, sesi 051) = 2 botol bertutup lolos hari ini, keduanya TUTUP BIRU -> butuh data tutup biru/biru muda untuk latih ulang model tutup.
**Cairan, gabungan data hari ini** (72 botol kosong; 35 botol berisi air bening bertutup, diisi user):

| aturan 'berisi' | kosong salah 'berisi' | air terdeteksi |
|---|---|---|
| aktif RH | 12/72 | 28/35 |
| kandidat cairan_berisi | 2/72 | 24/35 |
| ATAU | 12/72 | 29/35 |
| DAN | 2/72 | 23/35 |
| rata-rata p >= 0,5 | 5/72 | 26/35 |

Tarik-ulur: RH menangkap lebih banyak air tetapi menolak 1 dari 6 botol kosong sah; kandidat hampir tak salah tolak tetapi meloloskan ~30% air. Usul: rata-rata keduanya sebagai kandidat berikutnya. Catatan: semua botol air bertutup (tutup ikut terlihat); kasus nyata yang perlu ditangkap = air TANPA tutup -> perlu sesi kecil air bening tanpa tutup. Pilihan aturan = kebijakan (mana lebih buruk: botol sah ditolak vs air lolos) -> keputusan user/mitra.
**Pencatat 'ada benda' (patokan belt kosong, mode catat):** BELUM layak: kejadian 'tak dikenal' palsu dari noda belt yang bergerak, getaran kamera saat belt jalan (tepi rangka), dan perubahan cahaya; tidak memengaruhi keputusan. Data tersimpan untuk penyetelan. **Penyebab perubahan cahaya dipastikan user:** lampu ditutup PLASTIK LEPAS yang bergerak -> cahaya bocor berubah-ubah (sesi 062: 11 kejadian palsu tanpa benda). Rekomendasi: peredam lampu kaku terpasang mati (akrilik/mika buram) + tutup atas zona kamera.
**Kesimpulan sementara:** kandidat cairan_berisi jelas lebih baik dari RH pada botol kosong berlabel (60 -> 70/72); model tutup aktif tetap andal kecuali tutup biru muda; benda bukan botol tertolak semua; merek lemah pada Frestea & pesaing berlabel merah/hijau.

### L-26 Kesimpulan uji belt 27 Okt (6 sesi, 108 botol + 22 bukan botol) — disusun 27 Okt malam
Statistik: rentang = interval Wilson 95%; perbandingan aktif vs kandidat = uji McNemar berpasangan (botol yang sama).
**Didukung data:**
1. **Sistem utuh berjalan di belt nyata** (kamera + model + PLC + berhenti-lalu-pindai), 130 benda, berhenti median 1,7 s/botol (maks 2,4 s). Keputusan akhir botol baru: aktif 101/107 (94%; 88-97%), kandidat 104/107 (97%; 92-99%).
2. **Kandidat cairan_berisi lebih baik pada botol KOSONG** (bukan kebetulan): salah 'berisi' 2/72 vs aktif 12/72; dari 10 botol yang berbeda hasil, 10 dimenangkan kandidat (McNemar p=0,002). Kesalahan aktif terkumpul pada botol berlabel penuh/berwarna/semi-buram (Fanta, Ichitan, Frestea, Big PET hijau).
3. **Model tutup andal** (106/108) dengan **titik buta tutup BIRU/biru muda**: kedua botol bertutup yang lolos bertutup biru (Alfamart 1,5 L sesi3; Nestlé Pure Life berisi air sesi5). Belum terbukti semua tutup biru gagal; prioritas data latih.
4. **Benda bukan botol tertolak 22/22** (18 langsung 'bukan botol', 4 lewat tahap tutup/isi); ver11 tidak mengira botol satu pun.
5. **Model merek belum layak untuk bonus**: 64/72; Coca-Cola dikenali 20/22, tetapi 6/50 non-Coca-Cola dianggap Coca-Cola (Vit, Fitri) -> ~1 dari 8 botol non-Coca-Cola akan dapat bonus; Frestea tidak dikenali.
6. **Kondisi operasi berpengaruh besar**: laptop berbaterai (GPU hemat daya) -> lambat & tersendat; cahaya bocor dari penutup lampu plastik yang bergerak -> salah alarm pencatat 'ada benda'. Sejalan dengan uji ketahanan (§8): daya & pencahayaan harus distabilkan.
**Belum bisa disimpulkan:** aktif vs kandidat secara keseluruhan (101 vs 104, McNemar p=0,38: setara secara statistik); deteksi AIR (aktif 28/35 vs kandidat 24/35, p=0,22; n kecil; semua botol air bertutup, padahal kasus nyata = air TANPA tutup); ukuran (tidak ada label ml); medan lain (satu hari, satu belt, satu pencahayaan); label = label kelompok (satu botol salah kelompok = salah label; sesi5 tutup +-1).
**Rekomendasi:** (1) aktifkan cairan_berisi atau rata-rata kedua model (bukti terkuat: jauh lebih sedikit botol sah ditolak), didahului sesi kecil air bening TANPA tutup — keputusan user; (2) pertahankan model tutup aktif, kumpulkan botol bertutup biru/biru muda untuk latih ulang; (3) bonus merek tetap MATI sampai dilatih ulang dengan data 27 Okt (Frestea, Vit, Fitri); (4) ver11 belum menggantikan ver9c (kotaknya menyebabkan 2 salah 'bertutup' + 1 botol tak terdeteksi); (5) stabilkan mesin: daya selalu tercolok (atau Jetson + adaptor 5V 4A), peredam lampu kaku.

### L-27 Latih ulang model cairan 'berisi' dengan data belt 27 Okt (28 Okt dini hari)
Kode: `scripts/pipeline_cairan/s36_berisi_27okt.py` (resep s35 = s32 RN + air bening 22 Okt + 3 salinan geser-hue; yolov8n-cls 224, 30 epoch, kelas seimbang). Data baru: frame diam 27 Okt, potongan dari kotak ver9c terbesar per frame (`fold_crop`), 321 frame / 107 botol (sesi1-4 kosong = 72 botol -> 'kosong'; sesi5 air bening diisi user = 35 botol -> 'berwarna' = berisi; baris bukan_botol & `buang_audit` dikeluarkan). Daftar: `datasets/4_eksperimen/cairan/crops27.csv` (id `T27_<sesi>_<botol>_<frame>`).
**Validasi:** CV 5 fold, kelompok = SESI 27 Okt (bukan botol: botol fisik yang sama kemungkinan dipakai di sesi1&2 dan sesi3&4) + kelompok lama seperti s32; tiap fold menguji SATU sesi 27 Okt yang tidak dilatih. Keputusan per botol = median p dari 3 frame (sama dengan aplikasi), ambang 0,5. Pembanding pada botol yang sama: cairan_RH (aktif) dan cairan_berisi_kandidat (s35), keduanya tak pernah melihat data 27 Okt. Hasil: `cv_27okt.jsonl`, `cv_27okt_report.txt`.

| model | kosong salah 'berisi' (72) | air terdeteksi (35) | per sesi benar: s1 / s2 / s3 / s4 / s5 |
|---|---|---|---|
| aktif RH | 15/72 | **31/35** | 8/11, 10/11, 19/27, 20/23, 31/35 |
| kandidat berisi (s35) | **3/72** | 27/35 | 11/11, 11/11, 25/27, 22/23, 27/35 |
| berisi_v2 (CV, + data 27 Okt) | **3/72** | 24/35 | 11/11, 11/11, 24/27, 23/23, 24/35 |
| rata-rata p RH & s35 | 6/72 | 28/35 | |

Regresi kelompok lama (frame, fold yang sama): berisi_v2 benar 319/351 (berwarna 175/198, kosong 144/153).
**Tafsir jujur:** (1) menambah 72 botol kosong belt TIDAK memperbaiki botol kosong sesi yang tak dilatih (3/72 = sama dengan s35) — s35 sudah nyaris sempurna di sini; (2) air turun 27 -> 24/35, tetapi fold sesi5 TIDAK pernah melihat air di belt (semua air 27 Okt ada di sesi5), jadi angka ini hanya menunjukkan bahwa tambahan botol kosong sedikit menggeser model ke 'kosong'; manfaat data air belt sesi5 BELUM dapat diukur dengan data yang ada; (3) angka ulang-potong offline sedikit berbeda dengan log aplikasi (RH 15 vs 12, s35 air 27 vs 24) karena kotak dan potongan dibuat ulang dari foto (aplikasi memakai kotak pelacak), arah kesimpulan sama.
**Model final:** `datasets/3_model_final/cairan_berisi_v2_kandidat.pt` (semua data termasuk 35 botol air belt). TIDAK menggantikan kandidat s35 maupun model aktif. Cara menilai: sesi belt berikutnya dengan air bening (sebaiknya TANPA tutup, per level isi) -> skor offline frame tersimpan untuk RH, s35, v2 dan rata-rata. Catatan teknis: percobaan pertama gagal karena folder data latih = folder keluaran (bobot terhapus); diperbaiki (`cv27_data/`).

### L-28 Audit data untuk merek & detektor ver12 (28 Okt dini hari)
**Merek botol air sesi5** (https://claude.ai/artifact/54QAtZEtWudesJDd2FH9GE, `datasets/4_eksperimen/merek_air_audit/hasil_audit.csv`): usulan = merek_coca_kandidat; user mengubah 5/35 (model benar 30/35): BIG Cola & 3 botol lain dikira Coca-Cola -> bukan; Frestea dikira bukan -> Coca-Cola. Hasil: 10 Coca-Cola Co., 25 bukan. Label merek sesi1-4 = label kelompok (Coca-Cola: sesi1-2, bukan: sesi3-4).
**Kotak detektor ver12** (https://claude.ai/artifact/Fh3joDUr8T3YuyfTAfPGhA, `datasets/4_eksperimen/ver12_audit/usulan.csv`): usulan = ver11 per frame (3 frame/benda), kelas mengikuti sesi (sesi6 = non_bottle). 132 benda; user menandai 3 'salah' (V53B214 kotak di tepi tanpa botol utama, V60B33, V60B45 kotak goyang) -> 129 benda / 387 frame dipakai. Deteksi tambahan ver11 (kotak kuning, 13 benda) diperiksa visual: pantulan cahaya di belt (tepi kiri), benda gelap/rangka di ujung belt (tepi kanan), sekali tangan — tidak ada botol terpotong -> dibiarkan TANPA label sebagai contoh negatif (rencana awal menutup abu dibatalkan). Latih ver12: `scripts/pipeline_botol/ver12_belt27.py` (CV 2 fold per pasangan sesi: A = sesi1+2+5, B = sesi3+4+6; dinilai per frame asli vs ver11).
**Merek v2** (`scripts/pipeline_merek/m08_belt27.py`; resep A m07 + 321 frame / 107 botol 27 Okt; CV 2 fold: A uji sesi1+2 + separuh sesi5, B uji sesi3+4 + separuh sesi5; per botol = rata-rata p 3 frame; hasil `datasets/4_eksperimen/merek/cv_belt27_report.txt`):

| model | benar | Coca-Cola Co. dikenali | non-Coca salah dapat bonus |
|---|---|---|---|
| m07 (kandidat, tanpa data 27 Okt) | 101/107 | 29/32 | 3/75 |
| m08 v2 (CV) | **104/107** | 30/32 | **1/75** |

Berbeda hasil pada 5 botol: 4 dimenangkan v2, 1 oleh m07 (McNemar p=0,38: belum signifikan, n kecil). Yang tetap salah: 2 botol sesi1 (S49B9, S49B22, p 0,00 — Frestea; Frestea hanya ada beberapa botol, sebagian di fold uji) dan S52B117 (non, p 0,73). Kebocoran mungkin: botol sesi5 bisa botol fisik yang sama dengan sesi1-4 -> angka sesi5 (35/35) optimistis. Angka m07 offline (sesi1-4: 67/72) berbeda dengan log aplikasi (64/72) karena potongan dibuat ulang dari kotak ver9c. Final: `datasets/3_model_final/merek_coca_v2_kandidat.pt` (semua data), BELUM dipakai; bonus merek tetap mati. Uji sah: sesi belt berikutnya dengan botol yang belum pernah dilatih (terutama Frestea, Vit, Fitri, BIG).
**Detektor ver12** (hasil `datasets/4_eksperimen/botol/ver12/cv_report.txt`, 387 frame / 129 benda, ambang 0,35):

| | ver11 | ver12 (CV) |
|---|---|---|
| botol terdeteksi (frame, IoU>=0,5 dg kotak audit) | 327/327 | 327/327 |
| botol salah non_bottle | 0/327 | 0/327 |
| bukan botol ditandai non_bottle / diterima sbg botol | 60/60 / 0 | 60/60 / 0 |
| frame dengan deteksi TAMBAHAN (alarm palsu: pantulan, rangka ujung belt, tangan) | 21/387 | **4/387** |

Tafsir jujur: kotak acuan = usulan ver11 yang diaudit, dan frame yang tidak dideteksi ver11 tidak masuk data -> deteksi 100% ver11 terjadi BY CONSTRUCTION (bias menguntungkan ver11); ver12 menyamainya. Peningkatan yang nyata = alarm palsu turun ~5x (penting untuk benda asing/penyapu). Masalah kotak ver11 yang menggeser area leher (2 salah 'bertutup' 27 Okt) TIDAK diuji di sini -> perlu skor ulang pipa tutup dengan kotak ver12 pada frame 27 Okt atau sesi berikutnya. Final: `datasets/3_model_final/botol_ver12_kandidat.pt`, BELUM dipakai (kandidat bayangan app masih ver11).

### L-29 Cek ambang cairan & jejak data tak-teraudit (28 Okt, permintaan user setelah tinjauan kesalahan proses)
**Ambang (botol 27 Okt, p per botol dari `cv_27okt.jsonl`, tanpa latih ulang):** AUC RH 0,896 · s35 0,937 · v2 (CV) 0,917 · rata-rata RH+s35 **0,941**.

| ambang | RH kosong salah / air | s35 kosong salah / air | rata-rata kosong salah / air |
|---|---|---|---|
| 0,1 | 22 / 33 | 4 / 27 | 18 / 33 |
| 0,3 | 16 / 33 | 4 / 27 | 14 / 31 |
| 0,5 | 15 / 31 | 3 / 27 | 6 / 28 |
| 0,7 | 13 / 30 | 3 / 24 | 3 / 26 |
| 0,9 | 10 / 27 | 2 / 22 | 3 / 22 |

Kesimpulan: ambang BUKAN penyebab utama di data 27 Okt. Air yang luput dari s35 diberi p < 0,08 (salah yakin, 8 botol: S53B185, S53B195, S59B318, S53B177, S59B301, S54B229, S57B265, S54B226), jadi menurunkan ambang tidak menolong; RH hanya bisa menukar salah tolak dengan air lolos. Kalibrasi silang (ambang dipilih di separuh botol, diuji di separuh lain) untuk rata-rata RH+s35 memilih ambang ±0,5 (stabil), untuk s35/v2 memilih ambang ~0,0001-0,07 (tidak stabil). Perbaikan nyata untuk air harus dari DATA (air di belt), bukan ambang. Peringatan: dihitung dan dinilai pada data yang sama (kecuali kalibrasi silang) -> deskriptif.
**Jejak data yang dipilah Claude tanpa audit:** (a) pemilahan visual `bottle_detection` (berwarna/bukan/ragu, `old_relabel_manual.csv`) dipakai `s06_det_data.py` hanya untuk MEMBUANG gambar berkotak 'ragu' pada data detektor B2 (ver8 -> ver8b/ver9c/ver10/ver11/ver12); label kotaknya tetap label tim asli (bottle/non_bottle), jadi pemilahan Claude tidak mengubah label detektor aktif, hanya pilihan gambar. Kelas berwarna/bukan dari pemilahan itu hanya dipakai detektor 3 kelas A3 (ditolak) dan opsi `s03 --old` (percobaan awal). (b) 540 potongan WEB (Fanta internet) di `crops.csv` dan potongan warisan TIDAK masuk model cairan aktif/kandidat: `s30_final3`/`s32`/`s35`/`s36` hanya memakai src W16, V20, crops22 (+22b, +27 Okt). (c) Label kotak data warisan tim sendiri tidak pernah kita audit penuh (contoh: 1 Aqua berlabel non_bottle, §4b).

### L-30 Batasan lingkup: botol cover penuh / buram / berwarna gelap tidak dihitung (28 Okt, keputusan user)
Alasan: isi botol tidak terlihat kamera (sleeve/label menutup badan botol, botol buram atau berwarna) -> di luar kemampuan model berbasis kamera. Usulan dari foto oleh Claude (19 botol), DIAUDIT user (https://claude.ai/artifact/MTyrmaX1kMs9VhWKdaFSHy, db `cover_audit`): 2 dikembalikan ke hitungan (S49B20, S53B177 = Minute Maid Pulpy, label tidak menutup penuh) -> **17 dikeluarkan**: Frestea besar, Nutriboost (3), Ichitan Banana Milk (3), botol susu putih buram (2), botol jeruk sleeve penuh, Isoplus (3), BIG hijau (4). Daftar: `datasets/4_eksperimen/batasan_cover_27okt.json`.

| metrik (27 Okt) | semua 107 botol | dalam batasan (90) | yang dikeluarkan (17) |
|---|---|---|---|
| keputusan benar — aktif | 101/107 = 94,4% [88–97] | **88/90 = 97,8% [92–99]** | 13/17 = 76,5% |
| keputusan benar — kandidat | 104/107 = 97,2% [92–99] | 87/90 = 96,7% [91–99] | 17/17 = 100% |
| kosong terbaca kosong — aktif | 60/72 = 83,3% [73–90] | **58/61 = 95,1% [87–98]** | 2/11 = 18% |
| kosong terbaca kosong — kandidat | 70/72 = 97,2% | 61/61 = 100% [94–100] | 9/11 = 82% |
| air terdeteksi — aktif | 28/35 = 80,0% | 22/29 = 75,9% [58–88] | 6/6 |
| air terdeteksi — kandidat | 24/35 = 68,6% | 20/29 = 69,0% [51–83] | 4/6 |
| tutup benar — aktif | 105/107 = 98,1% | 88/90 = 97,8% | 17/17 |
| merek benar (m07) | 94/107 = 87,9% | 79/90 = 87,8% | 15/17 |
| non-Coca salah dapat bonus | 10/75 = 13,3% | 8/61 = 13,1% | 2/14 |

Temuan: 4 dari 6 kesalahan keputusan model aktif = botol kosong ber-cover penuh terbaca 'berisi' (label berwarna dikira cairan). Dalam batasan, sisa kesalahan aktif hanya 2 botol bertutup BIRU yang lolos (S51B96, S53B195); kandidat menambah 1 salah 'bertutup' (S52B107, kotak ver11). Batasan TIDAK memperbaiki deteksi air (~70–80%), tutup biru, maupun merek. Penulisan TA: laporkan KEDUA angka (semua & dalam batasan) dan nyatakan batasan sebagai lingkup, bukan menghapus data. Kebijakan mesin untuk botol di luar batasan belum diputuskan (kini model aktif cenderung menolaknya sebagai 'berisi').
**Keterangan penguji (29 Okt):** kedua botol tutup biru yang lolos (S51B96 Alfamart 1,5 L, S53B195 Nestlé) memakai tutup TIPIS PENGGANTI dari botol lain, warnanya beda dengan cincin leher aslinya -> dugaan: model tutup (dilatih dari tutup & cincin sewarna) membaca leher sebagai "cincin tanpa tutup"; perlu uji botol bertutup pengganti lain. **Air tanpa tutup** diperkirakan dari data ini (model isi menilai badan botol, tutup dinilai terpisah): yang lolos sebagai kosong bila tanpa tutup = yang tak terdeteksi berisi -> aktif 7/29 (24%) dalam batasan, 7/35 (20%) semua; kandidat 9/29 (31%). Tutup ikut terlihat di foto -> pengaruhnya pada skor isi belum diuji langsung. Laporan: Claude Docs "Hasil Uji Belt RVM 27 Oktober 2026" https://claude.ai/artifact/RKV4yrUbMgPY8Y3xMqs1e6.
**Rencana 30 Okt (keputusan user, 29 Okt):** TIDAK langsung latih ulang model isi dengan data air; uji dulu PER TINGKAT ISI untuk menentukan batas deteksi (lingkup "andal bila isi >= x"). Alasan: air sesi 27 Okt diisi ACAK (penuh/sedikit) tanpa label per botol -> hasil air (76–80%) tak bisa dipecah per tingkat; dugaan user: yang luput = air sangat sedikit (±1/6) yang dari atas mirip botol kosong (skor isi botol luput ~0, konsisten). Desain: kosong, ±1/6, ±1/3, ±1/2, penuh; botol fisik SAMA di tiap tingkat; tiap botol sekali bertutup & sekali tanpa tutup (sekaligus uji pengaruh tutup pada skor isi); 8–10 botol per tingkat, di dalam batasan L-30. Kiosk Capturer: label isi baru `air_1-6`, `air_1-3`, `air_1-2`, `air_penuh` (harapan = tolak). Sesi 27 Okt = evaluasi umum. Latih ulang hanya bila batas terukur terlalu tinggi untuk kebijakan mitra; sisa air sangat sedikit dapat dilengkapi load cell di bawah talang.

### L-31 Sesi belt 29 Okt (Kiosk Capturer + PLC, talang ESP32 tidak dipasang)
Data: `datasets/1_foto_mentah/2026-10-29_scan_app/` (sesi_105 salah label -> dipindah ke `_dibuang_user/`). Model AKTIF = ver9c + cap_belt_v2 + RH + merek v2 (bonus +10) + ukuran 'panjang'; bayangan = ver12 + cairan_berisi (s35) + merek v2. Semua botol di dalam batasan L-30 (tanpa cover penuh). Rentang = Wilson 95%.

| sesi | label | botol | hasil model aktif |
|---|---|---|---|
| 106 | kosong, bertutup, Coca-Cola Co. | 25 | ditolak benar 25/25 = 100% [87–100] |
| 107 | kosong, tanpa tutup, Coca-Cola Co. (botol sama dengan 106) | 25 | diterima 23/25 |
| 108+109 | kosong, tanpa tutup, non-Coca | 10+40 | diterima 48/50 |
| 110 | kosong, bertutup, non-Coca, **tutup dicampur** (sebagian sesuai warna cincin, sebagian ditukar; tidak dicatat per botol) | 50 | ditolak benar **44/50 = 88,0% [76–94]** |
| 111 | **air ±1/2**, bertutup (tutup dicampur), Coca-Cola Co. | 25 | ditolak benar 24/25; air terdeteksi aktif RH **23/25 = 92% [75–98]**, kandidat 19/25 = 76% [57–89], rata-rata 22/25; tutup terlihat 23/25; 1 LOLOS (#271: tutup tak terlihat DAN air tak terdeteksi, p 0,001) |
| 112 | **air ±1/6**, bertutup (tutup dicampur), Coca-Cola Co. | 25 | ditolak benar 25/25 (24 tertolong tutup); air terdeteksi aktif RH **15/25 = 60% [41–77]**, kandidat **5/25 = 20% [9–39]**, rata-rata 8/25, ATAU 16/25; tutup terlihat 24/25 |
| 113 | **air ±1/2**, bertutup, non-Coca | 50 | ditolak benar 50/50; air terdeteksi aktif RH **48/50 = 96% [87–99]**, kandidat 46/50 = 92% [81–97], rata-rata 48/50; tutup terlihat 41/50 |
| 114 | **air ±1/6**, bertutup (tutup dicampur), non-Coca | 50 | ditolak 47/50 (**3 berisi LOLOS**: tutup & air sama-sama tak terlihat); air terdeteksi aktif RH **42/50 = 84% [71–92]**, kandidat 27/50 = 54% [40–67], rata-rata 34/50 |

**Hasil cairan per tingkat isi (gabungan Coca + non-Coca, dinilai dari skor isi saja, tutup diabaikan):** ±1/2: RH **71/75 = 95% [87–98]**, kandidat 65/75 = 87% [77–93]; ±1/6: RH **57/75 = 76% [65–84]**, kandidat 32/75 = 43% [32–54]. Per produk ±1/6: Coca 60% vs non-Coca 84% (RH). Kesimpulan: air ±1/2 andal (≥ 92% di kedua kelompok); air ±1/6 di bawah andal, terutama pada botol Coca-Cola. RH jauh lebih baik dari cairan_berisi untuk air sedikit -> RH tetap model isi aktif. Dalam operasi, botol ±1/6 tanpa tutup diperkirakan lolos ~1 dari 4.
**Air PENUH (sesi 115–119, 62 botol bertutup; sesi 117 kosong/terputus):** Coca 4/9 (44%), non-Coca 22/25 (88%), campur botol besar (galon 2) 12/13 (92%), "penuh maksimal" hampir tanpa rongga udara (campur tanggung/kecil) 12/15 (80%). Gabungan RH **50/62 = 81% [69–89]**, kandidat 13/62 = 21% [13–33]. TEMUAN: air penuh TERDETEKSI LEBIH BURUK dari ±1/2 (95%). Dugaan: model membaca BATAS air–udara (garis permukaan/pantulan); botol bening terisi penuh tanpa rongga udara tampak seragam seperti botol kosong. Konsekuensi: lingkup deteksi air bening bukan "makin penuh makin mudah" -> paling andal pada isi sebagian (±1/2), menurun pada ±1/6 dan pada penuh. Data penuh & ±1/6 = bahan latih ulang model isi.

**Kurva tingkat isi (sementara, Coca-Cola, bertutup):** air terdeteksi RH ±1/2 = 92% -> ±1/6 = 60%; kandidat 76% -> 20%. Batas deteksi andal terletak antara 1/6 dan 1/2 (perlu ±1/3); pada ±1/6 skor RH menyebar 0,00–1,00 (tidak ada ambang yang memisahkan). Keputusan akhir tetap benar karena tertolong tutup; air ±1/6 TANPA tutup akan lolos ~40% (RH).
**Keputusan user (29 Okt):** tingkat isi cukup ±1/2 dan ±1/6 (tanpa ±1/3); cairan dinilai dari skor isi saja (tutup DIABAIKAN dalam evaluasi cairan, hasil tutup menjadi bahan latih ulang model tutup). Sisa: sesi non-Coca air ±1/2 dan ±1/6.

- **Isi (botol kosong tanpa tutup, 75):** aktif RH diterima 71/75 = 94,7% [87–98] (4 botol kosong dikira berisi, skor 0,59–0,90); kandidat cairan_berisi 150/150 botol kosong terbaca kosong (skor 0,0). Total botol kosong = 150: 75 tanpa tutup (sesi 107–109) + 75 bertutup (sesi 106 tutup asli 25/25 tertolak, sesi 110 tutup campuran 44/50) -> bertutup tertolak 69/75 = 92%.
- **Tutup campuran (sesi 110):** 6 botol bertutup LOLOS (0/3 frame melihat tutup, satu 1/3): pola dari foto = tutup kemungkinan beda warna dgn cincin (#174, #207, #244), tutup biru tua (#171, #173), botol miring di belt (#238); tetapi tutup tukaran tidak selalu lolos (#180 biru muda di cincin merah tertolak benar). Penguji tidak mencatat botol mana yang ditukar -> dievaluasi umum (tutup sama + beda warna), data menjadi bahan latih ulang model tutup (perlu audit kotak leher/tutup).
- **[TIDAK SAHIH — bug bayangan, lihat L-32]** **Detektor kandidat ver12 + model tutup:** sesi 106: 4 botol tidak terdeteksi + 3 botol bertutup diterima; sesi 110: 2 tidak terdeteksi, 7 diterima -> kekhawatiran L-28 terbukti: kotak ver12 mengganggu tutup; ver9c tetap aktif.
- **[TIDAK SAHIH — bug bayangan, lihat L-32; merek v2 offline: Coca 108/109, non-Coca bonus salah 10/225]** **Merek v2 pada botol BARU (bonus aktif):** Coca-Cola tanpa tutup dikenali 19/25 = 76,0% [57–89]; Coca-Cola bertutup 11/25 = 44,0% [27–63]; non-Coca salah dapat bonus 7/40 = 17,5% [9–32] -> jauh di bawah CV (97%): model merek belum menggeneralisasi ke botol baru; keputusan user: bonus tetap aktif, kesalahan menjadi bahan evaluasi & latih ulang.
- Insiden: venv proyek sempat dibuat ulang dengan Python 3.14 (`python -m venv venv`, langkah untuk laptop teman) -> app gagal start (pydantic_core cp313); dipulihkan ke Python 3.13 tanpa menghapus paket.

### L-32 Latih ulang isi, tutup, merek dengan data belt 27 + 29 Okt; bug kandidat bayangan (29 Okt malam)
**Desain validasi (semua model): PER HARI** — model dilatih TANPA satu hari belt lalu diuji pada hari itu (A29: latih data lama + 27 Okt, uji 29 Okt; B27: latih lama + 29 Okt, uji 27 Okt). Botol uji tidak pernah dilihat saat latih. Per botol: isi = median 3 frame diam (ambang 0,5), merek = rata-rata, tutup = suara mayoritas 3 frame. Rentang = Wilson 95%. Model final = latih SEMUA data (lama + 27 + 29), disimpan sebagai `*_kandidat.pt`; model aktif TIDAK diganti tanpa keputusan user.

**Isi cairan v3** (`scripts/pipeline_cairan/s37_isi_29okt.py`, yolov8n-cls 224; data 351 baris lama + 321 frame 27 Okt + 1085 frame 29 Okt, 3 salinan geser-hue; final `cairan_isi_v3_kandidat.pt`):

| uji (hari tak dilatih) | RH aktif | s35 (cairan_berisi) | **v3** |
|---|---|---|---|
| 29 Okt kosong (75 bertutup + 75 tanpa) | 122/150 = 81,3% [74–87] | 150/150 | **149/150 = 99,3% [96–100]** |
| 29 Okt air ±1/2 | 72/75 = 96,0% [89–99] | 71/75 | **73/75 = 97,3% [91–99]** |
| 29 Okt air ±1/6 | 60/75 = 80,0% [70–87] | 39/75 | **66/75 = 88,0% [79–94]** |
| 29 Okt air penuh | 52/62 = 83,9% [73–91] | 26/62 | **53/62 = 85,5% [75–92]** |
| 27 Okt air acak | 31/35 = 88,6% [74–95] | 27/35 | **33/35 = 94,3% [81–98]** |
| 27 Okt kosong | 57/72 = 79,2% [68–87] | 69/72 | **67/72 = 93,1% [85–97]** |

Temuan: v3 mempertahankan kepekaan RH terhadap air (bahkan naik di ±1/6) SEKALIGUS hampir menghilangkan salah-tolak botol kosong (RH menolak 28/150 botol kosong 29 Okt sebagai berisi; v3 hanya 1). Trade-off RH (peka tapi banyak salah tolak) vs s35 (bersih tapi buta air sedikit) terselesaikan oleh data belt berlabel tingkat isi. Air penuh tetap paling sulit (~85%), konsisten dengan dugaan L-31 (model membaca batas air–udara). Catatan: angka RH di sini (potongan ulang offline, kotak ver9c) sedikit berbeda dari angka langsung L-31 (mis. ±1/6 60 vs 57) karena potongan & frame tidak identik.

**Merek v3** (`scripts/pipeline_merek/m09_belt29.py`, 256 px; sesi tutup-campur tidak dipakai): uji 29 Okt (334 botol) — m08/v2 dan v3 sama-sama 323/334; Coca-Cola dikenali **108/109 = 99,1% [95–100]**, non-Coca salah dapat bonus **10/225 = 4,4% [2–8]** (m07 lama: 17/225 = 7,6%); uji 27 Okt — v2 107/107, v3 (fold B27) 105/107. Kesimpulan: v2 sudah menggeneralisasi ke botol 29 Okt; v3 tidak memberi tambahan berarti. Final `merek_coca_v3_kandidat.pt` disimpan, rekomendasi tetap v2.

**Tutup v3** (`scripts/pipeline_cap/cap_v3_belt.py`; 397 foto cap_belt_v2 + frame belt 27 & 29 Okt, kotak leher dari usulan cap_belt_v2 yang DIAUDIT user, kelas dari label sesi; imgsz 960; final `cap_belt_v3_kandidat.pt`). Per botol (mayoritas 3 frame), bertutup benar / tanpa tutup benar:

| uji | mode | v2 aktif | v3 |
|---|---|---|---|
| 29 Okt (tutup ditukar ikut) | ROI (cara app) | 242/287, 72/75 | 226/287, 73/75 |
| 29 Okt | full frame | 252/287, 75/75 | 263/287, 75/75 |
| 27 Okt | ROI | 69/73, 32/34 | 67/73, 30/34 |
| 27 Okt | full frame | 72/73, 34/34 | 73/73, 33/34 |

Per frame, v3 lebih lemah di mode ROI tetapi lebih kuat di full frame -> metrik per frame saja tidak cukup; penentu = simulasi keputusan di bawah.

**Simulasi ulang keputusan mesin** (`scripts/alat/replay_keputusan.py`; frame diam tersimpan, profil nano, `app.tracker.decide`, model baru per hari = model yang TIDAK melihat hari itu; benar = keputusan sesuai harapan label kelompok):

| varian | 27 Okt (107) | 29 Okt (362) | gabungan | salah terima | salah tolak |
|---|---|---|---|---|---|
| aktif (ver9c + cap v2 + RH) | 101 | 348 | 449/469 = 95,7% | 12 | 8 |
| tutup baru saja | 101 | 354 | 455/469 | 6 | 8 |
| isi baru saja | 104 | 353 | 457/469 | 11 | 1 |
| **semua baru (cap v3 + isi v3)** | **104** | **357** | **461/469 = 98,3%** | **7** | **1** |

Isi v3 menghapus hampir semua salah tolak botol kosong (8 -> 1, sesuai prinsip "jangan rugikan customer karena noise"); tutup v3 memangkas botol bertutup yang lolos di 29 Okt (salah terima 10 -> 4), netral di 27 Okt. Sisa salah terima = botol bertutup dengan tutup tak terlihat (sebagian tutup ditukar warna). **Diaktifkan 30 Okt atas persetujuan user** (`rvm_kp/config.py MODEL_SETS`: `cap_belt_v3_kandidat` + `cairan_isi_v3_kandidat`; detektor ver9c & merek v2 tetap). Tidak perlu uji belt baru untuk keputusan ini: semua angka dari botol yang tidak dilatih.

**Logika lingkup uji isi (koreksi user 30 Okt):** botol berbaring di belt, jadi botol berisi ±1/2 atau penuh TANPA tutup tidak realistis (airnya tumpah) -> botol seperti itu selalu bertutup dan tertolak oleh pemeriksaan tutup. Satu-satunya botol berisi yang realistis datang tanpa tutup = sisa air sedikit (±1/6). Maka metrik isi yang paling menentukan keputusan = air ±1/6; lemahnya air penuh (~85%) berdampak kecil selama tutup terdeteksi. Uji tanpa tutup berikutnya cukup air sedikit.

**KOREKSI L-31 — bug kandidat bayangan (ditemukan 29 Okt malam saat memeriksa merek):** pekerja asinkron kandidat bayangan di `app/engine.py` membaca `t.box` (kotak track) SAAT DIPROSES, padahal track sudah bergerak/diperbarui -> potongan merek/isi/tutup kandidat diambil dari posisi yang salah. Akibatnya SEMUA angka kandidat langsung 27–29 Okt tidak sahih: merek v2 "76% / 44% / 17,5%", isi kandidat langsung, dan klaim "ver12 mengganggu tutup" di L-31. Perbaikan: kotak dibekukan saat masuk antrian (`_shadow_q.put((t, frames, plc_ok, b, tuple(t.box)))`). Jalur AKTIF tidak terdampak (bonus aktif terverifikasi: Coca 23/23 dapat bonus, non-Coca 2/48). Angka merek yang sahih = validasi offline di atas.

### L-33 Model tutup v4w: latih dengan warna tutup buatan (30 Okt) — TIDAK LULUS kriteria, tutup v3 tetap aktif
**Tujuan:** memutus jalan pintas "tutup sewarna cincin" (kegagalan tutup biru & tutup tukaran, L-32). **Data:** resep identik tutup v3 + satu salinan berwarna buatan per foto latih (`scripts/pipeline_cap/warna_tutup.py` versi 3: warna mengikuti KROMA tutup asli, kecerahan/bayangan dipertahankan, cincin tetap; tutup hitam/belang/arah ragu dilewati). Versi 1 (garis potong lurus) ditolak audit user (tutup separuh berwarna, salah ujung, belang). Versi 2 diaudit user per foto: 1.461 salinan, 221 dibuang -> 1.240 dipakai (510 tutup diwarnai, 356 tutup+cincin digeser, 374 tanpa tutup cincin digeser). 25 salinan yang sempat terbuat setelah audit (k-means tak deterministik) diarsipkan, tidak dipakai. Latih di Colab A100 (akun Azza, resep sama kecuali workers=8; `_cap_v4w/colab/`), dinilai di laptop.
**Kriteria lulus ditetapkan SEBELUM hasil (user):** di kedua hari uji (1) salah terima lebih sedikit, (2) tanpa tutup terbaca bertutup / salah tolak tidak bertambah, (3) tutup ASLI tidak lebih buruk dari v3.

| uji (hari tak dilatih) | v3 aktif | v4w |
|---|---|---|
| 29 Okt, per botol ROI: tutup asli / campuran / tanpa | 108/137, 118/150, 73/75 | **122/137, 131/150**, 73/75 |
| 29 Okt, simulasi keputusan (isi v3) | 357/362 (salah terima 5, salah tolak 0) | **359/362 (3, 0)** |
| 27 Okt, per botol ROI: tutup asli / campuran / tanpa | 67/71, 0/2, 30/34 | 67/71, 0/2, **27/34** |
| 27 Okt, simulasi keputusan | 104/107 (salah terima 2, salah tolak 1) | 103/107 (2, **2**) |

**Putusan: TIDAK LULUS** — 29 Okt lulus ketiganya (tutup asli justru membaik), tetapi 27 Okt gagal kriteria 1 (salah terima tetap 2) dan 2 (salah tolak 1 -> 2: botol kosong tanpa tutup S52B107 ditolak alasan "leher"). Pola: pada 4 botol tanpa tutup 27 Okt (D49B19, D52B102, D52B108, D52B142) v4w tidak memberi kotak leher sama sekali (keyakinan < 0,25) padahal v3 benar -> model jadi kurang yakin pada leher tanpa tutup; dalam aplikasi ini berujung tolak "leher". Model final v4w (dilatih juga di Colab) TIDAK dipakai. Catatan jujur: selisih 27 Okt kecil (1 botol keputusan, 3-4 botol per-botol) dan bisa kebetulan, tetapi aturan yang disepakati dipegang. Menguji varian lain pada dua hari yang sama berulang kali = menyetel ke data uji; penentu yang sah = sesi belt baru (botol & tutup belum pernah dipakai).

### L-34 Uji aplikasi utuh dengan model aktif baru + persiapan sesi belt berikutnya (30 Okt)
**Uji aplikasi utuh** (`scripts/alat/uji_app_putar.py`): `app.engine.Machine` asli (deteksi -> berhenti-lalu-pindai -> keputusan -> bonus merek -> catatan DB/CSV) dijalankan dengan KAMERA TIRUAN yang memutar frame diam 469 botol 27 & 29 Okt (talang & PLC dimatikan, data ke folder sementara; zona pemicu = zona periksa karena frame diam = posisi akhir botol). Hasil: 469/469 tercatat, 0 terlewat, 0 ganda, tanpa galat; berhenti per botol median 1,3 s (maks 2,5 s). Keputusan aplikasi **identik 469/469** dengan simulasi `replay_keputusan` memakai model yang sama (varian `aktif_final`) -> simulasi sah mewakili aplikasi. Catatan: benar 467/469 di uji ini BUKAN akurasi (model final aktif sudah melihat foto 27/29 Okt); akurasi sah = validasi per hari (L-32: 461/469).
**Persiapan sesi berikutnya:** kandidat bayangan kini bisa memakai model TUTUP lain (`app/config.py KANDIDAT['cap']`); set bayangan = sama dengan aktif kecuali tutup = `cap_belt_v4w_kandidat.pt` (final v4w: 28 epoch di Colab, 2 epoch terakhir dilanjutkan di laptop dari checkpoint yang sama, data identik 3.044 foto). Label kelompok baru: `tutup_asli` / `tutup_biru` / `tutup_tukar` (dihitung bertutup). Protokol: `docs/PROTOKOL-SESI-BELT-BERIKUTNYA.md`.

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

10. **(29 Okt malam)** Angka kandidat bayangan langsung 27–29 Okt (merek v2 76%/44%/17,5%; ver12 "mengganggu tutup"; isi kandidat) tidak sahih karena bug kotak bergerak di pekerja asinkron (L-32). Diganti validasi offline per hari.

## 8. Keterbatasan
- W16 hanya 5 botol, satu sesi, satu pencahayaan, foto statis; Fanta Grape belum ada; pemilahan data warisan belum divalidasi manusia.
- P15 punya confound; hanya dipakai sebagai uji, tidak pernah dilatih.
- Level isi `seperempat` tidak konsisten (sebagian hampir penuh, sebagian hampir kosong).
- Belum diuji pada belt bergerak dan di Jetson.
- **Botol berlabel penuh (rencana batasan, keputusan user 23 Okt — dipakai di TA HANYA bila setelah latih ulang dengan data tambahan masih gagal):** dibedakan dua jenis. (1) Label penuh dengan celah bening (leher/bahu/pantat; Minute Maid, Isoplus, Mizone): cairan masih terlihat lewat celah → diperbaiki dengan data berpasangan berisi/kosong botol yang sama di belt (Isoplus kosong 1/8 → 7/8 dengan data 22 Okt). (2) Label penuh TANPA celah (mis. Oi Ocha): isi secara fisik tidak terlihat kamera mana pun → rumusan batasan: "kamera tidak dapat menentukan isi botol berlabel penuh tanpa celah; kategori ini memerlukan sensor berat (load cell) atau ditolak". Opsi sistem hibrida (kamera utama, sensor berat hanya saat kamera ragu) = keputusan mitra. Botol berlabel penuh TIDAK ditolak semua karena banyak produk mitra (Minute Maid, Nutriboost, Fanta) termasuk jenis (1). Untuk demo, mitra memilih botol.
- Model tutup: tutup hitam sudah dilatih (C-10, v2) tetapi baru 2 produk bertutup hitam (Coke Zero, Powerade); Powerade masih 17/20 (tutup sport lebar). Semua uji = foto statis.
- Hasil "tidak membantu" untuk data luar berlaku untuk resep dan data yang dicoba, bukan bukti umum.
- **Ketahanan terhadap perubahan medan (concern user 24 Okt; langkah 1 DIUKUR 24 Okt, lihat hasil di bawah).** Semua model dilatih & divalidasi di SATU medan: belt capstone (hijau), satu kamera C922 di satu ketinggian, pencahayaan ruangan itu. Validasi fold per botol membuktikan generalisasi ke BOTOL baru, **tidak** ke belt/kamera/cahaya baru. Bukti dari proyek bahwa domain berpengaruh: data internet tidak membantu cairan & merek, `cap_v4` (studio/sintetis/web) lemah sebelum dilatih foto belt, model merek menghafal produk (skema P 158/261). Perkiraan kerentanan per bagian (dugaan awal dari cara kerja; diuji di langkah 1 di bawah): ukuran = pasti perlu kalibrasi ulang (piksel→ml bergantung tinggi/jarak kamera); cairan = rapuh (warna/transparansi, latar, ambang terkalibrasi); tutup = sedang (skala ROI bergantung skala piksel); merek = sedang (potongan botol, latar kurang berpengaruh, cahaya/sudut tetap); deteksi botol = paling tahan (data paling beragam); kode/app/PLC/alat = tidak bergantung medan. Yang membuat warisan tetap berguna: fine-tune butuh data sedikit (180 foto belt → tutup ±98%), siklus kumpul→label→latih→validasi sudah tersedia (app menyimpan frame diam, `lembar_cek`, skrip latih + CV berkelompok). Rekomendasi rekayasa: seragamkan kondisi (ruang pindai tertutup + lampu sendiri + latar tetap), seperti RVM komersial.
  **Hasil langkah 1 (24 Okt siang, `scripts/pipeline_ketahanan/k01_uji_ketahanan.py`, `datasets/4_eksperimen/ketahanan/laporan.txt`, contoh gangguan `contoh_gangguan.jpg`).** Model AKTIF, profil nano, tanpa latih ulang.
  - *A. Foto belt 22 Okt (64 foto, 5 botol, tidak ikut melatih cap_belt_v2/RH) + 16 gangguan buatan.* Deteksi botol 60–64/64 di semua gangguan. **Tutup tahan**: 64/64 asli; 58–64/64 dengan gangguan (terendah blur kuat 58, cahaya dingin 60, belt biru 60). Cairan RH: botol berwarna terdeteksi stabil 16–20/24 di semua gangguan; botol kosong benar hanya 14/32 bahkan di foto ASLI — bukan efek medan, melainkan kelemahan lama: botol PET berwarna (biru, ungu) yang kosong terbaca berisi (p 0,2–1,0), sehingga angka kosong di bawah gangguan (8–18/32) tidak bisa ditafsirkan sebagai efek medan. **Ukuran rapuh terhadap jarak kamera**: kamera lebih tinggi (isi frame 80%/60%) → kelas ukuran sama hanya 37/64 (kalibrasi wajib); blur kuat 53/64.
  - *B. Medan NYATA lain: foto studio Sep (350 foto, 28 botol kosong; kain cokelat + 2 lampu / alas putih + lampu kamar; kamera dekat, sudut atas & miring).* Deteksi botol 308/350 (alas putih lebih lemah 63/90). **Tutup**: benar 199/350 — bertutup dikenali 139/175, tetapi tanpa tutup hanya 60/175 (model condong menjawab 'tutup' di medan asing; 58 leher tak ditemukan). **Cairan gagal total**: kosong benar 9/350 (median p_cairan 0,997) — latar cokelat/putih yang terlihat menembus plastik bening dibaca sebagai cairan berwarna. Ukuran: semua 'besar'/'?' (kamera dekat) → tak bermakna tanpa kalibrasi.
  - *C. Merek (model fold CV skema F, potongan tak dilatih; patokan 222/261 coca, salah terima 11/290).* Tahan JPEG (212, 11) dan belt hitam (215, 11). Rapuh terhadap **noise sensor** (σ15: coca 48/261), **blur** (σ2: salah terima 49, σ5: 81), **warna cahaya** (pink 156/261, dingin 188/261), belt biru/abu (183/197 coca), gelap 50% (salah terima 28), resolusi 50% (salah terima 29).
  - **Kesimpulan jujur:** (1) urutan kerentanan terbukti sesuai dugaan: **cairan & ukuran paling bergantung medan**, tutup tahan gangguan ringan tetapi bias di medan yang benar-benar lain, merek peka kualitas kamera (noise/blur/warna cahaya), detektor botol paling tahan; (2) model ini **tidak bisa dipindah begitu saja** ke medan lain — minimal kalibrasi ukuran + latih ulang cairan dengan foto medan baru, dan fine-tune tutup; (3) tuas paling murah: **seragamkan kondisi** (ruang pindai tertutup, lampu tetap, latar tetap, kamera fokus); (4) perbaikan murah yang bisa diuji kemudian: augmentasi noise/blur/warna cahaya saat melatih merek & cairan.
  - *Keterbatasan uji:* gangguan buatan hanya mendekati kondisi nyata (kamera tinggi disimulasikan dengan memperkecil frame + tepi dipantulkan, menimbulkan artefak; ganti belt memakai topeng warna hijau yang juga mengenai label hijau); studio berbeda dalam banyak hal sekaligus (latar, jarak, sudut, cahaya) sehingga penyebab tidak dapat dipisahkan; patokan A hanya 5 botol; satu kali jalan tanpa selang kepercayaan.
  **Rencana (disetujui sebagai rencana 24 Okt, langkah 1 SELESAI 24 Okt (hasil di atas); langkah 2–3 belum):**
  1. *Ukur kerentanan dengan data yang ada:* model tutup belt diuji pada foto studio Sep (latar beda); cairan antar-sesi/cahaya beda (`1_foto_mentah/bg_lighting_tests`, sesi 16/20/22 Okt); uji ketahanan buatan (kecerahan, suhu warna, blur, skala, botol ditempel ke latar lain) → angka penurunan per model untuk bab keterbatasan.
  2. *Dokumen "batas operasi":* kondisi tempat model divalidasi (tinggi & sudut kamera, resolusi, warna/tekstur belt, lux pencahayaan) + foto setup; diukur saat sesi Senin.
  3. *Panduan "pindah medan" untuk penerus:* kalibrasi ukuran → kumpulkan ±100–200 foto per model lewat app → label via lembar cek / model-in-the-loop → fine-tune dari bobot sekarang → CV berkelompok + syarat lolos sebelum dipakai (jumlah data yang benar-benar cukup di medan baru belum diketahui).

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
