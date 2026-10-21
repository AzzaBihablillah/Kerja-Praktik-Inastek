# Hasil Eksperimen Deteksi Cairan Berwarna (malam 17–18 Okt 2026)

Dikerjakan otomatis semalam. Semua angka berasal dari botol yang **tidak ikut dilatih** (validasi silang per botol), kecuali disebut lain.

## 0. Jawaban singkat
- **Pendekatan terbaik: dua tahap.** Detektor botol dilatih ulang (`ver8`, 2 kelas seperti `ver7`), lalu classifier kecil pada potongan botol menjawab berisi berwarna atau kosong.
- **Detektor 3 kelas (botol / botol berwarna / non-botol) kalah.** Ia melewatkan Fanta sama sekali (0/8), akurasinya lebih rendah, dan salahnya pada data lama dua kali lebih banyak.
- **Melatih ulang detektor perlu, apa pun pilihan pendekatannya.** `ver7` hanya mendeteksi **20 dari 40** foto botol berisi di set 16 Okt. Setelah foto belt ditambahkan ke data latih (`ver8`): **40/40**, tanpa penurunan pada data lama (99,8%).
- **Classifier memisahkan berisi dan kosong dengan sangat baik bila diukur tanpa ambang (AUC 0,97–1,00).** Hasil akhirnya ditentukan oleh pilihan ambang dan data negatif (tabel bagian 5).
- **Pada produk whitelist (Coca-Cola, Fanta, Sprite Nipis Mint), konfigurasi terbaik tidak membuat satu kesalahan pun pada data uji kita.** Konfigurasinya `ver8` + classifier "W16" pada ambang 0,5. Tetapi ini baru **5 botol** di set 16 Okt plus satu sesi pilot, jadi ini **bukti awal, belum bukti "mampu"**.

## 1. Lingkup dan target
- **Yang dideteksi:** botol berisi **cairan berwarna** (lingkup rapat mitra 16 Okt). Air bening di luar lingkup; load cell tetap penentu akhir.
- **Produk yang diterima:** Coca-Cola dan Fanta (Sprite Nipis Mint opsional).
- **Pertanyaan eksperimen:**
  1. Apakah detektor botol perlu dilatih ulang?
  2. Lebih baik detektor 3 kelas atau detektor + classifier terpisah?
  3. Data apa yang membantu dan apa yang merusak?
- **Aturan kerja:** keputusan tiap tahap diambil dari hasil tahap sebelumnya, bukan dari tebakan awal.

## 2. Data
| kode | isi | dipakai untuk |
|---|---|---|
| W16 | set 16 Okt, 5 botol (Sprite Nipis Mint, Coca-Cola Zero, Frestea, botol berlabel Fanta, Coca-Cola kecil). Penuh dan setengah = berwarna (40 foto), kosong (40 foto), seperempat (20 foto, ambigu: hanya dilaporkan) | latih dan uji per botol |
| C13 | belt 13 Okt, 30 botol **kosong** berbagai merek (180 foto) | negatif; latih dan uji per botol |
| P15 | pilot 15 Okt, **tidak pernah dilatih**: Fanta, Coca-Cola, dan Frestea berisi (32 foto), Coca-Cola kosong (9), botol diisi air putih (35) | uji lintas sesi |
| WEB | 540 foto Fanta penuh dari internet (Roboflow, CC BY 4.0) | percobaan tambahan |
| OLD | data warisan `bottle_detection` (2178 gambar). Pemilahan kotak botol oleh Claude: 143 berwarna, 723 bukan, 226 ragu (**belum diverifikasi manusia**) | latih detektor; percobaan tambahan classifier |

Semua kotak botol foto belt kuperiksa visual (8 lembar). Data warisan dibagi tetap: 85% latih dan 15% `old_test` untuk uji regresi.

## 3. Cara evaluasi
- **5 fold per botol.** Fold k menguji botol W16 ke-k (semua levelnya) plus 6 botol C13. Botol uji tidak pernah dilihat saat latih.
- **Tanpa memilih epoch terbaik:** dipakai bobot epoch terakhir.
- **P15 sebagai sesi lain.** Diuji oleh kelima model fold, dan dilaporkan sebagai jumlah gabungan (32 foto × 5 = 160).
- **Uji regresi** pada `old_test`. Tujuannya memastikan detektor baru tidak merusak deteksi botol dan non-botol lama.
- **Aturan keputusan meniru server:** kotak teratas dengan conf ≥ 0,35 dan kelasnya botol.

## 4. Urutan eksperimen dan keputusan
**T1. Baseline `ver7` di belt (deteksi botol saja).**
- Botol berisi W16: 20/40 (50%). Sprite Nipis Mint dan cola penuh paling sering luput.
- Botol kosong C13: 170/180 (94%).
- Sesi lain (P15): 150/160 (94%).

**Keputusan:** detektor harus dilatih ulang dengan foto botol berisi.

**T2. Baseline fitur warna (M0, regresi logistik).**
- Akurasi seimbang W16: 0,78.
- Gagal total pada Fanta (0/8) dan hampir total pada Sprite pucat (1/8).
- Menyebut 6 dari 35 botol air putih sebagai berwarna.

**Keputusan:** butuh model yang belajar dari gambar.

**T3. Classifier crop (YOLOv8n-cls, 224 px).** Hasil per varian ada di tabel bagian 5A. Yang kupelajari:
- **Crop "dilipat" (botol dipotong dua lalu ditumpuk) lebih baik daripada crop dipad (V2 vs V1).** Botol rebah sangat memanjang, dan melipatnya menggandakan resolusi badan botol.
- **Foto Fanta dari internet (V3) hasilnya campuran.** Naik di W16, turun di sesi lain (cola 6/10, teh 1/11).
- **Data warisan memperburuk hasil (V5, V6).** Sprite Nipis Mint jatuh ke 0–1 dari 8 dan teh ke 0/11. Data warisan penuh botol PET hijau dan Sprite bening berlabel "bukan", jadi model belajar bahwa botol hijau berarti tidak berwarna. Data ini kubuang dari classifier.
- **Negatif botol kosong merek lain (C13) membuat model konservatif (V2 vs V4).** Tanpa C13 (V4), recall sempurna, tapi 21 dari 180 botol kosong berlabel mencolok disebut berwarna (Isoplus biru, Ichitan, Natsbee, botol minyak, BIG hijau).
- **Semua varian kecuali M0 tidak pernah menyebut air putih sebagai berwarna** (0/35).

**T4. Detektor (5 fold, dari bobot `ver7`, 10 epoch).**
- `ver8` dan 3 kelas sama-sama mendeteksi **100%** botol di semua kelompok uji belt (`ver7`: 50–94%).
- Untuk menentukan cairan, 3 kelas kalah dari `ver8` + classifier (bagian 5B).

**Keputusan:** pakai dua tahap.

**T5. Sapuan ambang** pada pipeline dua tahap (bagian 5C), untuk memilih titik kerja.

## 5. Angka
### 5A. Classifier saja (potongan dari kotak yang sudah diperiksa), ambang 0,5
| varian | data latih | W16 berisi→berwarna | W16 kosong diterima | C13 kosong diterima | P15 berisi→berwarna | P15 air putih→berwarna | AUC W16 / P15 |
|---|---|---|---|---|---|---|---|
| M0 fitur warna | W16+C13 | 25/40 | 37/40 | 166/180 | 19/32 | 6/35 | – |
| V1 crop pad | W16+C13 | 26/40 | 40/40 | 178/180 | 21/32 | 0/35 | 0,998 / 0,972 |
| **V2 crop lipat** | W16+C13 | 33/40 | 40/40 | 178/180 | 23/32 | 0/35 | 1,000 / 1,000 |
| V3 | +Fanta internet | 38/40 | 40/40 | 176/180 | 18/32 | 0/35 | 1,000 / 0,999 |
| **V4** | **W16 saja** | **40/40** | 40/40 | 159/180 | **32/32** | 0/35 | 1,000 / 1,000 |
| V5 | +internet+warisan | 32/40 | 40/40 | 174/180 | 20/32 | 0/35 | 1,000 / 0,977 |
| V6 | +warisan | 28/40 | 40/40 | 172/180 | 21/32 | 0/35 | 1,000 / 0,929 |

**Khusus produk whitelist** (W16: Sprite Nipis Mint dan 2 Coca-Cola; P15: Coca-Cola dan Fanta):
- V4: berisi 24/24 dan 21/21, kosong atau air diterima 24/24 dan 19/19.
- V2: 23/24 dan 19/21, kosong diterima semua.

### 5B. Ujung-ke-ujung (detektor → keputusan), 5 fold
| kelompok | `ver7` saja: terdeteksi | 3 kelas: berisi→berwarna | `ver8` + V2 (0,5) | `ver8` + V4 (0,5) |
|---|---|---|---|---|
| W16 berisi (40) | 20 | 24 (b04 Fanta **0/8**, Sprite 3/8) | 31 | **40** |
| P15 berisi (160) | 150 | 114 (71%) | 108 (68%) | **154 (96%)** |
| W16 kosong diterima (40) | 40 | 40 | 40 | 40 |
| P15 kosong diterima (45) | 45 | 43 | 45 | 45 |
| P15 air putih diterima (175) | 175 | 170 | 175 | 175 |
| C13 kosong merek lain diterima (180) | 170 | 179 | 180 | 163 |

**Regresi data lama (`old_test`):**

| model | botol terdeteksi | non-botol disebut botol |
|---|---|---|
| `ver7` | 100% | 0% |
| `ver8` | 99,8% | 0,5% |
| 3 kelas | 99,2% | 1,1% |

`ver7` kemungkinan pernah melihat gambar ini saat latih, jadi angka 100%-nya terlalu optimis.

### 5C. Sapuan ambang (`ver8` + classifier)
| konfigurasi | W16 berisi | P15 berisi | W16, P15 kosong + air diterima | C13 kosong diterima |
|---|---|---|---|---|
| V2 @0,5 | 78% | 68% | 100% | 100% |
| V2 @0,2 | 90% | 76% | 100% | 98% |
| V4 @0,7 | 100% | 89% | 100% | 93% |
| **V4 @0,5** | **100%** | **96%** | **100%** | 91% |
| V4 @0,3 | 100% | 100% | 100% | 85% |

## 6. Rekomendasi
1. **Pipeline:** `ver8` (2 kelas) → potong botol (crop lipat) → classifier → keputusan.
2. **Classifier:**
   - **Pilihan utama: resep V4 dengan ambang 0,5.** Alasannya, botol kosong merek lain sudah ditolak oleh pemeriksaan merek, jadi salah alarm pada C13 tidak mengubah hasil akhir (tetap ditolak). Pada produk whitelist, V4 tidak membuat kesalahan di data uji.
   - **Kalau model merek belum bisa diandalkan,** pakai V2 dengan ambang 0,2. Lebih aman untuk botol kosong merek lain, tapi sekitar 10–25% botol berisi lolos ke load cell.
   - **Catatan:** model merek lama menerima BIG Cola dan sejenisnya sebagai `coca_cola`. Jadi pilihan antara V4 dan V2 bergantung pada perbaikan model merek.
3. **Jangan pakai detektor 3 kelas** untuk cairan.
4. **Load cell tetap penentu akhir.** Kamera berfungsi sebagai penyaring awal dan pemberi alasan penolakan.

## 7. Keterbatasan (wajib disebut di laporan)
- **Data uji kecil.** W16 hanya 5 botol berbeda, dan foto dalam satu botol tidak saling bebas. Selang kepercayaan per foto (misalnya V4 40/40, CI 91–100%) terlalu optimis. Satuan yang jujur adalah botol, dan jumlahnya baru 5 + beberapa botol P15.
- **Keragaman positif sedikit.** Warna yang ada hanya hijau pucat, cola, teh, kuning, dan merah. **Fanta Grape (ungu) belum pernah dilihat model.**
- **P15 punya confound.** Sesi pilot dan status tutup berbeda antar kelompok. Ia hanya dipakai sebagai uji, tidak pernah dilatih, jadi model tidak bisa memanfaatkan confound-nya. Tetap saja ini satu sesi.
- **Foto statis, satu pencahayaan.** Belum diuji di belt bergerak.
- **Level seperempat tidak konsisten** (sebagian hampir penuh, sebagian hampir kosong), jadi tidak dipakai melatih. Model menyebutnya berwarna 40–80% tergantung varian.
- **Pemilahan data warisan dilakukan Claude** dan belum diverifikasi. Data itu hanya dipakai untuk detektor dan percobaan V5/V6.
- **Botol b04 berlabel Fanta** disebut "fanta fake kuning" oleh kamu. Hasilnya tetap dilaporkan, tapi tidak dihitung di metrik whitelist.

## 8. File
- **Skrip:** `datasets/liquid_exp/s00`–`s10` dan `common.py`, berurutan. Hasil ada di `datasets/liquid_exp/out/`: `cls_summary.txt`, `cls_auc.txt`, `det_eval_*.txt`, `threshold_sweep.txt`, dan lembar periksa.
- **Model prototipe** (semua data belt berlabel, resep sama dengan CV; angka akurasinya = angka CV di atas):
  - `datasets/liquid_exp/out/final/det_ver8/weights/last.pt`: detektor botol `ver8`.
  - `datasets/liquid_exp/out/final/cls_W16/weights/last.pt`: classifier V4, ambang 0,5.
  - `datasets/liquid_exp/out/final/cls_W16_C13/weights/last.pt`: classifier V2, ambang 0,2.
- **Belum dipasang di server.** Sesuai lingkup Capstone 1 dan uji live yang belum dilakukan.
- **Pemilahan data warisan:** `datasets/liquid_exp/out/old_relabel_manual.csv`, dengan lembar `old_sheets/`.

## 9. Langkah berikutnya (saat kamu memotret 2 hari lagi)
1. **Utamakan botol berisi yang berbeda-beda:** Fanta Orange, Strawberry, dan **Grape**; Coca-Cola dalam beberapa ukuran; tambahan Sprite Nipis Mint bila ada.
2. **Potret juga botol kosong produk whitelist:** inilah negatif yang paling relevan.
3. **Botol berlabel mencolok merek lain** cukup sedikit sebagai uji salah alarm.
4. **Setelah itu jalankan ulang `s01`–`s09`.** Penambahan data ini yang akan menjawab "mampu atau tidak" dengan satuan botol yang cukup.
5. **Uji live** di belt bergerak dengan `ver8` dan classifier (butuh konfirmasimu soal keamanan belt).
