# Peta Dataset dan Rencana Pemerataan Proporsi (19 Okt 2026)

> **Lokasi file berubah 21 Okt 2026** (direktori dirapikan). Peta lokasi baru: `docs/00_STATUS_PROYEK.md`; daftar lengkap lama→baru: `datasets/PINDAH-LOG.csv`.


Dicatat untuk laporan. Tidak ada file yang dipindah atau dihapus. Registri mesin-terbaca: `datasets/4_eksperimen/cairan/dataset_registry.csv` (dibuat `s20_registry.py`). Semua percobaan data luar dan preprocessing sudah selesai (19 Okt malam); hasilnya ada di bagian 3 dan 5.

## 1. Temuan utama tentang "proporsi jomplang"
Rasio kelas per kotak hampir semua dataset YOLO kita wajar (1,0–1,9). **Yang jomplang adalah domain dan asal data**, bukan jumlah per kelas:

| model | masalah proporsi | angka |
|---|---|---|
| **bottle** (`bottle_detection`, 2178 gambar) | botol berisi cairan berwarna sangat sedikit; foto belt 1080p hampir tidak ada | kotak `bottle` 1092 vs `non_bottle` 1607. Dari kotak `bottle` (pilah visual Claude, belum diverifikasi): **berwarna 143 (13%)**, kosong atau bening 723 (66%), ragu 226 (21%) |
| **cap** (`cap_v4`, 3204 gambar) | didominasi sintetis; data belt tidak masuk | train 3026 = **sintetis 2104 (70%)**, web 656 (22%), studio asli 266 (9%), belt 0. Val 99 dan test 79 sepenuhnya real |
| **merek** (`brand2`, 3308 gambar) | label `unknown` memuat produk Coca-Cola; Sprite Nipis Mint tidak ada | aqua 1118, le_mineral 1444, unknown 1151, coca_cola 763 (kotak). Contoh 48 kotak `unknown` memuat 8 produk Coca-Cola Company (Fanta, Sprite, 7up) → **perkiraan ±12% (±140 kotak) salah label** menurut definisi mitra |
| **cairan** (belt) | negatif jauh lebih banyak dan satu domain | berwarna 40 (W16) vs negatif 220 (W16 kosong 40 + C13 180) = **1 : 5,5**; positif hanya dari 5 botol |
| **non_bottle** | kurang negatif sulit | belum ada plastik bening diremas, film, kantong, wadah makanan |

## 2. Inventaris
| kode | gambar | kelas (kotak) | domain | catatan |
|---|---|---|---|---|
| bottle_detection (warisan) | 2178 | bottle 1092, non_bottle 1607 | web + rig belt lama | data ver7; di luar repo (Downloads) |
| brand2 (warisan) | 3308 | aqua 1118, coca_cola 763, le_mineral 1444, unknown 1151 | web + rig lama | data brand ver5 |
| cap_v4 | 3204 | cap 1882, no_cap 1645 | sintetis 70% + web + studio | model cap v4 |
| cap_belt_all | 180 | cap 90, no_cap 90 | **belt asli** 1080p | model final cap_belt_final |
| cap_shoot_v1 / v2 | 260 / 90 | 130:130 / 45:45 | studio | |
| synth_cap_v1/v2/v3 | 5731 / 6511 / 6491 | ±1,0–1,4 | sintetis | v2 gagal |
| cap_exp_* (4 set) | 2481–5462 | cap:no_cap sampai 3,4:1 | komposit | eksperimen |
| Roboflow bottle cap detection 2 | 450 | Loose_cap 184, No_Caps 153, With_cap 110 | Roboflow | CC BY 4.0 |
| Roboflow cap_defect v2 | 580 | 5 kelas cacat | Roboflow | di luar lingkup |
| water_internet_fanta | 540 | Fanta hijau/oranye/merah 180 masing-masing | Roboflow, diaugmentasi 3× | split per foto sumber |
| Water Detection (Roboflow) | 900 | Green, Orange, Pepsi, Red, Sprite 180 | Roboflow, diaugmentasi 3× | |
| RF100 soda-bottles | 2249 | coca-cola 18505, sprite 18744, fanta 17419 | foto lemari pendingin | CC BY 4.0; semua penuh |
| primary/2026-10-13_conveyor_v1 | 180 | 30 botol kosong, cap/no_cap | **belt asli** | |
| primary/2026-10-16_water_v1 | 100 | 5 botol, 4 level | **belt asli** | nama diperbaiki 18 Okt |
| pilot 15 Okt (`foto_botol_update`) | 76 | Fanta/Coca-Cola/Frestea + air putih | belt, 640×360, sesi lain | hanya untuk uji |
| primary shoot v1/v2 | 260 / 90 | 13 / 15 botol | studio 108 MP | |

## 3. Pelajaran dari percobaan data luar (semua dengan uji per botol)
Metrik: botol berisi dari set 16 Okt yang terbaca "berwarna" (dari 40) dan botol kosong merek lain (C13) yang diterima (dari 180). Pembanding V2 = 33/40 dan 178/180.

| tambahan pada V2 | berwarna | kosong C13 | sesi pilot | kesimpulan |
|---|---|---|---|---|
| foto Fanta internet | 38 | 176 | turun 23→18 | campuran |
| data warisan (kotak dipilah) | 28–32 | 172–174 | turun | **merusak** (PET hijau berlabel "bukan") |
| RF100 (foto lemari) | 30 | 180 | turun 23→20 | tidak membantu |
| kilau sintetis | 24 | 180 | turun 23→20 | tidak membantu; V4+kilau tidak berubah |
| RF100 + kilau | 29 | 180 | turun 23→21 | tidak membantu; Fanta berlabel b04 jatuh ke 0/8 |
| **Open Food Facts** (pos + neg dari sumber yang sama; 696 berwarna, 881 bening) | 27 (V2) / 32 (V4) | 172 / 162 | turun 23→21 / 32→22 | **merusak**: AUC botol berisi turun 1,000→0,912 (V2) dan Sprite Nipis Mint 7/8→0/8 |

**Pola:** positif dari domain lain membuat model memakai tanda domain ("bukan belt" = berwarna) karena semua negatifnya dari belt. Karena itu: (a) setiap domain tambahan harus membawa **positif dan negatif sekaligus**; (b) bobot per domain harus dibatasi; (c) uji hanya di belt dan per botol.

## 4. Rencana kombinasi latih per model
Aturan umum: satuan sampel = **botol** (bukan foto); tiap botol diberi bobot sama; domain non-belt dibatasi maksimal sama dengan jumlah belt; val/test tidak pernah memakai sintetis atau data luar.

| model | rencana | status |
|---|---|---|
| **bottle ver8** | ver7 + belt (W16 + C13) + negatif TACO; tambah botol berisi sebagai `bottle` | belt sudah dicoba (100% deteksi, regresi 99,8%); TACO sedang diunduh |
| **cairan (classifier)** | **belt saja**: resep V4 (atau V2) + crop lipat + P1 opsional. Semua data luar yang dicoba (Fanta web, warisan, RF100, kilau, Open Food Facts) memperburuk hasil, termasuk yang membawa negatif dari sumber sama | selesai diuji; kesimpulan tetap |
| **cap** | cap_belt_final (belt 180) sebagai inti; kurangi sintetis ke maksimal 1:1 terhadap real; tambah belt baru | belum dicoba |
| **merek** | definisi baru `produk_mitra` vs `lain`; relabel `unknown` yang memuat Fanta/Sprite; tambah W16 dan foto kemasan Open Food Facts; BIG Cola dan sejenisnya sebagai negatif sulit | belum dikerjakan |

## 5. Preprocessing yang dicoba
| varian | isi | hasil |
|---|---|---|
| crop pad vs crop lipat | botol dipotong dua dan ditumpuk (resolusi 2×) | **lipat lebih baik** (V1 26/40 → V2 33/40) |
| P1 | normalisasi warna belt | **terbaik, modest**: produk whitelist 24/24, botol kosong merek lain 180/180, sesi pilot 26/32 (V2: 23/32); total botol berisi 31/40 (V2: 33/40), b04 turun 7→4 |
| P2 | CLAHE | whitelist 24/24 dan 180/180, tapi sesi pilot 17/32 (lebih buruk dari V2); Frestea 0/8 |
| P3 | supresi latar (piksel dekat warna belt → abu) untuk menghilangkan tanda domain | **tidak membantu**: resep V2 28/40 dan sesi pilot 21/32; resep V4 botol kosong merek lain diterima turun 159→152 |
| kilau sintetis | elips dan garis putih di-blur | tidak membantu |

## 6. Dataset tambahan untuk non_bottle dan negatif sulit
| sumber | isi | lisensi | keputusan |
|---|---|---|---|
| **TACO** (`_dl_taco`) | sampah di alam; film plastik 451, pembungkus 260, kantong, gelas, kaleng, karton, kaca. Kupakai 812 gambar (1177 kotak) tanpa botol PET berlabel | CC BY 4.0 (wajib atribusi) | **diambil**, sedang diunduh; dimasukkan sebagai `non_bottle` |
| Open Food Facts | foto depan kemasan, 5779 foto terunduh, 1577 lolos saring (696 berwarna, 881 bening) | CC BY-SA | **diuji, merusak classifier** (bagian 3); arsip saja |
| Roboflow "Recycle Project", "plastic-bag", "plastic_wastage" | kantong dan pembungkus plastik | CC BY 4.0 | kandidat, belum diambil |
| SortWaste | sampah industri, plastik lunak | belum dicek | kandidat; periksa lisensi |
| ZeroWaste | belt daur ulang, plastik lunak, ±30 GB, berkata sandi | | **dilewati** (terlalu besar, sandi) |
| Bottled Goods (Hugging Face) | botol untuk ritel | CC BY-NC-SA 4.0 | tidak dipakai (non-komersial; domain tidak cocok) |
| Roboflow Detection_of_Bottle | 9,8 ribu foto stok botol air | CC BY 4.0 | **ditolak** (domain dan label tidak cocok) |

**Hasil pemeriksaan TACO (19 Okt, 812 gambar, 1177 kotak):** semuanya foto sampah di luar ruangan (pasir, rumput, daun, jalan) dengan objek kecil di latar luas; domain jauh dari belt dan kotaknya kecil. Sebagai gambar penuh nilainya rendah untuk detektor belt. **Yang dipakai: potongan objek** (film plastik 278, pembungkus 124, kantong 62, gelas 80+51, kaleng 125, karton, wadah) untuk **ditempel ke foto belt** sebagai `non_bottle`. Risiko: tepi tempelan menjadi tanda palsu `non_bottle`; mitigasi dengan menempelkan juga potongan botol asli sebagai `bottle`, dan menguji hanya di foto belt asli.

Kekurangan yang tersisa pada non_bottle: **plastik bening diremas di belt** (belum ada di sumber mana pun yang kutemukan). Rekomendasi: potret sendiri 20–30 contoh di belt (kantong bening, plastik pembungkus, gelas plastik, botol kaca bening).

## 7. Langkah berikutnya
1. (selesai 19 Okt) Percobaan preprocessing dan data luar. Kesimpulan: tambahan harus berupa **foto belt nyata**, bukan data luar.
2. Gabungkan negatif TACO dan retrain ver8; ukur regresi di `old_test` dan hasil di belt.
3. Relabel `unknown` di `brand2` (kandidat: lembar kotak) setelah definisi `produk_mitra` ditetapkan.
4. Susun dataset latih akhir per model (daftar file, bukan memindah file) dengan proporsi di bagian 4.
5. Potret negatif nyata: plastik bening diremas di belt.

## 8. Kurva belajar CNN: berapa botol berwarna nyata yang diperlukan (20 Okt)

> **Catatan:** kurva di bawah TERKONFOUND ketidakseimbangan kelas dan diganti oleh hasil yang benar di `CATATAN-PERCOBAAN-TA.md` bagian L-8b (AUC naik bertahap 0,80 → 0,92 → 0,98 → 0,998; masalah utama = kalibrasi ambang; klaim "±15 botol" tidak didukung).
Cara: untuk tiap botol uji (5 botol W16), latih dengan k botol berwarna lain (k = 1–4, subset acak) + botol kosong C13 di luar fold-nya; 15 epoch (setengah dari resep biasa, jadi **cenderung pesimis**); 35 latihan. Uji pada botol uji, botol kosong merek lain, dan sesi pilot P15.

| k botol berwarna latih | recall botol uji (rata-rata [rentang]) | recall sesi pilot | botol kosong diterima |
|---|---|---|---|
| 1 | 0,25 [0,00–1,00] | 0,14 | 100% |
| 2 | 0,30 [0,00–1,00] | 0,19 | 100% |
| 3 | 0,71 [0,00–1,00] | 0,52 | 100% |
| 4 | 0,70 [0,00–1,00] | 0,60 | 99% |

Per botol uji, k = 1 / 2 / 3 / 4: b01 (Sprite NM) 0,00 / 0,25 / 1,00 / 1,00; b02 (cola hitam) 0,00 / 0,50 / 1,00 / 1,00; b05 (cola kecil) 1,00 / 0,50 / 1,00 / 1,00; **b03 (Frestea) 0,25 / 0,25 / 0,56 / 0,50; b04 (Fanta kuning) selalu 0,00**.

**Bacaan:**
- Lonjakan terjadi pada **3 botol** (dari ~0,3 ke ~0,7). Dengan 1–2 botol model belum bisa menggeneralisasi.
- **Keragaman warna lebih penting dari jumlah:** botol yang punya "kerabat warna" di data latih (cola hitam ↔ cola kecil, hijau pucat) cepat sampai 100%. Botol dengan warna tanpa kerabat (Fanta kuning, Frestea coklat) tidak pernah dikenali, berapa pun k.
- **Perkiraan kebutuhan:** ±3 botol per kelompok warna. Dengan 5 kelompok yang relevan (cola gelap, Fanta merah, Fanta oranye, Fanta anggur, Sprite Nipis Mint) → **±15 botol**, sesuai perkiraan awal 12–16.
- **Batasan:** 15 epoch (b04 tetap 0,00 padahal pada resep 30 epoch ia 7/8), 5 botol uji, 5–10 latihan per k, sehingga rentangnya lebar. Ini arah, bukan angka pasti.

## 9. Model final ke-2 (semua data belt nyata ikut dilatih)
`datasets/4_eksperimen/cairan/final2/`: `cls2_V4_all` (72 positif : 112 negatif; ambang 0,5), `cls2_V2_all` (72 : 292), `det_ver8b` (ver8 + sesi pilot + 28 foto tambahan). Data: set 16 Okt, belt 13 Okt (V2), sesi pilot 15 Okt, 28 foto tambahan (Frestea dan Fanta kosong, dicek visual). Tidak ada uji bersih; akurasi = hasil validasi silang di atas. Belum dipasang di server.
