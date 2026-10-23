# Sesi Pembuktian di Belt (disusun 23 Okt 2026)

**Tujuan:** membuktikan di HARI LAIN dengan BENDA BARU bahwa perbaikan 22–23 Okt bertahan (bukan hanya cocok dengan sesi 22 Okt), dan mengumpulkan data kelemahan yang tersisa. Hasil sesi ini = angka uji yang sah untuk TA.

Model yang diuji (belum aktif sampai sesi ini lolos): detektor `botol_ver10` (CV per benda 22 Okt: non-botol 0/79, botol 154/154), cairan `cairan_berisi_kandidat` (definisi baru: air bening juga ditolak) dibanding RH, ukuran berbasis panjang, dan **merek** `merek_coca_kandidat` (coca_cola vs non_coca_cola, resep A; tambah 24 Okt — sesi ini = uji terpisah pertama model merek dengan botol baru di hari lain).

## Cara (semua lewat aplikasi kiosk, mode berhenti-lalu-pindai, frame diam otomatis tersimpan)
1. Pastikan area belt aman, lalu minta Claude menjalankan aplikasi:
   `venv/Scripts/python.exe -m app --backend pytorch --cam 1 --plc --jeda 1.0 --kandidat`
   Layar & poin = model AKTIF (ver9c, cap_belt_v2, cairan_RH, ukuran luas). `--kandidat` = model kandidat (ver10, cairan_berisi, ukuran panjang, merek coca_cola)
   ikut menilai FRAME DIAM YANG SAMA dan dicatat ke `SCAN_KANDIDAT.csv` → satu sesi langsung jadi perbandingan, tidak perlu mengulang botol.
   (Diuji offline 23 Okt dengan kamera & PLC tiruan; PLC kini sambung ulang sekali bila putus.)
2. Masukkan benda satu per satu sesuai daftar; **catat urutan DAN nama produk di HP** (mis. "1 kresek hitam, 2 Aqua kosong, 3 Sprite kosong, ..."). Nama produk dipakai menilai model merek pada SEMUA botol sesi, bukan hanya bagian G.
3. Selesai → Claude membuat `LEMBAR-CEK.jpg` (`scripts/alat/lembar_cek.py`) → kamu sebut nomor yang salah (±5 menit).
4. Lembar cek (`scripts/alat/lembar_cek.py`) menampilkan keputusan model aktif DAN kandidat per botol, plus baris MEREK (Coca-Cola Co. / bukan, p); label kamu dipakai untuk menilai keduanya.

## Daftar (±60 benda/pindaian, ±45 menit)
| bagian | isi | jumlah | catatan |
|---|---|---|---|
| A. Non-botol BARU | kresek & plastik yang BELUM pernah dipakai (warna/ketebalan lain), gelas cup, wadah makanan, kaleng, botol KACA, kotak susu, tisu, tangan | 20 pindaian | jangan pakai kantong yang sama dengan 22 Okt |
| B. Uji iseng | foto botol di HP (HP lain bila ada), foto botol dicetak di kertas, busa/kardus bergambar botol | 10 | belum pernah diuji: kertas & busa |
| C. Botol kosong tanpa tutup | 10 botol berbeda, termasuk **berlabel penuh** (Minute Maid, Mizone, Nutriboost, Teh Pucuk, Isoplus) | 15 | kelemahan terbesar model cairan |
| D. Botol berisi berwarna tanpa tutup | warna **pucat** (oren encer, teh, kuning muda) dan pekat; isi ulang botol bekas boleh | 10 | Fanta oren pucat kemarin lolos |
| E. Ukuran | botol ramping 400 ml, 1–1,5 L, galon kecil, 250–330 ml (tulis ml-nya) | 5 | uji aturan ukuran panjang vs luas |
| F. **Air bening (WAJIB, keputusan 23 Okt: ditolak)** | botol berisi air bening **TANPA tutup**, merek lain dari sebelumnya (Aqua, Club, Vit, Le Minerale besar, botol soda isi air), penuh & setengah | 10 | uji kandidat `cairan_berisi`; lingkup baru perlu dikabarkan ke mitra |

| G. **Merek: botol fisik produk Coca-Cola Company (tambahan 23 Okt)** | botol BEDA (bukan botol yang sama difoto ulang) per produk, kosong tanpa tutup; label menghadap atas, sebagian juga label menyamping | ±25 botol | lihat tabel di bawah |

### G. Target botol fisik merek (hasil uji model merek 23 Okt)
Model merek gagal pada produk yang botol fisiknya sedikit (data internet tidak membantu, CATATAN §5). Target ≥5 botol fisik berbeda per produk:

| produk | botol fisik sekarang | tambah | prioritas |
|---|---|---|---|
| Minute Maid Nutriboost | 1 (model 0/16) | 4 | **utama** |
| Powerade | 2 (model 8–11/20) | 3 | **utama** |
| Sprite (termasuk Nipis Mint) | 3 | 2–3 | **utama** |
| Frestea | 3 | 2 | **utama** |
| Coca-Cola biasa | 2 | 3 | sedang |
| Minute Maid (Pulpy, dll.) | 2 | 3 | sedang |
| Fanta / Coca-Cola Zero | 4 | 1 (bila ada) | rendah |
| produk Coca-Cola Company lain yang ada di toko (Ades, Fuze Tea, Schweppes, dll.) | 0 | 1–2 per produk | bonus |

Tulis nama produk di catatan HP. Pembanding non-Coca-Cola yang mirip dan sering salah diterima: **Sunbay, Pristine, Big Cola/Big Nipis Madu, Mizone, Superindo**. Masing-masing 1–2 botol beda bila ada.

### H. Batas operasi (±5 menit, rencana ketahanan medan 24 Okt)
Catat kondisi medan tempat model divalidasi (untuk TA & penerus):
- tinggi lensa kamera dari permukaan belt (cm, meteran) dan kemiringan kamera (lurus ke bawah / miring);
- lebar belt yang terlihat di frame (cm) — sekalian cek kalibrasi ukuran;
- pencahayaan: lampu apa saja yang menyala, ada cahaya jendela atau tidak; lux di atas belt bila ada aplikasi light meter di HP;
- foto setup dari samping (kamera + belt + lampu) dan satu foto belt kosong dari kamera.

## Setelah sesi
Claude: hitung per bagian (diterima/ditolak benar), bandingkan dengan model lama, catat di `CATATAN-PERCOBAAN-TA.md`, lalu (bila lolos) usulkan mengaktifkan `ver10` / gabungan cairan / ukuran panjang. Frame berlabel juga jadi data latih berikutnya.
