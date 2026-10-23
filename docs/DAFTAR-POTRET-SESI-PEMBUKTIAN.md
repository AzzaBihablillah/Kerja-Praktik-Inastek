# Sesi Pembuktian di Belt (disusun 23 Okt 2026)

**Tujuan:** membuktikan di HARI LAIN dengan BENDA BARU bahwa perbaikan 22–23 Okt bertahan (bukan hanya cocok dengan sesi 22 Okt), dan mengumpulkan data kelemahan yang tersisa. Hasil sesi ini = angka uji yang sah untuk TA.

Model yang diuji (belum aktif sampai sesi ini lolos): detektor `botol_ver10` (CV per benda 22 Okt: non-botol 0/79, botol 154/154), cairan `cairan_berisi_kandidat` (definisi baru: air bening juga ditolak) dibanding RH, ukuran berbasis panjang.

## Cara (semua lewat aplikasi kiosk, mode berhenti-lalu-pindai, frame diam otomatis tersimpan)
1. Pastikan area belt aman, lalu minta Claude menjalankan aplikasi dengan model kandidat (`--plc`, jeda 1,0 s).
2. Masukkan benda satu per satu sesuai daftar; **catat urutan di HP** (mis. "1 kresek hitam, 2 Aqua kosong, ...").
3. Selesai → Claude membuat `LEMBAR-CEK.jpg` (`scripts/alat/lembar_cek.py`) → kamu sebut nomor yang salah (±5 menit).
4. Kalau sempat, ulangi sebagian benda dengan model aktif lama (`--terus` tidak perlu) supaya ada pembanding langsung.

## Daftar (±60 benda/pindaian, ±45 menit)
| bagian | isi | jumlah | catatan |
|---|---|---|---|
| A. Non-botol BARU | kresek & plastik yang BELUM pernah dipakai (warna/ketebalan lain), gelas cup, wadah makanan, kaleng, botol KACA, kotak susu, tisu, tangan | 20 pindaian | jangan pakai kantong yang sama dengan 22 Okt |
| B. Uji iseng | foto botol di HP (HP lain bila ada), foto botol dicetak di kertas, busa/kardus bergambar botol | 10 | belum pernah diuji: kertas & busa |
| C. Botol kosong tanpa tutup | 10 botol berbeda, termasuk **berlabel penuh** (Minute Maid, Mizone, Nutriboost, Teh Pucuk, Isoplus) | 15 | kelemahan terbesar model cairan |
| D. Botol berisi berwarna tanpa tutup | warna **pucat** (oren encer, teh, kuning muda) dan pekat; isi ulang botol bekas boleh | 10 | Fanta oren pucat kemarin lolos |
| E. Ukuran | botol ramping 400 ml, 1–1,5 L, galon kecil, 250–330 ml (tulis ml-nya) | 5 | uji aturan ukuran panjang vs luas |
| F. **Air bening (WAJIB, keputusan 23 Okt: ditolak)** | botol berisi air bening **TANPA tutup**, merek lain dari sebelumnya (Aqua, Club, Vit, Le Minerale besar, botol soda isi air), penuh & setengah | 10 | uji kandidat `cairan_berisi`; lingkup baru perlu dikabarkan ke mitra |

## Setelah sesi
Claude: hitung per bagian (diterima/ditolak benar), bandingkan dengan model lama, catat di `CATATAN-PERCOBAAN-TA.md`, lalu (bila lolos) usulkan mengaktifkan `ver10` / gabungan cairan / ukuran panjang. Frame berlabel juga jadi data latih berikutnya.
