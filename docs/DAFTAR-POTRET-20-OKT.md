# Daftar Potret Data Baru (disusun 20 Okt 2026)

> **Lokasi file berubah 21 Okt 2026** (direktori dirapikan). Peta lokasi baru: `docs/00_STATUS_PROYEK.md`; daftar lengkap lama→baru: `datasets/PINDAH-LOG.csv`.


Dasar: hasil analisis ulang 20 Okt (catatan TA L-8b, L-10, L-11).
- Tiap botol berwarna tambahan terbukti membantu; batas atasnya belum diketahui.
- **Botol berisi dan kosong harus dipotret di sesi yang sama** (sesi berbeda dapat dibedakan model 100% dari belt saja).
- **Ambang perlu dikalibrasi** di botol yang tidak ikut latih.
- Data luar tidak membantu, jadi semua data baru = foto belt nyata.

Lingkup: cairan berwarna (air bening di luar lingkup). Semua botol PET diterima; merek = bonus poin (whitelist Coca-Cola, Fanta, Sprite); ukuran: **kecil ≤ 350 ml, tanggung 351–800 ml, besar > 800 ml** (diubah 20 Okt).

## 1. Daftar botol (target 15 botol berwarna + 2 negatif sulit)
Satu botol = satu botol fisik berbeda. Isi = minuman aslinya.

| no | kelompok warna | contoh produk | ukuran | peran | bonus merek |
|---|---|---|---|---|---|
| 1 | cola gelap | Coca-Cola | besar (1–1,5 L) | latih | ya |
| 2 | cola gelap | Coca-Cola Zero | kecil (≤390 ml) | **kalibrasi** | ya |
| 3 | cola gelap | BIG Cola / Pepsi | tanggung | **uji** | tidak (negatif merek sulit) |
| 4 | merah | Fanta Strawberry | tanggung (390 ml) | latih | ya |
| 5 | merah | BIG Strawberry | besar | latih | tidak (negatif merek sulit) |
| 6 | oranye | Fanta Orange | tanggung | latih | ya |
| 7 | oranye | Mirinda / jus jeruk | kecil atau besar | **uji** | tidak |
| 8 | ungu | Fanta Grape (bila ada) | tanggung | latih | ya |
| 9 | ungu atau merah tua | minuman anggur/berry lain | bebas | **kalibrasi** | tidak |
| 10 | hijau pucat | Sprite Nipis Mint | tanggung (390 ml) | latih | ya |
| 11 | hijau pucat | Sprite Nipis Mint ukuran lain (1 L) | besar | **uji** | ya |
| 12 | coklat atau teh | teh botol / Frestea | tanggung | latih | tidak |
| 13 | kuning | teh lemon / isotonik kuning | kecil | latih | tidak |
| 14 | putih keruh | Pocari / susu | tanggung | **kalibrasi** | tidak |
| 15 | bebas, warna yang belum ada | | besar | **uji** | |
| N1 | **negatif: botol hijau, cairan bening** | Sprite biasa / 7up (isi asli) | bebas | latih | ya (Sprite) |
| N2 | **negatif: botol berwarna, cairan bening** | air mineral botol biru/tinted | bebas | latih | tidak |

Pembagian: **latih 9 (+2 negatif), kalibrasi 3, uji 4**. Tiap kelompok warna yang diuji punya minimal satu botol di latih, kecuali no 15 (sengaja warna baru untuk melihat batas kemampuan). Ukuran: kecil 3, tanggung 8, besar 5, cukup untuk memperluas uji kategori ukuran. Kalau produk tidak tersedia, ganti dengan warna sama dari merek lain dan catat.

N1 dan N2 penting: negatif "botol hijau berisi cairan bening" adalah penyebab Sprite Nipis Mint jatuh di percobaan data luar. Dipotret di belt kita, mereka menjadi negatif sulit yang benar.

## 2. Persiapan (±15 menit, sebelum mulai)
1. ☐ **Cek framing:** letakkan botol 1,5 L di tengah belt dan di posisi 1/3 lebar frame. Kalau terpotong tepi frame, pakai `--center` (semua sisi di tengah) untuk botol besar.
2. ☐ Kamera 1080p: `venv/Scripts/python.exe scripts/alat/capture_water.py --list` (dari folder proyek).
3. ☐ Lampu dan posisi kamera **tidak diubah** selama sesi. Siapkan lap kering dan wadah buang.
4. ☐ Belt berhenti (server PLC tidak berjalan). Minuman jauh dari PLC dan motor.

## 3. Protokol per botol (alat `capture_water.py`, ±7 menit per botol)
```bash
venv/Scripts/python.exe scripts/alat/capture_water.py --cam <index>
```
Untuk botol besar:
```bash
venv/Scripts/python.exe scripts/alat/capture_water.py --cam <index> --center
```
1. Tekan `b`, ketik nama produk + ukuran (mis. `coca-cola 1500`), Enter, ketik warna cairan, Enter. **Sertakan ml di nama**: dipakai untuk kategori ukuran.
2. Ikuti urutan di layar: **penuh + tutup** → **setengah + tutup** → **kosong tanpa tutup** → **kosong + tutup**, 4 sisi per langkah (16 foto). Putar botol 90° tiap sisi dan ikuti tulisan arah leher.
3. Bilah bawah merah atau kuning: periksa tutup/botol, `z`, ulangi.
4. Tekan `e` (foto belt kosong) di awal, tengah, dan akhir sesi.
5. **Acak urutan botol**, jangan per kelompok warna. **Semua botol di satu sesi** bila memungkinkan; bila harus dua hari, setiap botol tetap lengkap (berisi dan kosong) dalam hari yang sama.
6. Lap belt kering di antara botol.

## 4. Negatif non-botol (±15 menit, alat terpisah)
```bash
venv/Scripts/python.exe scripts/alat/capture_free.py --cam <index> --name plastik-bening-remas
```
20–30 foto: kantong plastik bening diremas, plastik pembungkus, gelas plastik, botol kaca bening. Tekan `n` untuk mengganti nama benda. Disimpan di `datasets/1_foto_mentah/<tanggal>_nonbottle_v1/`.

## 5. Perkiraan waktu
15 botol × ±7 menit + 2 negatif × ±5 menit + persiapan dan non-botol ±30 menit ≈ **2,5 jam**. Bisa dipecah dua sesi asal tiap botol lengkap dalam satu sesi.

## 6. Setelah memotret (aku yang kerjakan)
Audit `qc_water_set.py` → kotak dari `ver8` → crop → latih di botol "latih" → **kalibrasi ambang** di botol "kalibrasi" → uji sekali di botol "uji" (tidak disentuh sebelumnya) → laporan per botol dan per warna, plus uji kategori ukuran.

## 7. Jangan
- Jangan memotret ulang 5 botol lama.
- Jangan memotret level seperempat.
- Jangan memotret botol berisi di satu hari dan kosongnya di hari lain.
- Jangan memakai botol "uji" untuk mengecek hasil sebelum model final selesai.
