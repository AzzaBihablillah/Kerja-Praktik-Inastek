# Rencana yang Harus Kamu Lakukan (15–27 Okt 2026)

Asumsi tenggat: **27 Okt** (2 minggu sejak 13 Okt). Koreksi kalau berbeda.
Tanda: ☐ belum, ✔ sudah. Hal yang kukerjakan sendiri ada di bagian C; kamu cukup menjawab "ya".

## UPDATE 16 Okt — hasil rapat mitra: lingkup deteksi cairan = HANYA CAIRAN BERWARNA

- Disepakati bersama mitra: sistem hanya untuk **cairan berwarna**. **Air bening secara eksplisit di luar lingkup** (belum bisa dideteksi; load cell tetap penentu). Ini batasan resmi, bukan kegagalan, dan ditulis begitu di laporan.
- Pertanyaan A3 (target air dan kriteria) sebagian terjawab. Kriteria "mampu" kini hanya untuk cairan berwarna. Usulan (kamu dan mitra yang menentukan): akurasi ≥95% pada botol yang tidak ikut latih, batas bawah interval kepercayaan ≥90%, dilaporkan **per warna** dan **per tingkat isi** (penuh, sepertiga, sisa).
- Protokol E di bawah **diganti** oleh protokol set cairan berwarna ini. Alat `datasets/capture_water.py` sudah diperbarui (warna cairan, level kosong/sisa/sepertiga/penuh, garis posisi, tombol `e` referensi belt, `--list` untuk memilih kamera).

### E2. Protokol set cairan BERWARNA
**Kamera dulu:** colokkan C922 ke laptop, lalu `../venv/Scripts/python.exe capture_water.py --list` (dari folder `datasets`). Pilih index yang menampilkan belt dan harus berukuran **1920x1080**. (Per 16 Okt, C922 belum terdeteksi Windows: index 0 = webcam laptop, 1 = DroidCam, 2 = OBS.) Lalu jalankan `capture_water.py --cam <index>`; alat menolak mulai kalau bukan 1080p.

**Bahan (rekomendasi 24 botol, minimal 16):**
- **Berwarna, 18 botol:** minimal 6 warna/jenis cairan × 3 botol berbeda (merek/ukuran berbeda). Contoh: cola hitam; merah (Fanta stroberi); oranye (Fanta jeruk/jus); kuning (Frestea/teh lemon); coklat (teh/kopi susu); putih keruh (Pocari/susu); hijau (teh hijau). Makin beragam warna makin baik.
- **Air bening, 6 botol (hanya uji batas, dilaporkan terpisah):** untuk menunjukkan perilaku sistem pada air bening.
- **Negatif sulit, 3–4 botol:** botol KOSONG berplastik berwarna atau berlabel mencolok (misalnya Sprite hijau, botol kecoklatan).

**Per botol berwarna** (urut menurun, 2 foto per langkah: leher kanan di 1/3 lalu leher kiri di 2/3; tutup mengikuti kondisi nyata):
1. penuh + tutup ON (tekan `4`, tutup ON)
2. buka tutup, tuang hingga ±⅓ (batas tidak tumpah saat rebah) → `3` + tutup OFF
3. tuang hingga sisa ≤50 ml (1–2 jari) → `2` + tutup OFF
4. kosongkan → `1` + tutup OFF
5. pasang tutup → `1` + tutup ON
Total 10 foto/botol. Label: berwarna = level 2/3/4, negatif = level 1.
Botol air bening: langkah 1, 2, 4, 5 (8 foto). Negatif sulit: langkah 4 dan 5 (4 foto).
Perkiraan total: 18×10 + 6×8 + 4×4 ≈ 244 foto, ±3 jam termasuk menuang.

**Aturan konsistensi (penting, ini yang merusak pilot pertama):**
- Satu sesi, kamera dan lampu **tidak diubah**, fokus/exposure **tidak dikunci**, 1080p.
- Tekan `e` (foto belt kosong) di awal, tengah, dan akhir sesi untuk memantau drift cahaya.
- Urutan botol **diacak** (jangan semua cola dulu). Jangan ada tangan/benda lain di frame.
- Arah leher: **kedua arah** (mesin bisa menerima ke kiri, kecuali manual menyuruh pengguna menghadapkan leher). Foto ke-1 per kondisi = leher KANAN di posisi 1/3, foto ke-2 = leher KIRI di posisi 2/3; alat menampilkan tulisan "LEHER KE KANAN/KIRI" dan menamai file `neckr`/`neckl`.
- Botol sejajar belt di lembah V; lap belt kering di antara botol (minuman manis lengket mengubah tampilan).

**Keselamatan:** belt dalam keadaan berhenti (server PLC tidak berjalan; belt tidak akan kunyalakan), jauhkan minuman dari PLC dan motor, siapkan lap.

## A. Keputusan dan pertanyaan ke mitra (bisa lewat HP, ±15 menit)

1. ☐ **Rapih direktori:** setuju atau tidak (rencana ada di chat; tidak ada yang dihapus, hanya dipindah ke `datasets/_arsip/`). *Saranku: setuju.*
2. ☐ **Setelan kamera deploy:** 1920×1080 atau 640×480? Model dilatih di 16:9, dan model final tetap 98,3% di imgsz 640 maupun 960. *Saranku: kunci satu setelan, utamakan 1080p (sama dengan data latih), lalu uji live untuk memastikan.*
3. ☐ **Tanyakan ke mitra (PT Inastek):**
   - Botol **berring tanpa tutup** valid atau tidak? (Saat ini dianggap valid, ring ≠ tutup.)
   - **Target deteksi air:** apa tepatnya yang harus dideteksi (kosong vs berisi? keruh saja?) dan fokus Coca-Cola saja atau merek lain juga?
   - **Kriteria "mampu"** untuk deteksi air. *Usulan angka (kamu dan mitra yang menentukan):* keruh vs kosong ≥95% pada botol yang tidak ikut latih (batas bawah CI ≥90%); air putih vs kosong ≥90% (batas bawah CI ≥80%) → "layak sebagai peringatan, load cell tetap penentu"; di bawah itu → "belum layak dengan kamera saja".
4. ☐ **Jetson:** tipe (Nano lama / Orin Nano), RAM, jenis kamera, kapan bisa hidup/ada internet. Dengan itu aku siapkan langkah deploy yang spesifik.

## B. Tugas fisik (butuh kamu di mesin)

1. ☐ **Backup data hari ini:** salin `datasets/primary/` (2,3 GB, foto mentah, **tidak tergantikan**), `datasets/cap_belt_v1/`, `datasets/_rf_export/`, dan `datasets/cap_runs/cap_belt_final/` ke flashdisk atau cloud. Folder `datasets/` tidak ikut git.
2. ☐ **Uji live model final di belt** (bilang "uji live", aku ganti `CAP_MODEL` ke `final.pt`):
   - Pastikan area belt aman dan e-stop terjangkau, lalu tekan MULAI SESI.
   - Uji 10–15 botol (bertutup, tanpa tutup, **termasuk cola hitam**, botol tutup putih). Catat tiap salah baca (merek, foto layar).
   - Kalau konveyor sudah baik, ulangi sambil belt berjalan pelan.
3. ☐ **Set cairan kedua** (±1,5–2 jam, lihat protokol di bawah). Ini yang menjawab "mampu atau tidak".
4. ☐ **Jetson**: saat hidup, kabari; kita lakukan ekspor TensorRT dan benchmark.

## C. Yang akan kukerjakan (kamu cukup menjawab "ya")
1. Memperbarui **laporan KP** (v4, dataset belt, CV 5-fold, pilot cairan, bagian keterbatasan).
2. **Ekspor ONNX + skrip benchmark** (laptop sebagai baseline; Jetson menyusul).
3. **Rapih direktori** + `datasets/README.md` (setelah kamu setuju).
4. **Menangani cola gelap di model bottle** (analisis, lalu kebutuhan data tambahan).
5. Menganalisis set cairan kedua dengan uji tahan-botol dan interval kepercayaan.

## D. Jadwal usulan

| tanggal | kamu | aku |
|---|---|---|
| **15 Okt** | A1–A4 (jawab), B1 (backup) | laporan (draf), skrip benchmark, rapih direktori bila setuju |
| **16 Okt** | B2 (uji live), B3 (set cairan) bila bahan siap | ekspor ONNX, analisis cola gelap |
| **17–20 Okt** | pertanyaan mitra dijawab; B4 bila Jetson hidup | analisis set cairan, bagian cairan di laporan |
| **21–24 Okt** | Jetson: benchmark dan uji; keputusan "mampu" bersama mitra | retrain model bottle (bila diputuskan), finalisasi laporan |
| **25–27 Okt** | latihan menjelaskan mekanisme (chat belajar) | cadangan waktu, perapihan akhir |

## E. Protokol set cairan kedua (ringkas)

Tujuan: satu sesi, botol sama, **hanya cairan yang berbeda**. Alat: `capture_water.py`.
- **Bahan:** ±10 botol Coca-Cola berbeda (minimal 6) + 4–5 botol air putih merek lain bila ingin target tambahan. Air bersih, lap kering, jangan ada air dekat motor/PLC; belt dalam keadaan berhenti.
- **Setelan:** kamera 1080p (alat menolak mulai kalau bukan), **fokus/exposure tidak dikunci**, pencahayaan dan posisi kamera **tidak diubah** selama sesi. Jalankan semua botol dalam satu kali duduk.
- **Urutan per botol** (menurun, minim menuang; 3 foto per kondisi di posisi 1/3, tengah, 2/3):
  1. Penuh **bertutup** (`penuh` + tutup ON) × 3
  2. Buka tutup, tuang hingga ±¼ (cukup rendah agar tidak tumpah saat rebah) (`sedikit` + tutup OFF) × 3
  3. Kosongkan (`kosong` + tutup OFF) × 3
  4. Pasang tutup (`kosong` + tutup ON) × 3
  Total 12 foto per botol, ±120 foto untuk 10 botol.
- **Acak urutan antar botol** (jangan semua merek sama berurutan). Botol tanpa tutup yang berisi lebih dari separuh jangan direbahkan (tumpah).
- Kenapa begini: pasangan tutup ON (penuh vs kosong) dan tutup OFF (sedikit vs kosong) memisahkan efek **cairan** dari efek **tutup** dan **sesi**, dua hal yang merusak pilot pertama.
