# Daftar Kerja untuk Kamu (dicek 19 Okt 2026)

Tanda: ☐ belum, ✔ sudah. Asumsi tenggat 27 Okt (belum dikonfirmasi). Jendela pengambilan data: sekitar 2 hari mulai 19 Okt.
Urutan: **A (keputusan, 10 menit) → B (tanya mitra) → C (siapkan bahan) → D (potret) → E (cek mesin)**.

## A. Keputusan yang hanya kamu yang bisa ambil
1. ☐ **Fanta Grape ikut diterima?** (cairan ungu bagus untuk uji warna; ukuran PET-nya belum kupastikan)
2. ☐ **Satu ukuran saja atau lebih?** Saranku satu ukuran (misalnya 390 ml) supaya jumlah SKU kecil.
3. ☐ **Sprite Nipis Mint:** dimasukkan atau tidak? Saranku opsional. Hanya 1 botol yang sudah ada (b01); produknya baru meluncur Juni 2026 dan hanya 390 ml dan 1 L.
4. ☐ **Frestea:** ditolak (sesuai whitelist). Konfirmasi saja bahwa itu yang kalian mau.
5. ☐ **Botol di set 16 Okt ternyata ukuran berapa?** Manifest mencatat b01 "250", padahal sumber hanya menyebut 390 ml dan 1 L. Cek label botolnya.

## B. Pertanyaan ke mitra (bisa lewat chat)
1. ☐ Botol demo dipilih tim: oke. Tanyakan tetap: **botol berring tanpa tutup dianggap valid atau ditolak?** (sekarang kuanggap valid)
2. ☐ Apakah botol demo akan **diisi cairan** pada demo, dan boleh pakai cairan asli?
3. ☐ Lama demo dan perkiraan jumlah botol per sesi.

## C. Siapkan bahan (sebelum memotret)
Kit demo, 6 sampai 8 botol:
- ☐ Coca-Cola Original, ☐ Coca-Cola Zero
- ☐ Fanta Strawberry, ☐ Fanta Orange, ☐ (opsional) Fanta Grape
- ☐ Penolakan: 1 botol merah merek lain (BIG Cola atau BIG Strawberry), 1 botol air bening, 1 non-botol
- ☐ **Set bukti (jangan dipakai melatih):** botol lain dengan SKU yang sama dengan kit demo, 6 sampai 8 botol, hanya 2 sampai 4 foto per botol
- ☐ Lap kering, belt dalam keadaan berhenti, minuman jauh dari PLC dan motor

## D. Memotret (alat baru, ada pemeriksaan otomatis)
Dari folder `datasets`:
```bash
../venv/Scripts/python.exe capture_water.py --list
../venv/Scripts/python.exe capture_water.py --cam <index C922>
```
Alat menolak mulai kalau bukan 1080p. Tekan `b` untuk botol baru lalu ketik nama dan warna. Langkah protokol per botol (16 foto, 4 sisi tiap langkah):
1. penuh + tutup terpasang → 2. setengah + tutup terpasang → 3. kosong, tutup lepas → 4. kosong, tutup terpasang.
(Level seperempat sudah dihapus dari protokol.)
- ☐ Tekan `e` (foto belt kosong) di awal, tengah, dan akhir sesi.
- ☐ Acak urutan botol. Jangan ada tangan atau benda lain di frame. Lap belt di antara botol.
- ☐ Kalau bilah di bawah layar merah (bip) atau kuning: periksa tutup/botolnya, tekan `z` lalu foto ulang.
- ☐ **Putar botol 90° tiap sisi dan arahkan leher sesuai tulisan di layar.** Di set 16 Okt leher hampir selalu di kanan walau diminta kiri.
- ☐ Tombol penting: `c`/`x` tutup, `k` lewati, `n` koreksi nama, `v` botol sebelumnya, `z` undo.
- ☐ Selesai: audit hasilnya. Aku yang menjalankan `qc_water_set.py`; kamu cukup bilang "audit".

## E. Cek di mesin (butuh kamu di sana)
1. ☐ **Backup `datasets/primary/`** (foto mentah, tidak ada di git, tidak tergantikan) ke flashdisk atau cloud. Salin juga `datasets/cap_runs/cap_belt_final/`.
2. ☐ **Uji live** kit demo di belt berjalan. Bilang dulu "uji live"; aku ganti model ke `cap_belt_final`. Pastikan belt aman dan e-stop terjangkau sebelum MULAI SESI.
3. ☐ Kalau Jetson hidup: kabari tipe, RAM, dan kamera. Aku siapkan ekspor TensorRT dan benchmark.

## Yang aku kerjakan (kamu cukup menjawab "ya")
- Laporan KP: v4, dataset belt, CV 5-fold, lingkup cairan berwarna, whitelist merek, set 16 Okt, baseline warna, kurva belajar, dan keterbatasan.
- Skrip evaluasi model merek lama pada set 16 Okt dan foto belt dengan kelas `produk_mitra`, supaya baseline tercatat.
- Ekspor ONNX dan skrip benchmark (laptop dulu).
- Rapih direktori (menunggu persetujuanmu; tidak ada yang dihapus, hanya dipindah ke `datasets/_arsip/`).
- Setelah data baru ada: pelabelan, split per botol, training, evaluasi leave-one-bottle-out.

## Catatan jujur untuk laporan
- Hasil demo dengan botol yang dipilih sendiri bukan akurasi generalisasi. Laporkan dua angka: botol demo dan botol lain dengan SKU yang sama.
- Set cairan 16 Okt hanya 5 botol: pilot, bukan bukti "mampu". Fitur warna sederhana menghasilkan sekitar 88% (uji tahan-botol); Sprite Nipis Mint pucat di belt hijau adalah kasus terburuk (59%).
- Model merek lama menerima banyak produk non-Coca-Cola yang berwarna merah atau cola (BIG Cola, BIG Strawberry, Vit: 6 dari 6). Itu alasan model merek baru perlu data negatif.
