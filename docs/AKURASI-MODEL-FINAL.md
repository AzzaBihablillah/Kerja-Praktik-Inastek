# Akurasi final model yang dipakai (per 30 Okt 2026)

Set aktif (`rvm_kp/config.py MODEL_SETS`): detektor `botol_ver9c`, tutup `cap_belt_v3_kandidat`, isi `cairan_isi_v3_kandidat`, merek `merek_coca_v2_kandidat` (bonus +10), ukuran aturan `panjang`.

Semua angka dari botol yang **tidak ikut dilatih** model tersebut. Rentang dalam kurung siku = Wilson 95%.
Tutup & isi: validasi per hari (model dilatih tanpa hari uji, lalu diuji pada hari itu: 27 Okt dan 29 Okt), dinilai lewat jalur aplikasi (720p, profil nano, 3 frame diam per botol). Sumber: `datasets/4_eksperimen/replay_keputusan/hasil.jsonl` (varian `semua_baru`), CATATAN-PERCOBAAN-TA L-20, L-25, L-32, §4b.

## 1. Detektor botol `botol_ver9c`

| uji | hasil |
|---|---|
| Botol asli terdeteksi, sesi 20 Okt (tak dilatih) | 117/117 = 100% [97–100] |
| Botol asli terdeteksi, foto nyata 22 Okt | 64/64 = 100% [94–100] |
| Benda bukan botol di belt 27 Okt (live) | 22/22 ditolak = 100% [85–100] (18 langsung "bukan botol", 4 tertolak di tahap tutup/isi) |
| Botol salah ditandai benda asing | 0/391 foto botol |

## 2. Tutup `cap_belt_v3` (botol terbaca bertutup / tanpa tutup dengan benar)

| kategori | hasil |
|---|---|
| Tutup asli (sewarna cincin) | 202/208 = 97,1% [94–99] |
| Tutup campuran (sebagian ditukar warna) | 139/152 = 91,4% [86–95] |
| Tanpa tutup (tidak salah dibaca bertutup) | 108/109 = 99,1% [95–100] |
| Semua botol bertutup | 341/360 = 94,7% [92–97] |

## 3. Isi cairan `cairan_isi_v3` (skor isi saja, tutup diabaikan)

| kategori | hasil |
|---|---|
| Kosong (terbaca kosong) | 217/222 = 97,7% [95–99] |
| Air ±1/6 (satu-satunya air yang realistis datang tanpa tutup) | 63/75 = 84,0% [74–91] |
| Air ±1/2 | 74/75 = 98,7% [93–100] |
| Air penuh | 46/62 = 74,2% [62–83] |
| Air acak (27 Okt) | 32/35 = 91,4% [78–97] |
| Semua botol berisi | 215/247 = 87,0% [82–91] |

Air ±1/2 dan penuh selalu bertutup di belt (kalau tidak, tumpah), jadi tertangkap oleh pemeriksaan tutup: keputusan mesin untuk air penuh 62/62 dan ±1/2 75/75 benar.

## 4. Merek `merek_coca_v2` (bonus Coca-Cola Company)

Uji sah = 29 Okt saja (v2 sudah dilatih dengan data 27 Okt).

| kategori | hasil |
|---|---|
| Produk Coca-Cola Company dikenali (dapat bonus) | 108/109 = 99,1% [95–100] |
| Bukan Coca-Cola tidak dapat bonus | 215/225 = 95,6% [92–98] (10 salah dapat bonus) |
| Gabungan | 323/334 = 96,7% [94–98] |

## 5. Ukuran (aturan panjang kotak, bukan model latih)

| uji | hasil |
|---|---|
| 26 botol ber-ml (13/16/20 Okt), tinggalkan-satu-botol | 25/26 = 96,2% [81–99] (salah: galon 1 L pendek-gemuk; ditutup aturan tambahan luas ≥ 360k = besar) |

Belum diuji di sesi belt 27/29 Okt (tidak ada label ml). Wajib dikalibrasi ulang bila posisi kamera berubah.

## 6. Keputusan mesin keseluruhan (terima / tolak)

| hari | hasil | salah terima | salah tolak |
|---|---|---|---|
| 27 Okt | 104/107 | 2 | 1 |
| 29 Okt | 357/362 | 5 | 0 |
| **Gabungan** | **461/469 = 98,3% [97–99]** | **7** | **1** |

Pembanding set lama (ver9c + cap v2 + RH): 449/469 = 95,7%, salah terima 12, salah tolak 8.
Sisa salah terima: 6 botol kosong bertutup yang tutupnya tak terlihat (tutup biru tua / tutup tukaran) + 1 air ±1/6 bertutup.

### Seberapa banyak tutup "menolong" botol berisi

Semua 247 botol berisi di uji 27/29 Okt bertutup. Alasan tolak (isi v3 + tutup v3):

| alasan | botol |
|---|---|
| tutup DAN cairan terdeteksi | 203 |
| cairan saja | 12 |
| **tutup saja (cairan luput, tertolong tutup)** | **30** (air penuh 16, ±1/6 11, acak 2, ±1/2 1) |
| lolos diterima | 2 |

Angka 461/469 = benar untuk mesin dengan botol seperti yang diuji (berisi = bertutup). Bila tutup TIDAK dihitung untuk botol berisi (botol berisi hanya benar kalau model isi melihat airnya):
botol kosong 216/222 + botol berisi 215/247 = **431/469 = 91,9% [89–94]**.
Kasus nyata tanpa tutup (air ±1/6): **63/75 = 84,0% [74–91]**, kira-kira 1 dari 6 botol lolos.

## Batasan

- Satu medan: belt capstone, kamera C922, pencahayaan lab. Belum diuji di belt/kamera/cahaya lain.
- Label = label kelompok per sesi.
- Di luar lingkup (L-30): botol cover penuh, buram, berwarna gelap.
- Uji sah berikutnya: sesi belt dengan botol & tutup baru (`docs/PROTOKOL-SESI-BELT-BERIKUTNYA.md`).
