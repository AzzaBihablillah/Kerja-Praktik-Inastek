# Protokol sesi belt berikutnya (disiapkan 30 Okt)

Tujuan: **uji sah pertama** untuk model yang aktif sejak 30 Okt (tutup v3 + isi v3) dan sekaligus menilai model tutup
warna buatan (v4w) sebagai kandidat bayangan, pada **botol & tutup yang belum pernah dipakai** di sesi 22–29 Okt.

## Menjalankan aplikasi

```
venv\Scripts\python -m app --plc --ambil-data --kandidat
```

- `--plc` hanya setelah area belt dipastikan aman.
- `--kandidat`: layar & poin tetap model aktif; tutup v4w hanya dicatat di `SCAN_KANDIDAT.csv` (detektor, isi, ukuran kandidat = sama dengan aktif, jadi beda keputusan = beda model tutup saja).
- Talang ESP32 opsional (COM7). Tanpa talang, keputusan sortir hanya dicatat.

## Urutan sesi (satu label kelompok per sesi di halaman /operator)

| Sesi | Isi | Tutup (label baru) | Botol | Yang diuji |
|---|---|---|---|---|
| A | air ±1/6 | Tanpa tutup | ±50 (Coca + non-Coca) | celah utama model isi |
| B | kosong | Tutup biru | ±25 | tutup biru tua/muda (lolos di 27 & 29 Okt) |
| C | kosong | Tutup tukaran (beda warna) | ±25 | tutup dari botol lain, warna beda dengan cincin |
| D | kosong | Tutup asli (sewarna cincin) | ±25 | kasus umum tidak boleh memburuk |
| E | kosong | Tanpa tutup | ±25 | kontrol salah tolak |

Aturan:
- **Satu jenis tutup per sesi.** Jangan campur tutup asli dan tukaran dalam satu sesi (pelajaran 29 Okt: tidak bisa dipisah lagi).
- Botol & tutup **baru** (belum dipakai di sesi 22–29 Okt). Catat di kolom catatan bila terpaksa memakai botol lama.
- Air ±1/6 TANPA tutup saja; air ±1/2 / penuh tanpa tutup tidak realistis (tumpah di belt).
- Batasan lingkup L-30 tetap: tanpa botol cover penuh, buram, atau berwarna gelap.
- Botol air ±1/6 sebaiknya campur produk; bila ada, sertakan beberapa botol Coca-Cola (paling sulit di 29 Okt).

## Sesudah sesi (dikerjakan Claude)

1. Rekap keputusan aktif per sesi vs label (Wilson 95%).
2. Bandingkan tutup v3 (aktif) vs v4w (bayangan) per jenis tutup, botol demi botol, dengan kriteria L-33: v4w hanya
   menggantikan v3 bila salah terima turun, salah tolak tidak naik, dan tutup asli tidak memburuk.
3. Data sesi menjadi bahan latih ulang berikutnya (audit kotak tutup seperti 27/29 Okt).

Catatan teknis: label `tutup_asli` / `tutup_biru` / `tutup_tukar` dihitung sebagai bertutup (`app/config.py TUTUP_BERTUTUP`).
Skrip lama yang menyaring `k_tutup == "bertutup"` (audit tutup, replay) perlu memakai `TUTUP_BERTUTUP`.
