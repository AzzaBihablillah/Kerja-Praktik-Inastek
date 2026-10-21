# Daftar Potret di Belt — Negatif Nyata & Uji Iseng (disusun 22 Okt 2026)

**Tujuan:** menentukan model detektor botol mana yang dipakai (`botol_ver8b` sekarang vs kandidat `botol_ver9c`) dengan foto NYATA, dan mengukur ketahanan terhadap uji iseng penguji. Semua kandidat sudah diuji dengan foto tempelan/sintetis; keputusan akhir hanya sah dari foto ini (`CATATAN-PERCOBAAN-TA.md` bagian 4b).

Foto ini dipakai **untuk UJI dulu**. Baru setelah keputusan diambil, sebagian boleh dipakai latih (dibagi per benda, bukan per foto).

## 0. Aturan sesi (penting)
- **Belt MATI**, benda diletakkan diam (sama seperti sesi 13/16/20 Okt). Tidak perlu menyalakan PLC.
- Cahaya dan posisi kamera sama dengan sesi sebelumnya; kamera C922 1080p (alat menolak jalan jika bukan 1080p).
- **Satu sesi untuk semuanya**, termasuk kontrol botol PET (bagian 3). Tanpa kontrol di sesi yang sama, model bisa membedakan foto dari "hari"-nya, bukan dari bendanya (sudah terbukti di percobaan cairan).
- Tiap benda **3 foto**: posisi kiri, tengah, kanan pita belt, dan putar/balik bendanya tiap foto.
- Nama benda diketik saat ganti benda (tombol `n`); pakai nama dari kolom "nama" di bawah + nomor benda, mis. `plastik-bening-remas-1`.

```bash
venv/Scripts/python.exe scripts/alat/capture_free.py --cam 1 --name plastik-bening-remas-1
```
Tombol: `SPACE` simpan · `n` ganti nama benda (ketik di terminal) · `z` batalkan foto terakhir · `q` keluar.
Hasil: `datasets/1_foto_mentah/<tanggal>_nonbottle_v1/` + `MANIFEST.csv`.

## 1. Negatif nyata (bukan botol) — PRIORITAS 1
| no | nama | benda | jumlah benda | foto | catatan |
|---|---|---|---|---|---|
| 1 | `plastik-bening-remas` | plastik kemasan / kantong bening diremas jadi gumpal | 8 | 24 | **paling penting**: semua model masih menerima ini |
| 2 | `kantong-plastik` | kresek (hitam, putih, warna), kantong bening tidak diremas | 4 | 12 | |
| 3 | `gelas-plastik` | gelas cup minuman (dengan/tanpa tutup sealer), gelas air mineral | 4 | 12 | bening dan bulat, mirip badan botol |
| 4 | `wadah-makanan` | mika/clamshell, wadah styrofoam, kotak plastik | 4 | 12 | |
| 5 | `kaleng` | kaleng minuman (cola, bir/soda, kopi) | 4 | 12 | model lama menerima 1 dari 3 kaleng tempelan |
| 6 | `botol-kaca` | botol kaca (sirup, kecap, minuman botol kaca) | 3 | 9 | **negatif tersulit** (bentuk = botol, bahan bukan PET) |
| 7 | `karton-minuman` | kotak susu UHT / teh kotak | 3 | 9 | |
| | | | **30 benda** | **90 foto** | |

## 2. Uji iseng penguji (spoof) — PRIORITAS 1
| no | nama | cara | foto | catatan |
|---|---|---|---|---|
| 8 | `hp-foto-botol` | HP layar menghadap ke atas, menampilkan foto botol (Coca-Cola, Fanta, Sprite, Aqua, botol bening, botol berwarna) | 6 gambar × 2 kecerahan (penuh, sedang) = 12 | ukuran botol di layar kira-kira seukuran tampilan botol asli |
| 9 | `kertas-foto-botol` | foto botol dicetak seukuran asli; kertas doff dan glossy | 3 gambar × 2 jenis kertas × 2 posisi = 12 | |
| 10 | `busa-gambar-botol` | gambar botol ditempel ke busa/kardus tebal (3D palsu) | 2 benda × 3 = 6 | |
| | | | **30 foto** | |

## 3. Kontrol botol PET di sesi yang sama — WAJIB
| no | nama | benda | foto |
|---|---|---|---|
| 11 | `pet-<merek><ml>` mis. `pet-aqua600` | 10 botol PET **kosong tanpa tutup**, merek & ukuran berbeda (boleh botol bekas sesi lalu; usahakan 2 bertutup hitam, 2 berwarna) | 10 × 2 = 20 |

Tanpa bagian ini, hasil uji negatif tidak bisa dipercaya.

## 4. Opsional — kalau masih sempat (pekerjaan terbuka lama)
| no | nama | benda | foto | catatan |
|---|---|---|---|---|
| 12 | `botol-nonpet` | botol HDPE: sabun cair, sampo, jerigen kecil | 3 × 3 = 9 | **tanya mitra dulu**: botol non-PET ditolak atau diterima? |
| 13 | (pakai `capture_water.py`) | botol KOSONG berlabel penuh berwarna (Minute Maid, Isoplus) + botol bekas diisi sirup MERAH | ikuti protokol alat | untuk kelemahan model cairan (status item 1) |

## Ringkas
- Wajib: **±140 foto** (bagian 1 + 2 + 3), kira-kira 40–50 menit.
- Belanja/siapkan: 8 plastik/kantong bening, 4 kresek, 4 gelas cup, 4 wadah makanan, 4 kaleng, 3 botol kaca, 3 kotak susu/teh, 10 botol PET bekas, 1 HP, 6 cetakan foto botol (3 doff + 3 glossy), 2 busa/kardus bergambar botol.
- Setelah selesai: kabari Claude → uji `ver8b` vs `ver9c` (dan `ver9b`) per kategori → putuskan model.
