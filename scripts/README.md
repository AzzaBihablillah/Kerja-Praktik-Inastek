# scripts/ — semua kode KP (dijalankan dari folder proyek dengan `venv/Scripts/python.exe`)

Path data dihitung relatif terhadap folder proyek (`datasets/...`), jadi skrip bisa dijalankan dari mana saja.
Dipindah dari `datasets/` dan `datasets/liquid_exp/` pada 21 Okt 2026 (`datasets/PINDAH-LOG.csv`).

## alat/ — alat lapangan
| skrip | fungsi |
|---|---|
| `capture_water.py` | potret set cairan (penuh → setengah → kosong tanpa tutup → kosong bertutup, 4 sisi), QC otomatis, `--center` untuk botol besar |
| `capture_free.py` | potret bebas untuk negatif non-botol (plastik bening, kantong, gelas) |
| `capture_conveyor.py` | potret set cap/no_cap di belt (13 Okt) |
| `live_liquid_demo.py` | demo live: botol → tutup → cairan berwarna → ukuran; tanpa PLC, tidak menyimpan |
| `qc_water_set.py`, `water_qc.py` | audit folder foto: botol terdeteksi, tutup sesuai label, arah leher |
| `keep_awake.py` | mencegah laptop tidur saat training panjang |

## pipeline_cap/ — model tutup
`prelabel_belt.py` (pra-label dengan cap_v4) → koreksi di Roboflow → `build_belt_ds.py` (split per botol) → `train_belt.py` → `cv_belt.py` (CV 5-fold, angka laporan) → `train_belt_final.py`. `cap_decider.py` = keputusan stabil (gate ROI + voting), dipakai server PLC. `live_cap_test.py`, `live_ab_test.py`, `cap_decision_test.py` = uji live dan uji logika keputusan.

## pipeline_cairan/ — deteksi cairan berwarna (urutan nomor = urutan kerja)
| skrip | fungsi |
|---|---|
| `liquid_pilot_*.py`, `liquid_baseline.py`, `liquid_learning_curve.py` | pilot 15 Okt, baseline fitur warna, kurva fitur warna |
| `common.py` | `fold_crop` (crop lipat 224 px) dipakai bersama |
| `s00`–`s02b` | salin data warisan, indeks foto belt + kotak botol, crop |
| `s03`–`s04c` | CV classifier per botol (varian V1–V6, G, S, P, E, T), evaluasi, AUC, rincian per produk |
| `s05`–`s09` | data detektor, detektor 3 kelas vs ver8, evaluasi ujung-ke-ujung, sapuan ambang |
| `s10`, `s23`, `s30` | model final (final, final2, final3 = RH/RA) |
| `s11`–`s16` | kilau sintetis, RF100, preprocessing P1–P3, Open Food Facts, lembar brand2 |
| `s20`, `s21` | registri dataset, unduh TACO |
| `s22`–`s22d` | kurva belajar CNN (s22 cacat; **s22c/s22d = versi benar**) |
| `s24`, `s25` | cek jalan pintas sesi, pralatih data luar |
| `s26`–`s28b` | audit sesi 20 Okt, CV per botol, **CV per kelompok cairan (s28: RA/RB/RC/RH)** |
| `s29` | geser hue piksel botol (dasar resep RH) |

Keluaran semua skrip cairan: `datasets/4_eksperimen/cairan/`.

## _arsip/ — skrip lama
Pipeline sintetis dan studio (Sep). **Path di dalamnya belum diperbarui** (masih `datasets/<nama lama>`); sesuaikan dengan `datasets/PINDAH-LOG.csv` bila dijalankan lagi.

`_migrasi_21okt.py` dan `_migrasi_path_21okt.py` = catatan migrasi direktori 21 Okt (jangan dijalankan ulang).
