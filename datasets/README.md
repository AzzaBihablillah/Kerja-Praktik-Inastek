# datasets/ — data, model, dan hasil eksperimen (TIDAK ikut git; backup manual)

| folder | isi | catatan |
|---|---|---|
| `1_foto_mentah/` | foto asli kamera: `2026-10-13_conveyor_v1` (cap/no_cap, 30 botol), `2026-10-16_water_v1` (cairan, 5 botol), `2026-10-20_water_v1` (cairan, 16 botol), shoot studio Sep | **tak tergantikan**; tiap folder punya `MANIFEST.csv` |
| `2_dataset_cap/` | dataset tutup turunan: `cap_belt_v1` (pra-label), `_rf_export` (koreksi Roboflow), `cap_belt_ds`/`cap_belt_all` (split), `cap_v4`, `shoot_v1/v2`, `cap_shoot_v1/v2`, `bottle_cap_base`, `cap_conveyor_raw` | |
| `3_model_final/` | `botol_ver8b.pt`, `cap_belt_final.pt`, `cap_v4_640.pt` (dipakai server PLC), `cairan_RH.pt` (utama), `cairan_RA.pt`; `dasar/` = bobot awal yolov8* | salinan dari folder eksperimen |
| `4_eksperimen/cap/` | `cap_runs/` (semua run tutup), `_cv_belt/` (hasil CV 5-fold) | |
| `4_eksperimen/cairan/` | semua keluaran percobaan cairan: crop (`crops_fold*`), indeks (`*.csv`), laporan (`*_report.txt`, `cls_summary.txt`), `runs_*`, `final*/`; `log/` = log dan flag | angka di `docs/CATATAN-PERCOBAAN-TA.md` |
| `4_eksperimen/cairan_pilot/` | pilot 15 Okt | tidak dapat disimpulkan |
| `5_data_luar/` | `_dl_*` (Roboflow Water Detection, RF100 soda, Open Food Facts, TACO), `_warisan` (data tim lama), `water_internet_fanta` | lisensi CC BY / CC BY-SA, atribusi wajib |
| `_arsip/` | eksperimen cap lama (sintetis), gambar debug, foto cek kamera | |

`PINDAH-LOG.csv` mencatat setiap pemindahan 21 Okt (lokasi lama → baru).
