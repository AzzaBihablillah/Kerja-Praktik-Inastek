# Catatan Proyek — Deteksi Tutup (cap/no_cap) dan Isi Cairan

> **Lokasi file berubah 21 Okt 2026** (direktori dirapikan). Peta lokasi baru: `docs/00_STATUS_PROYEK.md`; daftar lengkap lama→baru: `datasets/PINDAH-LOG.csv`.

KP PT Inastek (DTETI UGM). Diperbarui 2026-10-15. Ringkasan semua yang sudah dikerjakan, diputuskan, dan masih terbuka.

## 1. Status singkat

| bagian | status |
|---|---|
| Model cap/no_cap domain belt | **selesai**: `datasets/4_eksperimen/cap/cap_runs/cap_belt_final/weights/final.pt` (YOLOv8n) |
| Akurasi (CV 5-fold per botol, 180 foto) | **98,3%** (CI 95,2–99,4%) pada imgsz 640 dan 960 |
| Dibanding baseline `cap_v4` | di imgsz 640: 97,8% (**tidak beda bermakna**); di imgsz 960: 92,8% |
| Dipasang di server PLC | **belum** (server masih `cap_v4_640`) |
| Uji live di belt | **belum** |
| Deteksi isi cairan | pilot **tidak bisa disimpulkan** (confound); butuh set kedua |
| Jetson | mati; tipe (Nano lama / Orin Nano) belum diketahui |
| Laporan KP (`docs/laporan-deteksi-cap-nocap.*`) | masih sampai v3; **belum** memuat v4, dataset belt, CV, cairan |
| Rapih direktori | rencana ada, **belum disetujui/dikerjakan** |

## 2. Kronologi dan hasil

**Model lama (September 2026).** v1 (sintetik + konveyor) baik di metrik tapi goyah live. v2 gagal karena degradasi gambar hanya diberikan ke kelas no_cap (model belajar "tajam = cap"). v3 (degradasi simetris) tidak lebih baik. v4 (jangkar data real + val/test 100% real, dibagi per botol): test mAP50 0,988 pada 79 foto real, model terbaik secara live. v4-8s (YOLOv8s) dan v5 (hard negative dasar botol) tidak mengalahkan v4, sehingga v4 dipakai.

**Dataset belt (2026-10-13).** 180 foto PNG 1920×1080 dari C922 di konveyor asli: 30 botol × 6 (3 cap + 3 no_cap, tutup di posisi 1/3, tengah, 2/3 lebar frame). Botol rebah di lembah belt V, leher selalu ke kanan, botol diam (konveyor rusak saat itu), cahaya 2 sumber dengan diffuser dan plastik hitam di tengah. Alat: `scripts/alat/capture_conveyor.py`. Fokus/exposure **tidak dikunci** (tombol `l` membuat gambar gelap setengahnya).

**Pelabelan.** Pra-label dengan `cap_v4` (kelas dari nama file, box dari model) → `datasets/2_dataset_cap/cap_belt_v1/`. Diunggah ke Roboflow (project `cap-nocap-belt-v1`), dikoreksi user, diekspor (Version 1, tanpa augmentasi/resize), lalu dibandingkan: 180/180 tepat 1 box, kelas cocok nama file, 124 sama, 52 digeser kecil, 3 sedang, 1 jauh. Box ketat berdasarkan pemeriksaan visual 16 sampel (bukan seluruh 180).

**Training.** Split per botol (train 22, val 4, test 4) lalu **CV 5-fold per botol** (30 botol → 5 fold × 6 botol, warna tutup disebar, epoch tetap 30, tanpa pemilihan epoch). Model final dilatih pada semua 180 foto dengan resep yang sama.

**Pilot cairan (2026-10-15).** Data: `Downloads/foto_botol_update` (76 foto, 640×360). Hasil tidak valid karena confound sesi dan tutup (lihat bagian 5).

**Server PLC.** `testing/cap version test/server.py`: model bottle + cap + `CapDecider` + snap7 ke PLC 192.168.0.1 (rack 0, slot 1), bit `MB10.0`.

## 3. Keputusan dan alasannya
- **Dua model terpisah (bottle + cap), jangan digabung 4 kelas.** Menggabung butuh semua gambar berlabel lengkap; dataset lama tidak punya box tutup, dan model bottle `ver7` sudah divalidasi untuk Capstone 1.
- **Cincin tamper bukan tutup.** no_cap = mulut leher berulir beserta cincin; cap = tutup beserta pita yang menempel. Leher halus panjang dan bahu botol tidak diikutkan. *Belum dikonfirmasi ke mitra.*
- **Split harus per botol**, bukan per foto, agar satu botol tidak muncul di train dan test.
- **Pakai `last.pt`/epoch akhir**, bukan `best.pt`: val 24 foto jenuh (mAP50 0,995 sejak epoch 1), jadi "terbaik" hanyalah noise.
- **Evaluasi di imgsz 640** (skala server). Jangan membandingkan model di skala yang tidak dipakai deploy.
- **Stabilisasi di level sistem dengan `CapDecider`** (gate ROI + voting 7–15 frame) untuk flicker dan deteksi ganda, tanpa retrain.
- **Kontrol kecepatan konveyor tidak bisa dari kode**: satu-satunya bit adalah `MB10.0` (ON/OFF); kecepatan diatur di HMI/PLC/inverter.

## 4. Angka kunci (untuk laporan)

| uji | hasil |
|---|---|
| CV 5-fold, model fine-tune, imgsz 640 | 177/180 = 98,3% (CI 95,2–99,4%) |
| CV 5-fold, model fine-tune, imgsz 960 | 177/180 = 98,3% |
| baseline `cap_v4`, imgsz 640 | 176/180 = 97,8% (CI 94,4–99,1%) |
| baseline `cap_v4`, imgsz 960 | 167/180 = 92,8% (no_cap 88,9%) |
| baseline `cap_v4`, imgsz 1280 (pengukuran awal) | no_cap 68/90 = 75,6% |
| mAP50 per fold (fine-tune) | 0,993 (sd 0,003); mAP50-95 0,770 |
| mean IoU box | sekitar 0,87–0,90 |

Kesalahan sisa model fine-tune (3/180, label benar): `b03_superindo_cap_03` (tutup biru berpita merah dibaca no_cap), `b25_natsbee_nocap_02` (leher putih berulir dibaca cap, 0,86), `b28_tropical1000_nocap_01` (leher bening + cincin kuning dibaca cap, 0,54).

**Catatan jujur untuk laporan:** keunggulan fine-tune adalah **ketahanan terhadap skala input**, bukan lonjakan akurasi di skala deploy. Semua uji memakai foto statis dari satu sesi dan satu pencahayaan (in-domain). Label akhir berawal dari tebakan `cap_v4`, sehingga IoU/mAP50-95 bisa sedikit menguntungkan `cap_v4`; akurasi kelas tidak terpengaruh (kelas dari nama file).

## 5. Kesalahan dan koreksi yang sudah terjadi
1. Klaim awal "`cap_v4` lemah di no_cap belt (76%)" hanya berlaku di imgsz 1280; di 640 ia 97,8%. **Jangan dikutip tanpa menyebut imgsz.**
2. `best.pt` epoch 1 adalah artefak early-stopping pada val yang jenuh.
3. Pilot air putih vs kosong tampak 100%, tetapi **kontrol menunjukkan area belt kosong saja memisahkan kedua kelompok 100%** (beda sesi/pencahayaan), ditambah beda tutup (air = tanpa tutup, kosong = bertutup). Hasilnya tidak bisa dipakai sebagai bukti.
4. Pemeriksaan otomatis (fill/hole/kontur/dominasi warna) tidak menangkap sisa jari di cutout lama; hanya pemeriksaan visual yang menemukannya.
5. v2 gagal karena degradasi asimetris antar kelas.
6. Pengalaman lama: memindahkan file data tanpa bertanya merusak kepercayaan. **Aturan kerja: tanya dulu sebelum memindah/menghapus data; pindah ke arsip, jangan hapus.**

## 5b. Keputusan rapat mitra (2026-10-16)
**Lingkup deteksi cairan = hanya cairan berwarna.** Air bening secara eksplisit di luar lingkup (belum bisa dideteksi dengan kamera); load cell tetap penentu validasi air. Pilot 2026-10-15 (air putih vs kosong) tidak bisa disimpulkan karena confound, sehingga tidak dijadikan klaim. Set kedua dirancang untuk cairan berwarna dengan uji tahan-botol dan tahan-warna; air bening hanya uji batas yang dilaporkan terpisah.

## 5c. Eksperimen cairan berwarna (malam 17–18 Okt 2026)
Rincian lengkap: `docs/HASIL-EKSPERIMEN-CAIRAN-18-OKT.md`.
- **Detektor `ver7` perlu dilatih ulang:** hanya mendeteksi 20 dari 40 foto botol berisi di set 16 Okt. `ver8` mendeteksi 100%.
- **Pendekatan yang dipakai: dua tahap** (`ver8` + classifier crop), bukan detektor 3 kelas.
- **Hasil classifier resep "W16" pada produk whitelist:** tanpa kesalahan di data uji yang ada (5 botol + sesi pilot). Masih bukti awal.
- **Model prototipe** ada di `datasets/4_eksperimen/cairan/final/` dan belum dipasang di server.

## 5d. Keputusan kebijakan (20 Okt 2026)
- **Botol berring** (cincin tamper tertinggal di leher) **valid**: yang penting bahan PET.
- **Ukuran bebas**, poin menurut 3 kategori: kecil, tanggung, besar (batas ml belum ditentukan).
- **Cairan di demo** dicek benar-benar ada air atau tidak. **Koreksi (20 Okt):** concern mitra adalah kamera menggantikan load cell, jadi rancangan dan klaim tidak boleh bergantung pada load cell. **Lingkup tetap cairan berwarna; air bening di luar lingkup** (keputusan rapat mitra 16 Okt): ditulis sebagai keterbatasan, bukan pekerjaan.
- **Uji kelayakan ukuran dari kamera** (24 botol C13 bervolume diketahui): korelasi peringkat 0,87; kecil terpisah jelas, tanggung dan besar tumpang tindih (sebagian karena box terpotong tepi frame).
- **Merek hanya tambahan poin** (diputuskan 20 Okt): semua botol PET diterima; non-Coca-Cola mendapat poin default. Bonus untuk semua produk Coca-Cola sebatas data yang ada; bila data tidak cukup, dibatasi whitelist (Coca-Cola, Fanta, Sprite incl. Nipis Mint). Metrik kunci model merek = salah memberi bonus.
- **Kategori ukuran (diubah user 20 Okt):** **kecil ≤ 350 ml, tanggung 351–800 ml (botol ritel 390 ml = tanggung), besar > 800 ml.** Sebelumnya Claude memakai kecil < 450 ml. Uji kamera → ukuran dengan batas baru (22 botol belt, leave-one-bottle-out): luas kotak **21/22 botol benar**; salah: Pristine 400 ml (luas 219k, sama dengan botol 350 ml 218k). Botol 350 ml dan 390–400 ml hampir tak terbedakan dari luas kotak; batas luas dipasang 228k (antara 350 ml ±218k dan 390 ml ±240k). Dengan batas lama (<450 ml) hasilnya 22/22.

## 5e. Keputusan model cairan (21 Okt 2026)
- **Resep utama: RH** (data belt 16 + 20 Okt, plus geser hue pada piksel botol, simetris berisi/kosong). Validasi per kelompok cairan: AUC 0,992; biru dikenali tanpa contoh biru; putih dan sangat pucat dikenali. Rincian: `CATATAN-PERCOBAAN-TA.md` L-14.
- **Cadangan lintas sesi:** ansambel RA + RH (AUC 0,979; sesi pilot 0,972).
- **Ambang:** 0,5, lalu dikalibrasi saat pemasangan (skor RH jenuh mendekati 1).
- **Kelemahan yang dicatat:** botol kosong berlabel penuh berwarna (Minute Maid); cairan merah belum ada di data.
- **Langkah:** latih model final RH dan RA dari semua data (W16 + 20 Okt), pasang ke demo live (`scripts/alat/live_liquid_demo.py`), belum ke server PLC.

## 6. Masih terbuka dan risiko
- **Resolusi/FOV deploy.** Server membaca 640×480 (FOV lebih sempit); model dilatih pada frame 16:9. Perlu keputusan satu setelan dan dikunci.
- **Model bottle lemah pada cola gelap**: Coca-Cola hitam terdeteksi 8/10 foto pada ambang server 0,35; satu foto tidak terdeteksi sama sekali. Relevan karena mitra fokus Coca-Cola.
- **Kebijakan mitra belum jelas:** botol berring tanpa tutup valid atau tidak; kriteria "mampu" untuk deteksi air; definisi target (kosong vs berisi, keruh).
- **Jetson:** tipe belum diketahui; OpenVINO tidak jalan di Jetson (pakai TensorRT/Ultralytics); `CapDecider` dan `BottleTracker` (numpy) bisa dipakai apa adanya.
- **Server PLC:** PLC pernah putus sementara tanpa peringatan di UI; kill paksa melewati penulisan bit OFF (belt bisa tertinggal ON).
- **Data tidak di git:** folder `datasets/` (±9,7 GB) tidak dilacak; `primary/` (2,3 GB, foto mentah) tidak tergantikan.
- **Lingkup:** CLAUDE.md menyatakan Capstone 1 hanya memakai deteksi botol/non-botol; cap/no_cap dan air adalah pekerjaan KP dan belum boleh masuk `main.py` sebelum diminta.

## 7. Peta file penting
- Model final: `datasets/4_eksperimen/cap/cap_runs/cap_belt_final/weights/final.pt`; model dasar: `datasets/4_eksperimen/cap/cap_runs/cap_v4_640/weights/best.pt`; model bottle: `export_model/bottle_best_ver7.pt`.
- Data belt: `datasets/1_foto_mentah/2026-10-13_conveyor_v1/` (mentah), `datasets/2_dataset_cap/cap_belt_v1/` (pra-label), `datasets/2_dataset_cap/_rf_export/` (label hasil koreksi), `datasets/2_dataset_cap/cap_belt_ds/` (split per botol), `datasets/2_dataset_cap/cap_belt_all/` (semua 180).
- Hasil CV: `datasets/4_eksperimen/cap/_cv_belt/cv_summary.txt`, `cv_records.jsonl`.
- Pilot cairan: `datasets/4_eksperimen/cairan_pilot/` (`report.txt`, `compare.jpg`, `index.csv`).
- Skrip baru: `capture_conveyor.py`, `capture_water.py`, `prelabel_belt.py`, `build_belt_ds.py`, `train_belt.py`, `train_belt_final.py`, `cv_belt.py`, `cap_decider.py`, `cap_decision_test.py`, `liquid_pilot_prep.py`, `liquid_pilot_analyze.py`.
- Server: `testing/cap version test/server.py`; antarmuka web lama: `open_vino/server.py`.

## 8. Mengulang pipeline (urutan)
```
python scripts/alat/capture_conveyor.py --cam 1        # ambil foto (1080p, PNG)
python datasets/prelabel_belt.py                   # label tebakan -> koreksi di Roboflow -> ekspor YOLOv8
python datasets/build_belt_ds.py                   # split per botol
python datasets/cv_belt.py                         # CV 5-fold (angka laporan)
python datasets/train_belt_final.py                # model final di semua foto
```
Semua dijalankan dengan `venv/Scripts/python.exe` dari folder proyek.

## 9. Keselamatan PLC/belt
- Belt hanya berjalan setelah **MULAI SESI**. Pastikan tangan jauh dari belt dan e-stop terjangkau.
- Hentikan server dengan **SELESAI** (atau `POST /conveyor/stop`) sebelum mematikan prosesnya.
- Setelah server berhenti, cek langsung bahwa belt diam; bit `MB10.0` bisa dibaca dengan snap7 kalau PLC terjangkau.
