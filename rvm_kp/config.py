"""Konfigurasi sistem RVM KP: lokasi model per backend, profil perangkat, ambang, batas ukuran.
Ubah di sini saat memasang di perangkat baru (setelah kalibrasi). Kode harus tetap jalan di Python 3.8 (Jetson Nano, Docker JetPack 4)."""
import os

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M = os.path.join(REPO, "datasets", "3_model_final")          # model .pt asli
E = os.path.join(M, "ekspor")                                  # hasil export_models.py

# Ukuran input model. Model tutup mode "roi" diekspor terpisah dengan ukuran CAP_ROI_IMGSZ (file *_roi.*).
IMGSZ_BOTTLE = 640
IMGSZ_CAP_FULL = 640
IMGSZ_LIQUID = 224
CAP_ROI_IMGSZ = 416           # kanvas potongan area botol (uji setelan 21 Okt malam, cv_cap_v2_roi.py: 416 > 320 untuk tutup hitam)
CAP_ROI_SCALE = 1 / 2         # skala potongan relatif frame selebar 1920 px (= skala latih model tutup, imgsz 960)
CAP_ROI_MARGIN = 0.08         # tambahan tepi kotak botol (fraksi lebar/tinggi kotak)

# 30 Okt (persetujuan user): tutup v3 + isi v3, latih ulang dgn data belt 27 + 29 Okt (CATATAN L-32; simulasi keputusan
# 449 -> 461/469). Kembali ke set lama: cap_belt_v3_kandidat -> cap_belt_v2, cairan_isi_v3_kandidat -> cairan_RH.
# tensorrt: engine v3 BELUM dibuat -> buat ulang di Nano (rvm_kp.export_models --format engine --half --only ...).
_CAP, _ISI = "cap_belt_v3_kandidat", "cairan_isi_v3_kandidat"
MODEL_SETS = {
    "pytorch": dict(bottle=f"{M}/botol_ver9c.pt", cap=f"{M}/{_CAP}.pt", cap_roi=f"{M}/{_CAP}.pt", liquid=f"{M}/{_ISI}.pt"),
    "onnx": dict(bottle=f"{E}/botol_ver9c.onnx", cap=f"{E}/{_CAP}.onnx", cap_roi=f"{E}/{_CAP}_roi.onnx", liquid=f"{E}/{_ISI}.onnx"),
    "openvino": dict(bottle=f"{E}/botol_ver9c_openvino_model", cap=f"{E}/{_CAP}_openvino_model",
                     cap_roi=f"{E}/{_CAP}_roi_openvino_model", liquid=f"{E}/{_ISI}_openvino_model"),
    "tensorrt": dict(bottle=f"{E}/botol_ver9c.engine", cap=f"{E}/{_CAP}.engine", cap_roi=f"{E}/{_CAP}_roi.engine", liquid=f"{E}/{_ISI}.engine"),
}

# Detektor botol: botol_ver9c (22 Okt; uji NYATA di belt: non-botol diterima 7/32 vs ver8b 13/32, kontrol PET 64/64 keduanya).
#   Kembali ke ver8b: ganti 'botol_ver9c' -> 'botol_ver8b' di MODEL_SETS.
# Model tutup: cap_belt_v2 (21 Okt) = 13 Okt + 16/20 Okt termasuk tutup hitam; CV per merek 99,0% (cap_belt_final: 88,2%).
# Profil perangkat. "nano" = batasan Jetson Nano 4 GB B01: kamera 720p, tutup hanya di area botol, FP16, 4 core CPU.
# Di laptop, profil "nano" MENIRU batasannya (bukan kecepatannya): angka kecepatan Nano hanya sah bila diukur di Nano.
PROFILES = {
    "laptop": dict(frame_width=1920, cap_mode="full", half=False, cpu_threads=None),
    "nano": dict(frame_width=1280, cap_mode="roi", half=True, cpu_threads=4),
}
DEFAULT_PROFILE = "nano"

CONF_BOTTLE = 0.35            # sama dengan server PLC
CONF_CAP = 0.25               # mode full-frame (sama dengan server PLC)
CONF_CAP_ROI = 0.03           # mode ROI: ambil kotak terbaik walau lemah (uji 21 Okt: 0 'tidak terdeteksi', akurasi = full-frame)
LIQUID_THR = 0.5              # KALIBRASI saat pemasangan (skor model RH jenuh mendekati 1)
# Gabungan model cairan (23 Okt, CV s32: rata-rata RH+RN salah 40/325 vs RH 45 vs RN 42; lebih stabil di warna pucat).
# None = mati (hanya model 'liquid'). Aktifkan: LIQUID_EXTRA = f"{M}/cairan_RN.pt"  (backend pytorch; backend lain perlu ekspor RN).
LIQUID_EXTRA = None
# Ukuran dari luas kotak botol, dinyatakan dalam piksel^2 SETARA frame 1920 px (resolusi kamera lain dinormalisasi otomatis).
# kecil <=350 ml, tanggung 351-800 ml, besar >800 ml (keputusan user 20 Okt). KALIBRASI per pemasangan (tinggi kamera).
AREA_KECIL_TANGGUNG = 228_000
AREA_TANGGUNG_BESAR = 357_000
# Alternatif (23 Okt, scripts/alat/analisis_ukuran.py, 26 botol uji tinggalkan-satu-botol): PANJANG botol (sisi panjang kotak, setara
# frame 1920 px) benar 24/26 vs luas 22/26 (luas salah di botol ramping 400 ml). Galon gemuk ditutup aturan luas: luas >= AREA_BESAR_PASTI -> besar.
# Pilih dengan SIZE_RULE = "luas" (bawaan, cara lama) atau "panjang". KALIBRASI per pemasangan.
SIZE_RULE = "luas"
LEN_KECIL_TANGGUNG = 817
LEN_TANGGUNG_BESAR = 1060
AREA_BESAR_PASTI = 360_000
