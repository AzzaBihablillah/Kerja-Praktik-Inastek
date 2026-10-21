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

MODEL_SETS = {
    "pytorch": dict(bottle=f"{M}/botol_ver8b.pt", cap=f"{M}/cap_belt_v2.pt", cap_roi=f"{M}/cap_belt_v2.pt", liquid=f"{M}/cairan_RH.pt"),
    "onnx": dict(bottle=f"{E}/botol_ver8b.onnx", cap=f"{E}/cap_belt_v2.onnx", cap_roi=f"{E}/cap_belt_v2_roi.onnx", liquid=f"{E}/cairan_RH.onnx"),
    "openvino": dict(bottle=f"{E}/botol_ver8b_openvino_model", cap=f"{E}/cap_belt_v2_openvino_model",
                     cap_roi=f"{E}/cap_belt_v2_roi_openvino_model", liquid=f"{E}/cairan_RH_openvino_model"),
    "tensorrt": dict(bottle=f"{E}/botol_ver8b.engine", cap=f"{E}/cap_belt_v2.engine", cap_roi=f"{E}/cap_belt_v2_roi.engine", liquid=f"{E}/cairan_RH.engine"),
}

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
# Ukuran dari luas kotak botol, dinyatakan dalam piksel^2 SETARA frame 1920 px (resolusi kamera lain dinormalisasi otomatis).
# kecil <=350 ml, tanggung 351-800 ml, besar >800 ml (keputusan user 20 Okt). KALIBRASI per pemasangan (tinggi kamera).
AREA_KECIL_TANGGUNG = 228_000
AREA_TANGGUNG_BESAR = 357_000
