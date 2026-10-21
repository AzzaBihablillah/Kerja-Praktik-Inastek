# rvm_kp/ — sistem analisis RVM untuk pekerjaan KP (satu kode, beberapa perangkat)

Tahap per botol: **botol** (`botol_ver8b`) → **tutup** (`cap_belt_final`) → **cairan berwarna** (`cairan_RH`, crop lipat 224 px) → **ukuran** (luas kotak).
Lingkup cairan: berwarna saja; air bening di luar lingkup. Bukan pengganti `open_vino/` (aplikasi Capstone 1 tim). Kode kompatibel Python 3.8.

## Perangkat target: Jetson Nano Developer Kit B01 4 GB
Panduan pemasangan lengkap: **`JETSON-NANO.md`** (JetPack 4.6.x + Docker Ultralytics JetPack 4 + engine TensorRT FP16).

## Profil (`config.PROFILES`, default `nano`)
| profil | resolusi kerja | deteksi tutup | presisi | GFLOPs/frame (1 botol) |
|---|---|---|---|---|
| `nano` | 1280 px (720p) | hanya di area tiap botol, kanvas 320 | FP16 | 10,7 |
| `laptop` | 1920 px (1080p) | seluruh frame 640 | FP32 | 16,8 |
Uji 217 foto belt (21 Okt): profil nano memberi keputusan cairan 217/217 dan ukuran 216/217 sama dengan profil laptop; akurasi tutup sama (lihat `docs/CATATAN-PERCOBAAN-TA.md` L-16).

## Backend
| perangkat | backend | model |
|---|---|---|
| Jetson Nano | `tensorrt` | `ekspor/*.engine` (DIBUAT di Nano) |
| laptop RTX 4050 | `pytorch` (CUDA) | `datasets/3_model_final/*.pt` |
| PC Intel tanpa GPU NVIDIA | `openvino` | `ekspor/*_openvino_model/` |
| (tidak disarankan, lambat lewat Ultralytics) | `onnx` | `ekspor/*.onnx` (opset 12) |

```python
from rvm_kp import RVMAnalyzer
from rvm_kp.kamera import Kamera
an = RVMAnalyzer("tensorrt", profile="nano")      # laptop: RVMAnalyzer("pytorch", profile="nano")
with Kamera(0, 1280, 720) as cam:
    hasil = an.analyze(cam.read())                 # list BottleResult: box, cap, cap_conf, p_liquid, has_liquid, size, area_1920
    print(an.last_timings)                         # ms per tahap: pra, botol, tutup, cairan, total
```

## Perintah (dari folder proyek)
| perintah | fungsi |
|---|---|
| `python -m rvm_kp.cek_perangkat` | cek kesiapan perangkat + daftar langkah yang kurang |
| `python -m rvm_kp.export_models --format onnx openvino` | ekspor di laptop (ONNX opset 12) |
| `python -m rvm_kp.export_models --format engine --half --workspace 1` | buat engine TensorRT di Nano |
| `python -m rvm_kp.benchmark --backend pytorch openvino --profil nano` | kecepatan + kesepakatan keputusan |
| `python -m rvm_kp.uji_profil` | uji pengaruh resolusi / mode tutup / FP16 di 217 foto belt |
| `python -m rvm_kp.buat_paket_nano` | folder salin ke Nano: `datasets/_paket_nano/` |
Di laptop pakai `venv/Scripts/python.exe` dan env `YOLO_AUTOINSTALL=False` (cegah Ultralytics memasang paket sendiri).
Laporkan angka kecepatan dari perangkat yang benar-benar diuji (`datasets/4_eksperimen/benchmark/`).

## Kalibrasi saat pemasangan (`config.py`)
- `LIQUID_THR`: potret beberapa botol berisi & kosong di lokasi, pilih ambang yang memisahkan keduanya.
- `AREA_KECIL_TANGGUNG`, `AREA_TANGGUNG_BESAR` (setara frame 1920 px; resolusi lain dinormalisasi otomatis): potret botol 330/350, 390/600, 1000/1500 ml; letakkan batas di antara nilai `area_1920`-nya.

## Keterbatasan diketahui
- Model tutup gagal pada **tutup hitam** (Coca-Cola Zero, Powerade): uji independen 71%, tanpa kedua merek ±94% (CATATAN C-9).
- Kecepatan Jetson Nano belum diukur.
