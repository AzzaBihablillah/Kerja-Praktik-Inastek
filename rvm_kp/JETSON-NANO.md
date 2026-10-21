# Memasang sistem RVM KP di Jetson Nano Developer Kit B01 (4 GB)

Board: P3450, revisi B01 (180-13449-DAAF-B01), 4 GB (945-13450-0000-100). GPU Maxwell 128 core, CPU 4× Cortex-A57, RAM 4 GB dipakai bersama CPU+GPU.
Batas perangkat lunak: **JetPack 4.6.x** (Ubuntu 18.04, Python 3.6, CUDA 10.2, TensorRT 8.2). Ultralytics modern butuh Python ≥ 3.8 → kode dijalankan **di dalam Docker**.
Jika langkah ini gagal total, sistem dijalankan di laptop dengan kode yang sama (`--profil laptop` atau `nano`).

## 0. Perangkat keras (sebelum menyalakan)
- **Daya: adaptor barrel 5V 4A + pasang jumper J48** (tanpa J48 daya diambil dari micro USB 2A → board melambat/mati saat tiga model jalan).
- Kipas di heatsink (mode 10 W terus-menerus panas).
- microSD ≥ 64 GB kelas A1/U3 (image Docker Ultralytics beberapa GB).
- Kamera C922 di port USB 3.0; jaringan LAN (untuk PLC & salin file).

## 1. Flash JetPack
Unduh image SD card "Jetson Nano Developer Kit" JetPack 4.6.x dari NVIDIA (developer.nvidia.com/embedded/jetpack-archive), tulis ke microSD dengan balenaEtcher, boot, selesaikan setup awal.
```bash
cat /etc/nv_tegra_release          # harus R32 (revision 7.x = JetPack 4.6.x)
```

## 2. Mode daya penuh
```bash
sudo nvpmodel -m 0                 # MAXN 10 W
sudo jetson_clocks                 # kunci clock maksimum (ulang setiap boot, atau pasang di rc.local)
```

## 3. Lega-kan RAM (4 GB dipakai bersama GPU)
```bash
sudo systemctl set-default multi-user.target    # boot tanpa desktop GUI (hemat ±1 GB); kembali: graphical.target
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile && sudo mkswap /swapfile && sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
sudo reboot
```

## 4. Salin paket dari laptop
Di laptop: `venv/Scripts/python.exe -m rvm_kp.buat_paket_nano` → folder `datasets/_paket_nano/` (±150 MB: kode, 3 model .pt, ONNX opset 12, 40 foto uji). Salin isinya ke Nano, misalnya `~/rvm` (flashdisk atau `scp -r`).

## 5. Docker Ultralytics untuk JetPack 4
```bash
sudo docker pull ultralytics/ultralytics:latest-jetson-jetpack4
sudo docker run -it --name rvm --ipc=host --runtime=nvidia --network host \
  --device /dev/video0 -v ~/rvm:/rvm -w /rvm ultralytics/ultralytics:latest-jetson-jetpack4
# di dalam container:
pip install -r rvm_kp/requirements-jetson.txt     # hanya psutil, fastapi, uvicorn, python-snap7
python -m rvm_kp.cek_perangkat                    # baca bagian "Langkah berikutnya"
```
Masuk lagi nanti: `sudo docker start -ai rvm`. Jangan upgrade torch/numpy/opencv di dalam container.

## 6. Buat engine TensorRT FP16 (di Nano, sekali saja)
```bash
python -m rvm_kp.export_models --format engine --half --workspace 1
```
±10–20 menit per model (4 model: botol 640, tutup full 640, tutup ROI, cairan 224). Opset ONNX 12 dipakai agar cocok dengan TensorRT 8.2.
Jika ekspor Ultralytics gagal, cadangan dengan `trtexec` dari ONNX yang ikut paket:
```bash
/usr/src/tensorrt/bin/trtexec --onnx=datasets/3_model_final/ekspor/botol_ver8b.onnx --saveEngine=/tmp/botol.engine --fp16 --workspace=1024
```
(engine trtexec tidak membawa metadata Ultralytics; laporkan dulu sebelum memakainya.)

## 7. Ukur dan uji
```bash
python -m rvm_kp.benchmark --backend tensorrt pytorch --profil nano --n 40   # tensorrt vs pytorch GPU Nano, termasuk kesepakatan keputusan
python - <<'EOF'
from rvm_kp import RVMAnalyzer
from rvm_kp.kamera import Kamera
an = RVMAnalyzer("tensorrt", profile="nano")
with Kamera(0, 1280, 720) as cam:
    for _ in range(30):
        print([(r.cap, round(r.p_liquid, 2), r.size) for r in an.analyze(cam.read())], an.last_timings["total"])
EOF
```
Hasil benchmark tersimpan di `datasets/4_eksperimen/benchmark/benchmark_<tanggal>_jetson_nano_*.txt` → salin balik ke laptop untuk laporan.

## PLC (python-snap7)
Wheel aarch64 python-snap7 mungkin tidak cocok dengan glibc Ubuntu 18.04. Jika `import snap7` gagal: (a) bangun pustaka snap7 dari sumber (snap7 1.4.2, `make -f arm_v7_linux.mk` / aarch64) lalu `pip install python-snap7==1.3`; atau (b) biarkan **laptop** yang memegang PLC & UI, Nano hanya mengirim hasil analisis lewat jaringan. Pilihan (b) lebih sederhana dan cukup untuk demo.

## Yang belum diketahui (catat hasilnya di docs/CATATAN-PERCOBAAN-TA.md)
- Apakah image Docker JetPack 4 masih mendukung Ultralytics 8.4.x dan ekspor engine TensorRT 8.2 tanpa galat.
- Kecepatan nyata per frame (perkiraan kasar sebelum diukur: ratusan ms per frame) dan pemakaian RAM.
- Suhu/throttling saat jalan lama (`tegrastats`).
