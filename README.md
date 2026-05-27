# Reverse Vending Machine - Sistem Deteksi Botol PET

Proyek capstone sistem Reverse Vending Machine (RVM) yang secara otomatis menerima, memvalidasi, dan menyimpan botol plastik PET. Sistem menjalankan deteksi objek berbasis YOLOv8 (dioptimasi dengan OpenVINO) dan menyajikan hasilnya lewat web interface real-time.

---

## Daftar Isi

- [Gambaran Sistem](#gambaran-sistem)
- [Struktur Folder](#struktur-folder)
- [Prasyarat](#prasyarat)
- [Instalasi](#instalasi)
- [Menjalankan Aplikasi](#menjalankan-aplikasi)
- [Web Interface](#web-interface)
- [Arsitektur Kode](#arsitektur-kode)
- [Model AI](#model-ai)
- [Komunikasi Hardware](#komunikasi-hardware)
- [Ekspor Model Baru](#ekspor-model-baru)
- [Troubleshooting](#troubleshooting)

---

## Gambaran Sistem

Alur kerja RVM:

1. User menekan tombol start - gate membuka, conveyor mulai bergerak
2. User meletakkan botol (barcode menghadap atas) di conveyor
3. Kamera di atas conveyor melakukan scanning real-time:
   - Deteksi apakah objek adalah botol PET (YOLO)
   - Baca barcode untuk identifikasi merek dan poin reward (opsional, future)
4. Bucket determiner menerima botol dan bertindak berdasarkan hasil kamera:
   - Tidak valid (bukan PET) - miring ke jalur penolakan, user mengambil kembali
   - Valid - timbang via load cell di bucket
5. Load cell: berat <= threshold (kosong) - masuk storage; berat > threshold (ada cairan) - tolak
6. Sistem mencatat merek dan mengakumulasi poin reward per sesi
7. Sesi berakhir saat user menekan stop atau tidak ada botol baru selama 30 detik

### Scope Capstone 1 (saat ini)

Hanya fitur **deteksi botol/non-botol** yang aktif. Fitur cap detection, brand classification, dan condition check belum diaktifkan.

---

## Struktur Folder

```
Kerja-Praktik-Inastek/
├── open_vino/                  # Aplikasi utama
│   ├── main.py                 # Loop inferensi standalone (OpenCV window)
│   ├── server.py               # FastAPI server + web interface
│   ├── session_manager.py      # State sesi, logika poin, alasan invalid
│   ├── models/
│   │   ├── bottle/             # Model deteksi botol PET
│   │   ├── brand/              # Model klasifikasi merek (belum aktif)
│   │   └── condition/          # Model kondisi botol (belum aktif)
│   ├── static/
│   │   └── index.html          # Frontend web interface
│   └── utils/
│       ├── run_models.py       # Wrapper inferensi multi-model OpenVINO
│       ├── tracker.py          # IoU tracker dengan velocity prediction
│       ├── video_capture.py    # Thread-safe camera reader
│       ├── serial_port.py      # Komunikasi serial ke mikrokontroler
│       ├── preprocess.py       # Letterbox, normalisasi gambar
│       ├── postprocess.py      # Inverse letterbox, xy_center
│       └── detector_utils.py   # Non-max suppression
├── export_model/               # Skrip ekspor model .pt ke OpenVINO
├── requirements.txt
└── README.md
```

---

## Prasyarat

- Python 3.10+
- Kamera (webcam atau USB camera)
- Mikrokontroler via USB serial (opsional, untuk hardware)

---

## Instalasi

### 1. Clone repository

```bash
git clone <url-repo>
cd Kerja-Praktik-Inastek
```

### 2. Buat virtual environment

```bash
python -m venv venv
```

Aktifkan:

- Windows: `venv\Scripts\activate`
- Linux/macOS: `source venv/bin/activate`

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> Untuk ekspor model saja (opsional), uncomment baris `ultralytics` di `requirements.txt` lalu install ulang.

---

## Menjalankan Aplikasi

> Semua perintah dijalankan dari folder `open_vino/` karena path model dibaca relatif terhadap direktori kerja.

### Mode Web Interface (direkomendasikan)

```bash
cd open_vino
python server.py
```

Buka browser ke `http://localhost:8000`.

### Mode Standalone OpenCV (tanpa web)

```bash
cd open_vino
python main.py
```

Jendela OpenCV terbuka. Tekan `q` untuk keluar.

---

## Web Interface

Interface berbasis browser dengan tiga tampilan:

**Idle** - Preview kamera langsung, tombol MULAI SESI. Garis abu-abu di preview menunjukkan batas zona deteksi aktif (35%-65% lebar frame).

**Sesi Aktif** - Layout dua kolom:
- Kiri: preview kamera real-time dengan bounding box dan label per track
- Kanan: scoreboard (botol diterima, poin, botol ditolak) dan daftar hasil botol

**Ringkasan Sesi** - Total botol, total poin, dan detail per botol setelah sesi diakhiri.

### Teknologi

| Komponen | Teknologi |
|---|---|
| Backend | FastAPI + Uvicorn |
| Real-time update | WebSocket |
| Camera stream | MJPEG (`/video_feed` endpoint) |
| Frontend | HTML + Tailwind CSS (CDN) + Alpine.js (CDN) |
| Penyimpanan | In-memory (per sesi) |

Tidak ada build step untuk frontend - semua berjalan langsung dari file statis.

### Endpoint API

| Method | Path | Fungsi |
|---|---|---|
| GET | `/` | Halaman utama |
| WS | `/ws` | WebSocket event stream |
| GET | `/video_feed` | MJPEG camera stream |
| POST | `/session/start` | Mulai sesi baru |
| POST | `/session/end` | Akhiri sesi, kembalikan ringkasan |
| GET | `/session/current` | State sesi saat ini (JSON) |
| GET | `/health` | Status sistem |

### Event WebSocket (server ke browser)

```jsonc
{ "type": "session_start", "session_id": "..." }
{ "type": "session_update", "total_valid": 3, "total_invalid": 1, "total_points": 150,
  "recent_bottles": [...] }   // recent_bottles disertakan tiap ~10 frame
{ "type": "bottle_done", "bottle": { "track_id": 4, "decision": "valid",
  "display_name": "Botol #4", "points": 50, "confidence": 0.94, ... } }
{ "type": "session_end", "summary": { ... } }
```

---

## Arsitektur Kode

### Pipeline Inferensi

```
server.py / main.py
  -> Models.predict(frame_bgr, model_index)
       -> letterbox()              # resize + pad ke 640x640
       -> preprocess_yolov8()      # BGR->RGB, normalize, transpose
       -> OpenVINO inference
       -> non_max_suppression()    # filter bbox
       -> inverse_letterbox()      # kembalikan koordinat ke ukuran asli
  -> BottleTracker.update(detections)
       -> IoU matching dengan velocity prediction
       -> returns active_tracks, completed_tracks
  -> SessionManager.add_bottle()   # saat track selesai
```

### Kelas Utama

**`Models`** (`utils/run_models.py`) - Muat satu atau lebih model OpenVINO dan jalankan inferensi. `predict(image, model_index)` mengembalikan array `(n, 6)` format `[x1, y1, x2, y2, conf, class_id]` dalam koordinat gambar asli, atau `None`.

**`BottleTracker`** (`utils/tracker.py`) - IoU tracker dengan velocity prediction. Melacak botol antar frame, tahan terhadap jeda deteksi singkat. Parameter penting:

```python
tracker = BottleTracker(
    iou_threshold=0.3,    # min IoU untuk match track-detection
    max_lost=20,          # frame tanpa deteksi sebelum track dianggap selesai (~0.67s @ 30fps)
    min_frames_seen=5,    # min frame terlihat agar track dihitung sebagai botol nyata
)
```

**`SessionManager`** (`session_manager.py`) - Thread-safe state manager satu sesi. Menyimpan daftar `BottleResult`, menghitung poin, dan menyediakan `get_live_state()` untuk dikirim via WebSocket.

**`VideoCapture`** (`utils/video_capture.py`) - Thread-safe camera reader dengan background thread dan `Queue(maxsize=1)` agar selalu mengembalikan frame terbaru.

### Zona Deteksi

Deteksi yang titik tengah bounding box-nya berada di luar zona tengah frame diabaikan:

```python
x_center_boundary = [frame_width * 0.35, frame_width * 0.65]
# hanya deteksi di 35%-65% lebar frame yang diproses
```

### Alasan Invalid (expandable)

| Kode | Sumber | Status |
|---|---|---|
| `bukan_botol_pet` | Model bottle detection | Aktif |
| `berisi_cairan` | Load cell (hardware) | Belum terintegrasi |
| `tutup_terpasang` | Model cap detection | Belum ada model |
| `botol_rusak` | Model condition | Model ada, belum diaktifkan |

### Identitas Botol (display_name)

Setiap botol memiliki `display_name` dengan fallback chain:
1. SKU dari barcode scanner - lookup di `SKU_CATALOG` (`session_manager.py`)
2. SKU ada tapi tidak dikenal - tampilkan SKU mentah
3. Brand dari visual model
4. Fallback: `Botol #ID`

Untuk menambahkan produk, isi `SKU_CATALOG` di `session_manager.py`:

```python
SKU_CATALOG: Dict[str, str] = {
    "8996001101055": "Aqua 600ml",
    "8998007300022": "Le Mineral 600ml",
}
```

---

## Model AI

Semua model menggunakan YOLOv8n, input 640x640, batch 1.

| Folder | Kelas | Status |
|---|---|---|
| `models/bottle/` | `bottle`, `not_bottle` | Aktif (Capstone 1) |
| `models/brand/` | `aqua`, `coca_cola`, `le_mineral`, `unknown` | Tersedia, belum diaktifkan |
| `models/condition/` | `Deformed`, `Good` | Tersedia, belum diaktifkan |

---

## Komunikasi Hardware

Komunikasi ke mikrokontroler (STM32 / smart relay) via serial menggunakan class `Micro`:

```python
from utils.serial_port import Micro, Command

micro = Micro()
micro.open("COM3")                        # sesuaikan port
micro.send_command(Command.BOTTLE)        # botol valid -> load cell
micro.send_command(Command.TOLAK_BENDA)   # botol invalid -> tolak
micro.close()
```

Perintah yang tersedia:

| Enum | Karakter | Fungsi |
|---|---|---|
| `Command.BENDA_MASUK` | `A` | Sinyal objek masuk |
| `Command.TOLAK_BENDA` | `B` | Tolak / jalur penolakan |
| `Command.BOTTLE` | `C` | Terima botol, kirim ke load cell |

Untuk mengaktifkan hardware, uncomment baris `micro.send_command(...)` di `server.py`.

---

## Ekspor Model Baru

Untuk melatih model baru dan mengekspornya dari `.pt` ke format OpenVINO:

1. Letakkan file `.pt` hasil training di `export_model/`
2. Edit `model_path` di `export_model/export_model.py`
3. Jalankan:

```bash
cd export_model
python export_model.py
```

4. Pindahkan folder `*_openvino_model/` yang dihasilkan ke `open_vino/models/<nama>/`

> Catatan hardware: OpenVINO berjalan optimal di CPU/GPU Intel. Untuk deployment ke Jetson Nano (ARM + NVIDIA), gunakan `utils/run_models_trt.py` dengan TensorRT sebagai backend inferensi - web interface dan session manager tidak perlu diubah.

---

## Troubleshooting

| Masalah | Solusi |
|---|---|
| `Failed to open video source` | Ganti indeks kamera di `server.py` atau `main.py` (coba `0`, `1`, `2`) |
| `Failed to compile model on device 'CPU'` | Pastikan OpenVINO terinstall: `pip install openvino` |
| `ModuleNotFoundError: No module named 'utils'` | Jalankan dari folder `open_vino/`, bukan dari root |
| Port serial tidak terdeteksi | Jalankan `micro.list_serial_port()` untuk melihat daftar port aktif |
| Web interface tidak terbuka | Gunakan `http://localhost:8000`, bukan `http://0.0.0.0:8000` |
| Daftar botol tidak update saat live | Pastikan sesi sudah dimulai (tombol MULAI SESI), daftar update tiap ~10 frame |
