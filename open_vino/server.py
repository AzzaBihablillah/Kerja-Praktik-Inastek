"""
FastAPI server untuk RVM web interface.

Menjalankan:
    cd open_vino
    python server.py
    # lalu buka http://localhost:8000 di browser

Arsitektur thread:
    - Main thread   : FastAPI / Uvicorn (event loop asyncio)
    - Thread 1      : VideoCapture background reader (sudah ada di main.py)
    - Thread 2      : Inference loop  ← dijalankan oleh server ini

Komunikasi inference → FastAPI:
    inference thread push event ke `_event_queue` (threading.Queue)
    FastAPI broadcast event tersebut ke semua WebSocket client yang terhubung.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

import numpy as np
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

# ── Import komponen RVM ───────────────────────────────────────────────────────
from session_manager import SessionManager
from utils.postprocess import xy_center
from utils.run_models import MODEL_DETECTION_TYPE, Models
from utils.tracker import BottleTracker
from utils.video_capture import VideoCapture

# Uncomment saat hardware terhubung:
# from utils.serial_port import Command, Micro

ROOT = os.getcwd()

# ── Konfigurasi model ─────────────────────────────────────────────────────────

models = Models(
    model_paths=[
        os.path.join(ROOT, "models", "bottle", "bottle_best_ver7.xml"),
    ],
    class_names=[["bottle", "not_bottle"]],
    types=MODEL_DETECTION_TYPE,
)

# ── Shared state ──────────────────────────────────────────────────────────────

session_manager = SessionManager(points_per_bottle=50)

# Queue untuk event dari inference thread → FastAPI broadcast
# Item adalah dict siap-JSON, contoh: {"type": "bottle_done", "bottle": {...}}
_event_queue: asyncio.Queue = None   # diinisialisasi di lifespan

# Set WebSocket connections yang aktif
_ws_clients: set[WebSocket] = set()
_ws_lock = threading.Lock()

# Flag untuk menghentikan inference thread
_stop_inference = threading.Event()

# Frame terbaru dari kamera (JPEG bytes) — dibaca oleh /video_feed endpoint
_frame_lock  = threading.Lock()
_latest_jpeg: bytes = b""

# ── Inference loop (berjalan di thread terpisah) ──────────────────────────────

def _inference_loop(loop: asyncio.AbstractEventLoop) -> None:
    """
    Loop deteksi botol.  Berjalan di background thread.
    Push event ke asyncio event loop via `loop.call_soon_threadsafe`.

    :param loop: event loop asyncio milik FastAPI, untuk enqueue events
    """
    import cv2

    def push_event(event: dict) -> None:
        """Kirim event ke asyncio queue secara thread-safe."""
        loop.call_soon_threadsafe(_event_queue.put_nowait, event)

    tracker     = BottleTracker(
        iou_threshold=0.3,
        max_lost=20,
        min_frames_seen=5,
    )
    frame_count = 0   # untuk throttle session_update

    try:
        cap = VideoCapture(0)
    except RuntimeError as e:
        push_event({"type": "error", "message": str(e)})
        return

    frame_height = int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    x_center_boundary = np.array(
        [frame_width * 0.35, frame_width - frame_width * 0.35]
    ).astype(np.int32)

    with cap:
        while not _stop_inference.is_set():
            try:
                frame_bgr = cap.read_frame(timeout=2.0)
            except RuntimeError:
                break

            result = models.predict(frame_bgr, model_index=0)

            # Filter ke zona tengah konveyor
            if result is not None:
                in_zone = [
                    result[i] for i in range(result.shape[0])
                    if x_center_boundary[0]
                    < xy_center(result[i, :4])[0]
                    < x_center_boundary[1]
                ]
                zone_dets = np.stack(in_zone) if in_zone else None
            else:
                zone_dets = None

            active_tracks, completed_tracks = tracker.update(zone_dets)

            # ── Annotasi frame + simpan sebagai JPEG untuk /video_feed ────
            global _latest_jpeg
            annotated = frame_bgr.copy()

            # Garis batas zona konveyor (abu-abu tipis)
            cv2.line(annotated, (x_center_boundary[0], 0),
                     (x_center_boundary[0], frame_height), (200, 200, 200), 1)
            cv2.line(annotated, (x_center_boundary[1], 0),
                     (x_center_boundary[1], frame_height), (200, 200, 200), 1)

            # Bounding box per track aktif
            for track in active_tracks:
                x1, y1, x2, y2 = track.bbox.astype(int)
                color  = (0, 255, 0) if track.class_id == 0 else (0, 0, 255)
                status = "VALID" if track.class_id == 0 else "INVALID"
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                label = f"#{track.id} {status} {track.conf * 100:.0f}%"
                (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(annotated, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

            # Encode ke JPEG (quality 70 — cukup untuk preview, hemat bandwidth)
            ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
            if ok:
                with _frame_lock:
                    _latest_jpeg = buf.tobytes()

            # ── Botol selesai ─────────────────────────────────────────────
            for track in completed_tracks:
                if not session_manager.is_active:
                    continue

                decision = tracker.final_decision(track)
                bottle   = session_manager.add_bottle(
                    track_id=track.id,
                    decision=decision,
                    confidence=track.conf,
                )

                # Kirim event ke WebSocket client
                push_event({
                    "type":   "bottle_done",
                    "bottle": bottle.to_dict(),
                })

                # Kirim command ke hardware (uncomment saat hardware terhubung)
                # if decision == 0:
                #     micro.send_command(Command.BOTTLE)
                # else:
                #     micro.send_command(Command.TOLAK_BENDA)

            # ── Update status sesi ───────────────────────────────────────
            # Selalu kirim counter (ringan). recent_bottles disertakan tiap
            # 10 frame (~3×/detik) sebagai sumber kebenaran tunggal daftar botol,
            # menghindari masalah reaktivitas Alpine dari incremental bottle_done.
            frame_count += 1
            if session_manager.is_active:
                state = session_manager.get_live_state()
                event = {
                    "type":          "session_update",
                    "total_valid":   state["total_valid"],
                    "total_invalid": state["total_invalid"],
                    "total_points":  state["total_points"],
                }
                if frame_count % 10 == 0:
                    event["recent_bottles"] = state["recent_bottles"]
                push_event(event)


# ── Lifespan: start/stop inference thread bersama server ─────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_queue
    _event_queue = asyncio.Queue()

    loop = asyncio.get_event_loop()
    _stop_inference.clear()

    t = threading.Thread(target=_inference_loop, args=(loop,), daemon=True)
    t.start()

    # Task untuk broadcast event dari queue ke semua WS clients
    broadcast_task = asyncio.create_task(_broadcast_loop())

    # Cetak URL yang benar — Uvicorn log menampilkan 0.0.0.0 (bind address),
    # tapi yang dipakai di browser adalah localhost
    print("\n" + "─" * 50)
    print("  ✅  RVM Interface siap")
    print("  🌐  Buka browser: http://localhost:8000")
    print("─" * 50 + "\n")

    yield   # ← server berjalan

    _stop_inference.set()
    broadcast_task.cancel()


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(title="RVM Interface", lifespan=lifespan)

# Sajikan file static (HTML/CSS/JS frontend) di /static
static_dir = os.path.join(ROOT, "static")
os.makedirs(static_dir, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ── Broadcast loop ────────────────────────────────────────────────────────────

async def _broadcast_loop() -> None:
    """Ambil event dari queue dan kirim ke semua WebSocket client."""
    while True:
        event = await _event_queue.get()
        payload = json.dumps(event, ensure_ascii=False)

        dead: list[WebSocket] = []
        for ws in list(_ws_clients):
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)

        for ws in dead:
            _ws_clients.discard(ws)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
async def index():
    """Sajikan halaman utama."""
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket endpoint — browser konek ke sini untuk menerima event real-time."""
    await ws.accept()
    _ws_clients.add(ws)
    try:
        # Langsung kirim state sesi saat ini supaya UI langsung sinkron
        await ws.send_text(json.dumps({
            "type":  "init",
            "state": session_manager.get_live_state(),
        }))
        # Jaga koneksi tetap hidup (browser akan terima push event dari broadcast_loop)
        while True:
            await ws.receive_text()   # tunggu pesan dari client (ping/keep-alive)
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


@app.post("/session/start")
async def start_session():
    """Mulai sesi baru."""
    if session_manager.is_active:
        return JSONResponse({"ok": False, "message": "Sesi sudah aktif."}, status_code=400)

    session_id = session_manager.start_session()

    # Broadcast ke semua client
    await _broadcast_event({"type": "session_start", "session_id": session_id})
    return {"ok": True, "session_id": session_id}


@app.post("/session/end")
async def end_session():
    """Akhiri sesi aktif dan kembalikan ringkasan."""
    if not session_manager.is_active:
        return JSONResponse({"ok": False, "message": "Tidak ada sesi aktif."}, status_code=400)

    summary = session_manager.end_session()
    summary_dict = summary.to_dict()

    await _broadcast_event({"type": "session_end", "summary": summary_dict})
    return {"ok": True, "summary": summary_dict}


@app.get("/session/current")
async def get_current_session():
    """State sesi saat ini (untuk REST polling, biasanya WebSocket lebih dipakai)."""
    return session_manager.get_live_state()


@app.get("/health")
async def health():
    """Status sistem — digunakan UI untuk cek apakah backend hidup."""
    return {
        "status":         "ok",
        "session_active": session_manager.is_active,
        "ws_clients":     len(_ws_clients),
    }


@app.get("/video_feed")
async def video_feed():
    """
    MJPEG stream dari kamera.  Digunakan sebagai src <img> di frontend.
    Contoh: <img src="/video_feed">
    """
    async def generate():
        while True:
            with _frame_lock:
                frame = _latest_jpeg
            if frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame +
                    b"\r\n"
                )
            await asyncio.sleep(0.033)   # ~30 fps maksimum

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ── Helper ────────────────────────────────────────────────────────────────────

async def _broadcast_event(event: dict) -> None:
    """Kirim event langsung (bukan lewat queue) dari coroutine FastAPI."""
    payload = json.dumps(event, ensure_ascii=False)
    dead: list[WebSocket] = []
    for ws in list(_ws_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,   # reload=True hanya untuk dev; matikan di produksi
    )
