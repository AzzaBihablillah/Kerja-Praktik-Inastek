"""RVM server: bottle + cap/no_cap detection with PLC conveyor control."""
from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
import snap7
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from snap7.util import set_bool
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OPEN_VINO = PROJECT_ROOT / "open_vino"
DATASETS = PROJECT_ROOT / "datasets"
sys.path.insert(0, str(OPEN_VINO))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "pipeline_cap"))   # cap_decider.py (pindah 21 Okt)

from cap_decider import CapDecider
from session_manager import SessionManager
from utils.tracker import BottleTracker
from utils.video_capture import VideoCapture

CAMERA_INDEX = 1
PLC_IP = "192.168.0.1"
BYTE_HMI_RUN = 10
BIT_HMI_RUN = 0
IDLE_TIMEOUT = 30
CONF_BOTTLE = 0.35
CONF_CAP = 0.25
# Ujung frame sering menghasilkan false positive no_cap. Pertahankan label
# cap sampai dukungan no_cap benar-benar kuat dan konsisten.
CAP_WIN = 15
CAP_FLIP_FRAC = 0.90
CAP_NECK_SIDE = "none"

BOTTLE_MODEL = PROJECT_ROOT / "export_model" / "bottle_best_ver7.pt"
CAP_MODEL = DATASETS / "3_model_final" / "cap_v4_640.pt"   # model sama, lokasi baru (21 Okt)
CAMERA_MODEL_SIZE = 640

bottle_model = YOLO(str(BOTTLE_MODEL))
cap_model = YOLO(str(CAP_MODEL))
session_manager = SessionManager(points_per_bottle=50)

_event_queue: asyncio.Queue = None
_ws_clients: set[WebSocket] = set()
_stop_inference = threading.Event()
_frame_lock = threading.Lock()
_latest_jpeg: bytes = b""


class ConveyorController:
    def __init__(self):
        self.plc = snap7.client.Client()
        self.is_running = False
        self.last_detection_time = 0.0
        self.timer_task: asyncio.Task | None = None
        self._lock = threading.Lock()

    def connect(self):
        try:
            self.plc.connect(PLC_IP, 0, 1)
            print("PLC connected." if self.plc.get_connected() else "PLC connection failed.")
        except Exception as exc:
            print(f"PLC connection error: {exc}")

    def disconnect(self):
        if self.plc.get_connected():
            self.plc.disconnect()
            print("PLC disconnected.")

    def write_bit(self, value: bool):
        try:
            if not self.plc.get_connected():
                print("PLC is not connected; cannot write conveyor bit.")
                return
            data = bytearray(self.plc.mb_read(start=BYTE_HMI_RUN, size=1))
            set_bool(data, byte_index=0, bool_index=BIT_HMI_RUN, value=value)
            self.plc.mb_write(start=BYTE_HMI_RUN, size=1, data=data)
        except Exception as exc:
            print(f"PLC write error: {exc}")

    def mark_bottle_seen(self):
        with self._lock:
            self.last_detection_time = time.time()

    def idle_time(self) -> float:
        with self._lock:
            return time.time() - self.last_detection_time

    async def start(self):
        if self.is_running:
            return
        self.write_bit(True)
        self.is_running = True
        self.mark_bottle_seen()
        self.timer_task = asyncio.create_task(self._monitor_idle())
        await broadcast_event({"type": "conveyor_status", "status": "running"})

    async def stop(self, reason="manual"):
        # Selalu tulis OFF saat shutdown, termasuk jika state lokal sudah reset
        # tetapi bit PLC sebelumnya masih tertinggal ON.
        self.write_bit(False)
        self.is_running = False
        if self.timer_task:
            self.timer_task.cancel()
            self.timer_task = None
        await broadcast_event({"type": "conveyor_status", "status": "stopped", "reason": reason})

    async def _monitor_idle(self):
        try:
            while self.is_running:
                await asyncio.sleep(1)
                idle = self.idle_time()
                await broadcast_event({"type": "conveyor_timer", "idle_time": int(idle), "max_idle": IDLE_TIMEOUT})
                if idle >= IDLE_TIMEOUT:
                    await self.stop(reason="idle_timeout")
                    if session_manager.is_active:
                        summary = session_manager.end_session().to_dict()
                        await broadcast_event({"type": "session_end", "summary": summary, "reason": "idle_timeout"})
                    break
        except asyncio.CancelledError:
            pass


conveyor = ConveyorController()


def push_event(loop: asyncio.AbstractEventLoop, event: dict):
    loop.call_soon_threadsafe(_event_queue.put_nowait, event)


def detections(result):
    if result is None or not result.boxes:
        return []
    return [
        (box.xyxy[0].cpu().numpy(), int(box.cls), float(box.conf))
        for box in result.boxes
    ]


def bottle_boxes(result):
    return [box for box, cls, _conf in detections(result) if cls == 0]


def find_cap_for_bottle(bottle_box, cap_results):
    candidates = []
    x1, y1, x2, y2 = bottle_box
    for box, cls, conf in cap_results:
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            candidates.append((conf, cls))
    return max(candidates, default=(0.0, None))


def _inference_loop(loop: asyncio.AbstractEventLoop):
    tracker = BottleTracker(iou_threshold=0.2, max_lost=12, min_frames_seen=1)
    decider = CapDecider(win=CAP_WIN, flip_frac=CAP_FLIP_FRAC, neck_side=CAP_NECK_SIDE)
    cap = None
    try:
        cap = VideoCapture(CAMERA_INDEX)
        frame_height = int(cap.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_width = int(cap.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        boundary = np.array([frame_width * 0.35, frame_width * 0.65], dtype=np.int32)

        with cap:
            while not _stop_inference.is_set():
                try:
                    frame = cap.read_frame(timeout=2.0)
                except RuntimeError as exc:
                    push_event(loop, {"type": "error", "message": str(exc)})
                    break

                bottle_result = bottle_model.predict(frame, imgsz=CAMERA_MODEL_SIZE, conf=CONF_BOTTLE, verbose=False)[0]
                cap_result = cap_model.predict(frame, imgsz=CAMERA_MODEL_SIZE, conf=CONF_CAP, verbose=False)[0]
                bottles = bottle_boxes(bottle_result)
                raw_caps = detections(cap_result)
                kept_caps = decider.step(bottles, raw_caps)
                voted_caps = {slot: (voted, confidence) for slot, _box, voted, confidence, _raw in kept_caps}

                zone_dets = []
                for bottle_index, bottle_box in enumerate(bottles):
                    center_x = (bottle_box[0] + bottle_box[2]) / 2
                    if not boundary[0] < center_x < boundary[1]:
                        continue
                    # Cap model: 0 = cap, 1 = no_cap. Business rule: no_cap is valid.
                    # Tanpa hasil cap yang valid, jangan anggap objek sebagai no_cap.
                    # Tunggu deteksi no_cap eksplisit agar non-bottle tidak diterima.
                    cap_class, confidence = voted_caps.get(bottle_index, (0, CONF_CAP))
                    decision_class = 1 - cap_class
                    zone_dets.append(np.array([*bottle_box, confidence, decision_class], dtype=np.float32))

                zone_array = np.stack(zone_dets) if zone_dets else None
                active_tracks, completed_tracks = tracker.update(zone_array)
                conveyor.mark_bottle_seen() if bottles else None

                annotated = frame.copy()
                cv2.line(annotated, (boundary[0], 0), (boundary[0], frame_height), (200, 200, 200), 1)
                cv2.line(annotated, (boundary[1], 0), (boundary[1], frame_height), (200, 200, 200), 1)
                for track in active_tracks:
                    x1, y1, x2, y2 = track.bbox.astype(int)
                    color = (0, 255, 0) if track.class_id == 0 else (0, 0, 255)
                    label = "NO_CAP" if track.class_id == 0 else "CAP"
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"#{track.id} {label} {track.conf * 100:.0f}%", (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                cv2.putText(annotated, f"GATE+VOTE | {len(kept_caps)} kept / {len(bottles)} bottle", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    with _frame_lock:
                        global _latest_jpeg
                        _latest_jpeg = encoded.tobytes()

                for track in completed_tracks:
                    if not session_manager.is_active:
                        continue
                    decision = tracker.final_decision(track)
                    bottle = session_manager.add_bottle(track_id=track.id, decision=decision, confidence=track.conf)
                    push_event(loop, {"type": "bottle_done", "bottle": bottle.to_dict()})

                if session_manager.is_active:
                    state = session_manager.get_live_state()
                    push_event(loop, {"type": "session_update", "total_valid": state["total_valid"], "total_invalid": state["total_invalid"], "total_points": state["total_points"]})
    finally:
        if cap is not None:
            cap.stop()


async def broadcast_event(event: dict):
    payload = json.dumps(event, ensure_ascii=False)
    dead = []
    for ws in list(_ws_clients):
        try:
            await ws.send_text(payload)
        except Exception:
            dead.append(ws)
    for ws in dead:
        _ws_clients.discard(ws)


async def broadcast_loop():
    while True:
        await broadcast_event(await _event_queue.get())


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _event_queue
    _event_queue = asyncio.Queue()
    conveyor.connect()
    loop = asyncio.get_running_loop()
    _stop_inference.clear()
    thread = threading.Thread(target=_inference_loop, args=(loop,), daemon=True)
    thread.start()
    broadcast_task = asyncio.create_task(broadcast_loop())
    print("RVM cap server: http://localhost:8000")
    try:
        yield
    finally:
        _stop_inference.set()
        await asyncio.to_thread(thread.join, 5.0)
        broadcast_task.cancel()
        try:
            await broadcast_task
        except asyncio.CancelledError:
            pass
        await conveyor.stop(reason="server_shutdown")
        conveyor.disconnect()


app = FastAPI(title="RVM Cap Test", lifespan=lifespan)
STATIC_DIR = OPEN_VINO / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    _ws_clients.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "init", "state": session_manager.get_live_state(), "conveyor_running": conveyor.is_running}))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _ws_clients.discard(ws)


@app.post("/session/start")
async def start_session():
    if session_manager.is_active:
        return JSONResponse({"ok": False, "message": "Sesi sudah aktif."}, status_code=400)
    session_id = session_manager.start_session()
    await conveyor.start()
    await broadcast_event({"type": "session_start", "session_id": session_id})
    return {"ok": True, "session_id": session_id}


@app.post("/session/end")
async def end_session():
    if not session_manager.is_active:
        return JSONResponse({"ok": False, "message": "Tidak ada sesi aktif."}, status_code=400)
    await conveyor.stop(reason="session_ended")
    summary = session_manager.end_session().to_dict()
    await broadcast_event({"type": "session_end", "summary": summary, "reason": "manual"})
    return {"ok": True, "summary": summary}


@app.get("/session/current")
async def current_session():
    return session_manager.get_live_state()


@app.post("/conveyor/start")
async def start_conveyor():
    await conveyor.start()
    return {"ok": True, "message": "Conveyor dimulai."}


@app.post("/conveyor/stop")
async def stop_conveyor():
    await conveyor.stop(reason="manual")
    return {"ok": True, "message": "Conveyor dihentikan."}


@app.get("/health")
async def health():
    return {"status": "ok", "session_active": session_manager.is_active, "conveyor_ready": conveyor.plc.get_connected(), "ws_clients": len(_ws_clients)}


@app.get("/video_feed")
async def video_feed():
    async def generate():
        while True:
            with _frame_lock:
                frame = _latest_jpeg
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.033)
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace; boundary=frame")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
