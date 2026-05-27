"""
Thread-safe camera reader.

Menggunakan background thread dan Queue(maxsize=1) untuk selalu
mengembalikan frame terbaru — frame lama dibuang otomatis.
"""

from __future__ import annotations

import logging
import queue
import threading

import cv2

logger = logging.getLogger(__name__)


class VideoCapture:
    def __init__(self, source: int | str):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._q    = queue.Queue(maxsize=1)
        self._stop = threading.Event()

        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame, stopping capture thread.")
                self._stop.set()
                break
            # Buang frame lama, simpan yang terbaru
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put(frame)
        self.cap.release()
        logger.info("Camera released.")

    def read_frame(self, timeout: float = 2.0):
        """
        Kembalikan frame terbaru.

        :raises RuntimeError: jika tidak ada frame dalam timeout atau kamera berhenti.
        """
        if self._stop.is_set() and self._q.empty():
            raise RuntimeError("Camera has stopped and no frames are available.")
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            raise RuntimeError(f"No frame received within {timeout}s timeout.")

    def is_running(self) -> bool:
        return not self._stop.is_set()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=3.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False
