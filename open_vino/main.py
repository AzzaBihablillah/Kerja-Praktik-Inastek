print("Importing dependecies...")

from utils.inference_classification import Classify
from utils.inference_object_detection import Detect
from utils.preprocess import preserve_aspect_ratio_resize, pad_to_square
import os
from utils.serial_port import Micro
import cv2
import time
from datetime import date, datetime
import threading
import queue
from utils.run_models import Models, MODEL_DETECTION_TYPE, MODEL_CLASSIFICATION_TYPE, MODEL_SEGMENTATION_TYPE
import numpy as np
from datetime import datetime
from copy import deepcopy
import time
import logging

ROOT = os.getcwd()

logger = logging.getLogger(__name__)

print("Preparing model...")



models = Models(
    model_paths = [
            os.path.join(
            ROOT,
            "models",
            "bottle",
            "bottle_best_ver7.xml",
            ),
        ],
    class_names=[
        ["bottle", "not_bottle"],
    ],
    types=MODEL_DETECTION_TYPE

)


class VideoCapture:
    def __init__(self, source: int | str):
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")

        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._q    = queue.Queue(maxsize=1)
        self._stop = threading.Event()

        self._thread        = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _reader(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()
            if not ret:
                logger.warning("Failed to read frame, stopping capture thread.")
                self._stop.set()
                break

            # Discard stale frame and replace with latest
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put(frame)

        self.cap.release()
        logger.info("Camera released.")

    def read_frame(self, timeout: float = 2.0):
        """
        Returns the latest frame.

        :param timeout: Seconds to wait before raising RuntimeError.
        :raises RuntimeError: If no frame is available within timeout or camera stopped.
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
        """Signal the reader thread to stop cleanly."""
        self._stop.set()
        self._thread.join(timeout=3.0)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False
    

if __name__ == "__main__":

    # model_img_size = 640
    # model_index = 0
    # micro = Micro()

    capture = VideoCapture(1)
    frame_height = capture.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frame_width = capture.cap.get(cv2.CAP_PROP_FRAME_WIDTH)

    # print(frame_height, frame_width)

    while True:
        frame_bgr = capture.read_frame()
        result = models.predict(frame_bgr, model_index=0)
        
        if result is not None:
            # print(result.shape)
            for i in range(result.shape[0]):
                annotated_frame = cv2.rectangle(
                                                frame_bgr, 
                                                (int(result[i,0]), int(result[i,1])), 
                                                (int(result[i,2]), int(result[i,3])), 
                                                (0,0,255), 
                                                2
            )
                annotated_frame = cv2.putText(
                                            annotated_frame, 
                                            # f'{models.class_names[i][int(result[i][5])]}: {int(result[i][4]*100)} %', 
                                            f'{models.class_names[0][int(result[i,5])]}: {result[i,4]}',
                                            (int(result[i,0]), int(result[i,1]) - 10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 
                                            0.6, 
                                            (0, 0, 255), 
                                            2
                                    )
                
        



        cv2.imshow(f"test", frame_bgr)

        if cv2.waitKey(1) == ord('q'):
            break