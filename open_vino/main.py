print("Importing dependecies...")

from utils.inference_classification import Classify
from utils.inference_object_detection import Detect
from utils.preprocess import preserve_aspect_ratio_resize, pad_to_square
from utils.postprocess import xy_center
import os
from utils.serial_port import Micro, Command
import cv2
import time
from datetime import date, datetime
import threading
import queue
from utils.run_models import Models, MODEL_DETECTION_TYPE, MODEL_CLASSIFICATION_TYPE, MODEL_SEGMENTATION_TYPE
from utils.tracker import BottleTracker
from utils.video_capture import VideoCapture
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


if __name__ == "__main__":

    # model_img_size = 640
    # model_index = 0

    # ── Serial (uncomment ketika hardware terhubung) ─────────────────────
    # micro = Micro()
    # micro.open(0)   # sesuaikan port jika perlu

    capture = VideoCapture(0)
    frame_height = int(capture.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_width  = int(capture.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    x_center_boundary = np.array(
        [frame_width * 0.35, frame_width - frame_width * 0.35]
    ).astype(np.int32)

    tracker = BottleTracker(
        # Minimum IoU antara posisi prediksi track dan deteksi baru agar dianggap
        # objek yang sama.  Turunkan (mis. 0.2) kalau konveyor cepat / sering miss-match.
        iou_threshold=0.3,

        # Berapa frame berturut-turut tanpa deteksi sebelum track dianggap "selesai".
        # @ 30 fps: 20 frame ≈ 0.67 detik.
        # Naikkan kalau botol sering dihitung dua kali (gap deteksi terlalu lama).
        # Turunkan kalau track "hantu" terlalu lama bertahan setelah botol sudah pergi.
        max_lost=20,

        # Track yang total frame terlihatnya < nilai ini tidak dihitung sebagai botol.
        # Berguna menyaring noise / deteksi kilat bukan botol.
        # @ 30 fps: 5 frame ≈ 0.17 detik.
        # Naikkan kalau masih ada false positive; turunkan kalau botol cepat melintas
        # dan tidak sempat terhitung.
        min_frames_seen=5,
    )
    valid_color   = (0, 255, 0)   # hijau  — VALID
    invalid_color = (0, 0, 255)   # merah  — INVALID

    # Akumulasi sesi (bertambah ketika botol benar-benar keluar dari zona)
    session_valid   = 0
    session_invalid = 0

    while True:
        frame_bgr = capture.read_frame()
        result    = models.predict(frame_bgr, model_index=0)

        # Salinan frame bersih untuk semua anotasi
        annotated_frame = frame_bgr.copy()

        # Gambar garis batas zona konveyor (abu-abu tipis)
        cv2.line(annotated_frame,
                 (x_center_boundary[0], 0), (x_center_boundary[0], frame_height),
                 (200, 200, 200), 1)
        cv2.line(annotated_frame,
                 (x_center_boundary[1], 0), (x_center_boundary[1], frame_height),
                 (200, 200, 200), 1)

        # Filter deteksi ke zona tengah, lalu update tracker
        if result is not None:
            in_zone = [
                result[i] for i in range(result.shape[0])
                if x_center_boundary[0] < xy_center(result[i, :4])[0] < x_center_boundary[1]
            ]
            zone_dets = np.stack(in_zone) if in_zone else None
        else:
            zone_dets = None

        active_tracks, completed_tracks = tracker.update(zone_dets)

        # ── Tangani botol yang baru saja keluar dari zona ─────────────────
        for track in completed_tracks:
            decision = tracker.final_decision(track)   # 0=VALID, 1=INVALID

            if decision == 0:
                session_valid += 1
                # micro.send_command(Command.BOTTLE)         # ← aktifkan saat hardware terhubung
                print(
                    f"[SERIAL] #{track.id} VALID "
                    f"(vote: {track.frames_as_valid}v / {track.frames_as_invalid}x) "
                    f"→ kirim BOTTLE (load cell)"
                )
            else:
                session_invalid += 1
                # micro.send_command(Command.TOLAK_BENDA)    # ← aktifkan saat hardware terhubung
                print(
                    f"[SERIAL] #{track.id} INVALID "
                    f"(vote: {track.frames_as_valid}v / {track.frames_as_invalid}x) "
                    f"→ kirim TOLAK_BENDA (reject)"
                )

        # ── Gambar bounding box + label per track aktif ───────────────────
        frame_valid_count   = 0
        frame_invalid_count = 0
        for track in active_tracks:
            x1, y1, x2, y2 = track.bbox.astype(int)
            is_valid = track.class_id == 0
            color    = valid_color if is_valid else invalid_color
            status   = "VALID" if is_valid else "INVALID"

            if is_valid:
                frame_valid_count += 1
            else:
                frame_invalid_count += 1

            # Bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)

            # Label: "#ID  STATUS  conf%"
            label = f"#{track.id}  {status}  {track.conf * 100:.0f}%"
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            # Background solid warna di atas bbox
            cv2.rectangle(annotated_frame,
                          (x1, y1 - lh - 10), (x1 + lw + 6, y1),
                          color, -1)
            # Teks putih supaya kontras
            cv2.putText(annotated_frame, label,
                        (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # ── Panel ringkasan — pojok kiri atas ─────────────────────────────
        # Baris 1-2: botol yang terlihat sekarang (per-frame)
        # Baris 3-4: total botol yang sudah diproses sesi ini (akumulasi)
        cv2.rectangle(annotated_frame, (8, 8), (290, 110), (30, 30, 30), -1)

        cv2.putText(annotated_frame, "Sekarang:",
                    (14, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(annotated_frame,
                    f"  Valid: {frame_valid_count}   Invalid: {frame_invalid_count}",
                    (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        cv2.putText(annotated_frame, "Sesi total:",
                    (14, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(annotated_frame,
                    f"  Valid: {session_valid}   Invalid: {session_invalid}",
                    (14, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (200, 200, 200), 1)

        cv2.imshow("RVM - Bottle Detection", annotated_frame)

        if cv2.waitKey(1) == ord('q'):
            break
