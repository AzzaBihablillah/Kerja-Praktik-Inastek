"""
IoU-based bottle tracker with velocity prediction.

No external dependencies beyond numpy — no scipy, SORT, or DeepSORT needed.

Perubahan dari versi sebelumnya
--------------------------------
* **Velocity prediction**: setiap Track menyimpan estimasi kecepatan gerak
  pusat bounding-box (vx, vy dalam piksel/frame).  Saat matching, digunakan
  *posisi prediksi* (bbox + velocity × frames_lost) alih-alih posisi terakhir,
  sehingga botol yang sempat tidak terdeteksi beberapa frame tetap bisa dicocokkan
  ke track lama dan tidak dihitung dua kali.
* **min_frames_seen**: track yang total frame-terlihat-nya < threshold tidak
  dikeluarkan sebagai "completed" — mencegah noise / false-positive singkat
  dihitung sebagai botol.

Usage
-----
    tracker = BottleTracker(iou_threshold=0.3, max_lost=20, min_frames_seen=5)

    # Inside inference loop:
    active_tracks, completed_tracks = tracker.update(detections)

    for track in active_tracks:
        print(track.id, track.class_id, track.bbox)

    for track in completed_tracks:
        decision = tracker.final_decision(track)   # 0=VALID, 1=INVALID
        print(f"Botol #{track.id} selesai → {'VALID' if decision == 0 else 'INVALID'}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# ── Track dataclass ──────────────────────────────────────────────────────────

@dataclass
class Track:
    """Represents a single tracked object across frames."""
    id: int
    bbox: np.ndarray          # [x1, y1, x2, y2]  float32
    class_id: int             # latest frame: 0 = bottle (VALID), 1 = not_bottle (INVALID)
    conf: float
    frames_seen: int   = 1   # total frames matched to this track
    frames_lost: int   = 0   # consecutive frames without a match
    frames_as_valid:   int = 0   # frames detected as class 0 (bottle)
    frames_as_invalid: int = 0   # frames detected as class 1 (not_bottle)

    # Estimated center velocity (pixels/frame), updated on each matched frame.
    # Menggunakan exponential moving average agar tidak terlalu sensitif terhadap noise.
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(2, dtype=np.float32))

    def __post_init__(self):
        # Count the first detection in the vote tally
        if self.class_id == 0:
            self.frames_as_valid = 1
        else:
            self.frames_as_invalid = 1

    # ------------------------------------------------------------------ #

    def predicted_bbox(self) -> np.ndarray:
        """
        Kembalikan bbox yang sudah digeser sesuai estimasi kecepatan dan
        jumlah frame yang hilang.  Digunakan untuk matching agar botol
        yang sempat tidak terdeteksi tetap bisa dicocokkan ke track lama.

        :return: ndarray [x1, y1, x2, y2] float32
        """
        if self.frames_lost == 0 or np.all(self.velocity == 0):
            return self.bbox
        shift = self.velocity * self.frames_lost          # (vx, vy) × frames
        offset = np.array([shift[0], shift[1], shift[0], shift[1]], dtype=np.float32)
        return self.bbox + offset

    def update_velocity(self, new_bbox: np.ndarray, alpha: float = 0.5) -> None:
        """
        Perbarui estimasi kecepatan menggunakan exponential moving average.

        :param new_bbox: bbox baru [x1, y1, x2, y2] setelah match.
        :param alpha:    bobot frame terbaru (0–1).  0.5 = rata-rata antara
                         kecepatan lama dan perpindahan baru.
        """
        old_cx = (self.bbox[0] + self.bbox[2]) / 2.0
        old_cy = (self.bbox[1] + self.bbox[3]) / 2.0
        new_cx = (new_bbox[0] + new_bbox[2]) / 2.0
        new_cy = (new_bbox[1] + new_bbox[3]) / 2.0

        # Jika ada frame yang hilang, bagi perpindahan dengan jumlah frame yang
        # seharusnya sudah terjadi agar estimasi per-frame tetap masuk akal.
        divisor = max(1, self.frames_lost + 1)
        measured_vx = (new_cx - old_cx) / divisor
        measured_vy = (new_cy - old_cy) / divisor

        self.velocity[0] = alpha * measured_vx + (1 - alpha) * self.velocity[0]
        self.velocity[1] = alpha * measured_vy + (1 - alpha) * self.velocity[1]


# ── IoU helper ───────────────────────────────────────────────────────────────

def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.

    :param box_a: array of shape (4,)
    :param box_b: array of shape (4,)
    :return: IoU value in [0, 1]
    """
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, xa2 - xa1) * max(0.0, ya2 - ya1)
    area_b = max(0.0, xb2 - xb1) * max(0.0, yb2 - yb1)
    union_area = area_a + area_b - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def _iou_matrix(tracks: List[Track], dets: np.ndarray, use_prediction: bool = True) -> np.ndarray:
    """
    Compute IoU matrix between all (track, detection) pairs.

    :param tracks:          list of active Track objects
    :param dets:            ndarray of shape (m, 6) — columns [x1,y1,x2,y2,conf,class_id]
    :param use_prediction:  if True, use track.predicted_bbox() instead of track.bbox
    :return:                ndarray of shape (n_tracks, m_dets) with IoU values
    """
    matrix = np.zeros((len(tracks), len(dets)), dtype=np.float32)
    for i, track in enumerate(tracks):
        pred_box = track.predicted_bbox() if use_prediction else track.bbox
        for j, det in enumerate(dets):
            matrix[i, j] = _iou(pred_box, det[:4])
    return matrix


# ── Tracker ──────────────────────────────────────────────────────────────────

class BottleTracker:
    """
    Greedy IoU-based tracker designed for bottles on a conveyor belt,
    with velocity prediction to survive brief detection gaps.

    On each call to ``update()``:
    1. Compute IoU between predicted track positions and new detections.
    2. Greedily match the highest-IoU pairs above ``iou_threshold``.
    3. Matched tracks  → update bbox / conf / class_id; update velocity;
                         accumulate vote counters; reset frames_lost.
    4. Unmatched tracks → increment frames_lost; emit as *completed* when
                          frames_lost > max_lost (only if frames_seen >= min_frames_seen).
    5. Unmatched dets  → create new tracks with auto-incrementing IDs.

    The returned ``completed_tracks`` carry the full vote history so the caller
    can make a stable final decision via ``final_decision(track)``.

    Parameters
    ----------
    iou_threshold:
        Minimum IoU required to consider a detection a match for an existing
        track.  Lower = more lenient (good for fast-moving bottles).
    max_lost:
        Number of consecutive frames a track can go unmatched before it is
        emitted as completed and removed.  At 30 fps, 20 frames ≈ 0.67 s.
    min_frames_seen:
        Track yang terlihat kurang dari nilai ini tidak akan dihitung sebagai
        botol selesai.  Berguna untuk menyaring noise / deteksi kilat yang bukan
        botol sungguhan.  Default 5 frames (≈ 0.17 s di 30 fps).
    velocity_alpha:
        Bobot EMA untuk update kecepatan (0–1).  Nilai lebih tinggi = lebih
        responsif terhadap perubahan kecepatan, tapi lebih berisik.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_lost: int = 20,
        min_frames_seen: int = 5,
        velocity_alpha: float = 0.5,
    ) -> None:
        self.iou_threshold   = iou_threshold
        self.max_lost        = max_lost
        self.min_frames_seen = min_frames_seen
        self.velocity_alpha  = velocity_alpha

        self._tracks:  List[Track] = []
        self._next_id: int = 1

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def update(
        self, detections: Optional[np.ndarray]
    ) -> Tuple[List[Track], List[Track]]:
        """
        Update the tracker with detections from the current frame.

        :param detections: ndarray of shape (n, 6) with columns
                           [x1, y1, x2, y2, conf, class_id], or ``None``
                           when there are no detections this frame.
        :return: ``(active_tracks, completed_tracks)``
                 - ``active_tracks``: tracks visible in the current frame
                   (frames_lost == 0), ready to draw on screen.
                 - ``completed_tracks``: tracks that just exceeded ``max_lost``
                   *dan* punya frames_seen >= min_frames_seen — gunakan vote
                   tallies-nya untuk keputusan valid/invalid final.
        """
        # ── Case: no detections this frame ──────────────────────────────
        if detections is None or len(detections) == 0:
            for track in self._tracks:
                track.frames_lost += 1
            completed = self._pop_stale_tracks()
            return self._active_tracks(), completed

        # ── Match existing tracks to new detections ──────────────────────
        if self._tracks:
            # Gunakan predicted_bbox agar botol yang sempat tidak terdeteksi
            # tetap bisa dicocokkan berdasarkan posisi prediksi
            iou_mat = _iou_matrix(self._tracks, detections, use_prediction=True)
            matched_track_ids: set[int] = set()
            matched_det_ids:   set[int] = set()

            # Greedy: repeatedly pick the highest-IoU unmatched pair
            while True:
                masked = iou_mat.copy()
                masked[list(matched_track_ids), :] = -1
                masked[:, list(matched_det_ids)]   = -1

                best_val = masked.max()
                if best_val < self.iou_threshold:
                    break

                ti, di = np.unravel_index(masked.argmax(), masked.shape)
                matched_track_ids.add(int(ti))
                matched_det_ids.add(int(di))

                # Update the matched track
                det   = detections[di]
                track = self._tracks[ti]

                # Perbarui kecepatan SEBELUM mengubah bbox
                track.update_velocity(det[:4], alpha=self.velocity_alpha)

                track.bbox      = det[:4].copy()
                track.conf      = float(det[4])
                track.class_id  = int(det[5])
                track.frames_seen += 1
                track.frames_lost  = 0

                # Accumulate vote counters
                if track.class_id == 0:
                    track.frames_as_valid += 1
                else:
                    track.frames_as_invalid += 1

            # Increment frames_lost for unmatched tracks
            for ti, track in enumerate(self._tracks):
                if ti not in matched_track_ids:
                    track.frames_lost += 1

            # Create new tracks for unmatched detections
            for di, det in enumerate(detections):
                if di not in matched_det_ids:
                    self._tracks.append(Track(
                        id       = self._next_id,
                        bbox     = det[:4].copy(),
                        conf     = float(det[4]),
                        class_id = int(det[5]),
                    ))
                    self._next_id += 1

        else:
            # No existing tracks — create one per detection
            for det in detections:
                self._tracks.append(Track(
                    id       = self._next_id,
                    bbox     = det[:4].copy(),
                    conf     = float(det[4]),
                    class_id = int(det[5]),
                ))
                self._next_id += 1

        completed = self._pop_stale_tracks()
        return self._active_tracks(), completed

    @staticmethod
    def final_decision(track: Track) -> int:
        """
        Determine the final valid/invalid decision for a completed track
        using majority vote over all frames it was seen.

        :param track: A completed (or any) Track object.
        :return: ``0`` if the majority of frames classified it as *bottle*
                 (VALID), ``1`` if the majority classified it as *not_bottle*
                 (INVALID).  Ties are resolved in favour of VALID (0).
        """
        return 0 if track.frames_as_valid >= track.frames_as_invalid else 1

    def reset(self) -> None:
        """Clear all tracks and reset the ID counter."""
        self._tracks.clear()
        self._next_id = 1

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _pop_stale_tracks(self) -> List[Track]:
        """
        Remove tracks that have been lost for too long.

        Hanya track dengan ``frames_seen >= min_frames_seen`` yang dikembalikan
        sebagai *completed* (yang akan dihitung sebagai botol).  Track yang
        terlalu singkat dianggap noise dan dibuang diam-diam.
        """
        stale        = [t for t in self._tracks if t.frames_lost > self.max_lost]
        self._tracks = [t for t in self._tracks if t.frames_lost <= self.max_lost]

        # Buang track noise / terlalu singkat
        meaningful = [t for t in stale if t.frames_seen >= self.min_frames_seen]
        return meaningful

    def _active_tracks(self) -> List[Track]:
        """Return only tracks that are currently visible (frames_lost == 0)."""
        return [t for t in self._tracks if t.frames_lost == 0]
