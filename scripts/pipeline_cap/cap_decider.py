"""CapDecider — system-level stabiliser for the v4 cap/no_cap model. No retrain.

Wraps raw per-frame cap/no_cap detections with:
  * ROI gate   — a box is trusted only if its centre is inside a bottle box
                 (and, if neck_side is set, in that end-fraction of the box).
                 One box per bottle: highest confidence wins.
  * temporal vote — per bottle slot, majority label over the last `win` frames
                    with hysteresis, so a stable label only flips when the new
                    label wins >= flip_frac of the window.

Framework-agnostic: feed it plain numpy boxes. Drop into open_vino/main.py by
keying `slot` on the BottleTracker track id instead of the bottle index.
"""
from __future__ import annotations
import collections
import numpy as np


class CapDecider:
    def __init__(self, win=7, flip_frac=0.6, neck_side="none", end_frac=0.40):
        self.win = win
        self.flip_frac = flip_frac
        self.neck_side = neck_side          # none | left | right | top | bottom
        self.end_frac = end_frac
        self._hist = collections.defaultdict(lambda: collections.deque(maxlen=win))
        self._stable: dict = {}

    @staticmethod
    def _inside(cx, cy, b):
        return b[0] <= cx <= b[2] and b[1] <= cy <= b[3]

    def _in_neck(self, cx, cy, b):
        if self.neck_side == "none":
            return True
        x1, y1, x2, y2 = b
        f = self.end_frac
        return {
            "left":   cx <= x1 + f * (x2 - x1),
            "right":  cx >= x2 - f * (x2 - x1),
            "top":    cy <= y1 + f * (y2 - y1),
            "bottom": cy >= y2 - f * (y2 - y1),
        }.get(self.neck_side, True)

    def step(self, bottle_boxes, raw_dets, slots=None):
        """bottle_boxes: list[(x1,y1,x2,y2)].
        raw_dets: list[(box, cls, conf)] from the cap model.
        slots: optional stable id per bottle (e.g. track id); defaults to index.
        returns: list[(slot, box, voted_cls, conf, raw_cls)]
        """
        if slots is None:
            slots = list(range(len(bottle_boxes)))
        out, seen = [], set()
        for slot, bb in zip(slots, bottle_boxes):
            cand = []
            for box, cls, cf in raw_dets:
                cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
                if self._inside(cx, cy, bb) and self._in_neck(cx, cy, bb):
                    cand.append((cf, box, cls))
            if not cand:
                continue
            cf, box, cls = max(cand, key=lambda t: t[0])
            seen.add(slot)
            h = self._hist[slot]
            h.append(cls)
            win_cls, win_n = collections.Counter(h).most_common(1)[0]
            prev = self._stable.get(slot)
            if prev is None:
                self._stable[slot] = win_cls
            elif win_cls != prev and win_n >= max(2, int(np.ceil(len(h) * self.flip_frac))):
                self._stable[slot] = win_cls
            out.append((slot, box, self._stable[slot], cf, cls))
        for k in [k for k in self._hist if k not in seen]:
            self._hist[k].clear()
            self._stable.pop(k, None)
        return out
