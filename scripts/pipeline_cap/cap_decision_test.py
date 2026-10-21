"""Option C — system-level fixes for the v4 cap/no_cap model, NO retrain.

Two mitigations, both live-tunable here before they go into open_vino/main.py:

  1. ROI GATE  — a raw cap/no_cap box is only trusted if its centre sits inside a
     BOTTLE detection (from the existing bottle model). Kills detections floating
     on background. If NECK_SIDE is set, it must also land in that end-third of the
     bottle box (the real RVM knows which end the neck faces) -> kills the
     "bottle base read as no_cap" FP. One box per bottle: highest-conf wins.

  2. TEMPORAL VOTE — per bottle, majority label over the last WIN frames with
     hysteresis. Kills the frame-to-frame flicker (tilted no_cap etc).

Window       = after gate + vote (thick), without the raw output.
args: [cam]  [neck_side: none|left|right|top|bottom]
  cam: webcam index, or DroidCam IP (uses http://IP:4747/video)
q / Esc / close window to quit.
"""
import sys, os, time
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;15000000|reconnect;1|reconnect_streamed;1")
import cv2, numpy as np
from ultralytics import YOLO
from cap_decider import CapDecider

DS = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek"
BOTTLE_W = f"{DS}/export_model/bottle_best_ver7.pt"
CAP_W = f"{DS}/datasets/4_eksperimen/cap/cap_runs/cap_v4_640/weights/best.pt"

_cam = sys.argv[1] if len(sys.argv) > 1 else "0"
if _cam.isdigit():
    CAM = int(_cam)
elif "." in _cam and "/" not in _cam:
    CAM = f"http://{_cam}:4747/video"
else:
    CAM = _cam
NECK_SIDE = (sys.argv[2].lower() if len(sys.argv) > 2 else "none")   # none|left|right|top|bottom

CONF_BOTTLE = 0.35
CONF_CAP = 0.25
WIN = 7                 # temporal window (frames)
FLIP_FRAC = 0.6         # need >=60% of the window to flip the stable label
END_FRAC = 0.40         # neck ROI = this fraction of the bottle box on NECK_SIDE
NAME = {0: "cap", 1: "no_cap"}
COL = {0: (0, 200, 0), 1: (0, 0, 255)}

mb, mc = YOLO(BOTTLE_W), YOLO(CAP_W)


def open_cam(src):
    if isinstance(src, int):
        c = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        c.set(cv2.CAP_PROP_FRAME_WIDTH, 1280); c.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        return c
    for url in [src, src.replace("/video", "/mjpegfeed?640x480")]:
        for _ in range(3):
            print("opening", url, flush=True)
            c = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            if c.isOpened() and c.read()[0]:
                print("stream OK", flush=True); return c
            c.release(); time.sleep(2)
    raise SystemExit(f"cannot open {src}")


cap = open_cam(CAM)
for _ in range(20):
    cap.read(); time.sleep(0.03)

decider = CapDecider(win=WIN, flip_frac=FLIP_FRAC, neck_side=NECK_SIDE, end_frac=END_FRAC)


def draw(img, box, cls, conf, thick, tag=""):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), COL[cls], thick)
    cv2.putText(img, f"{NAME[cls]} {conf:.2f}{tag}", (x1, max(y1 - 8, 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, COL[cls], 2)


WNAME = f"gate({NECK_SIDE}) + vote(win {WIN})     [q]"
cv2.namedWindow(WNAME, cv2.WINDOW_NORMAL)
miss = 0
while True:
    ok, frame = cap.read()
    if not ok:
        miss += 1
        if miss > 60:
            break
        continue
    miss = 0
    annotated = frame.copy()

    rb = mb.predict(frame, imgsz=640, conf=CONF_BOTTLE, verbose=False)[0]
    bottles = [b.xyxy[0].cpu().numpy() for b in rb.boxes if int(b.cls) == 0]
    for bb in bottles:
        x1, y1, x2, y2 = map(int, bb)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 180, 0), 1)

    rc = mc.predict(frame, imgsz=640, conf=CONF_CAP, verbose=False)[0]
    raw = [(b.xyxy[0].cpu().numpy(), int(b.cls), float(b.conf)) for b in rc.boxes]

    kept = decider.step(bottles, raw)
    for _slot, box, voted, cf, raw_cls in kept:
        draw(annotated, box, voted, cf, 3)

    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 30), (0, 0, 0), -1)
    cv2.putText(annotated, f"GATE+VOTE   {len(kept)} kept / {len(bottles)} bottle",
                (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow(WNAME, annotated)
    k = cv2.waitKey(1) & 0xFF
    if k in (ord('q'), 27):
        break
    if cv2.getWindowProperty(WNAME, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("done")
