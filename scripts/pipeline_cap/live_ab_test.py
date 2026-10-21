"""Live A/B: model A (left) vs model B (right) cap/no_cap on webcam.
Same frame to both models, boxes drawn per side, confidence shown big.
q / Esc / close-window to quit.
args: [camera_index] [runA] [runB]   default: 0 synth_cap_v1_640 synth_cap_v3_640
"""
import sys, time, os
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "timeout;15000000|reconnect;1|reconnect_streamed;1")
import cv2
from ultralytics import YOLO

DAT = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/4_eksperimen/cap/cap_runs"
RUN_A = sys.argv[2] if len(sys.argv) > 2 else "synth_cap_v1_640"
RUN_B = sys.argv[3] if len(sys.argv) > 3 else "synth_cap_v3_640"
V1 = f"{DAT}/{RUN_A}/weights/best.pt"
V2 = f"{DAT}/{RUN_B}/weights/best.pt"
_camarg = sys.argv[1] if len(sys.argv) > 1 else "0"
if _camarg.isdigit():
    CAM = int(_camarg)                       # local webcam index
elif "." in _camarg and "/" not in _camarg:
    CAM = f"http://{_camarg}:4747/video"     # bare DroidCam IP -> MJPEG url
else:
    CAM = _camarg                            # full url / device string
CONF = 0.15
NAME = {0: "cap", 1: "no_cap"}
COL = {0: (0, 200, 0), 1: (0, 0, 255)}
WIN = f"A: {RUN_A}   |   B: {RUN_B}      [q = quit]"

m1, m2 = YOLO(V1), YOLO(V2)
if isinstance(CAM, int):
    cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
else:
    cands = [CAM]
    if CAM.endswith("/video"):
        base = CAM[:-6]
        cands += [base + "/mjpegfeed?640x480", base + "/mjpegfeed", base + "/videofeed"]
    cap = None
    for url in cands:
        for attempt in range(3):
            print(f"opening stream: {url} (try {attempt+1})", flush=True)
            c = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            ok, _ = (c.read() if c.isOpened() else (False, None))
            if ok:
                cap = c
                print("stream OK:", url, flush=True)
                break
            c.release(); time.sleep(2)
        if cap:
            break
if cap is None or not cap.isOpened():
    raise SystemExit(f"cannot open camera source: {CAM}  "
                     f"(phone on same WiFi? DroidCam app in foreground? only 1 client at a time)")
for _ in range(30):
    cap.read(); time.sleep(0.03)


def annotate(frame, model, tag):
    im = frame.copy()
    r = model.predict(im, imgsz=640, conf=CONF, verbose=False)[0]
    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        c = int(b.cls); cf = float(b.conf)
        cv2.rectangle(im, (x1, y1), (x2, y2), COL[c], 3)
        cv2.putText(im, f"{NAME[c]} {cf:.2f}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COL[c], 2)
    h = im.shape[0]
    cv2.rectangle(im, (0, 0), (im.shape[1], 34), (0, 0, 0), -1)
    txt = f"{tag}   {len(r.boxes)} det"
    if len(r.boxes):
        bb = max(r.boxes, key=lambda z: float(z.conf))
        txt += f"   best: {NAME[int(bb.cls)]} {float(bb.conf):.2f}"
    cv2.putText(im, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return im


cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
shown, miss = 0, 0
while True:
    ok, frame = cap.read()
    if not ok:
        miss += 1
        if miss > 60:
            break
        continue
    miss = 0
    left = annotate(frame, m1, RUN_A)
    right = annotate(frame, m2, RUN_B)
    combo = cv2.hconcat([left, right])
    cv2.line(combo, (left.shape[1], 0), (left.shape[1], combo.shape[0]), (255, 255, 0), 2)
    cv2.imshow(WIN, combo)
    shown += 1
    k = cv2.waitKey(1) & 0xFF
    if k in (ord('q'), 27):
        break
    if shown > 5 and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("done")
