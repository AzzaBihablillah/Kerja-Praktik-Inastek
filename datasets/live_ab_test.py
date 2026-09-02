"""Live A/B: v1 (left) vs v2 (right) cap/no_cap on webcam.
Same frame to both models, boxes drawn per side, confidence shown big.
q / Esc / close-window to quit.  optional arg: camera index (default 0).
"""
import sys, time
import cv2
from ultralytics import YOLO

DAT = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/cap_runs"
V1 = f"{DAT}/synth_cap_v1_640/weights/best.pt"
V2 = f"{DAT}/synth_cap_v2_640/weights/best.pt"
CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CONF = 0.15
NAME = {0: "cap", 1: "no_cap"}
COL = {0: (0, 200, 0), 1: (0, 0, 255)}
WIN = "v1  vs  v2      [q = quit]"

m1, m2 = YOLO(V1), YOLO(V2)
cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
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
    left = annotate(frame, m1, "v1")
    right = annotate(frame, m2, "v2")
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
