"""Live webcam test of the trained cap / no_cap detector (synth_cap_v1_640/best.pt).
q or Esc or window-close to quit.  Optional arg: camera index (default 0).
"""
import sys, time
import cv2
from ultralytics import YOLO

W = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/cap_runs/synth_cap_v1_640/weights/best.pt"
CAM = int(sys.argv[1]) if len(sys.argv) > 1 else 0
CONF = 0.35
COLOR = {0: (0, 200, 0), 1: (0, 0, 255)}          # cap green, no_cap red
NAME = {0: "cap", 1: "no_cap"}
WIN = "cap / no_cap  -  best.pt   [q=quit]"

model = YOLO(W)
cap = cv2.VideoCapture(CAM, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# warmup (Windows DSHOW needs a few reads before frames are valid)
for _ in range(30):
    cap.read(); time.sleep(0.03)

cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
shown, miss, t0, fps = 0, 0, time.time(), 0.0
while True:
    ok, frame = cap.read()
    if not ok:
        miss += 1
        if miss > 60:
            print("camera lost"); break
        continue
    miss = 0
    r = model.predict(frame, imgsz=640, conf=CONF, verbose=False)[0]
    for b in r.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        c = int(b.cls); cf = float(b.conf)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR[c], 2)
        cv2.putText(frame, f"{NAME[c]} {cf:.2f}", (x1, y1 - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR[c], 2)
    fps = 0.9 * fps + 0.1 * (1.0 / max(time.time() - t0, 1e-3)); t0 = time.time()
    cv2.putText(frame, f"{fps:4.1f} FPS  {len(r.boxes)} det", (10, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow(WIN, frame)
    shown += 1
    k = cv2.waitKey(1) & 0xFF
    if k in (ord('q'), 27):
        break
    if shown > 5 and cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()
print("done")
