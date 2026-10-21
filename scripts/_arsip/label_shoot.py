"""Auto-label shoot_v1: COCO yolov8 'bottle' box (ignores the hand), class from MANIFEST."""
import glob, os, csv, cv2
import numpy as np
from ultralytics import YOLO

SHOOT = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/shoot_v1"
CLS = {"cap": 0, "nocap": 1}
BOTTLE = 39
det = YOLO("yolov8s.pt")

man = list(csv.DictReader(open(os.path.join(SHOOT, "MANIFEST.csv"), newline="")))
flags, areas, n = [], [], 0
CH = 16
paths = [os.path.join(SHOOT, r["file"]).replace("\\", "/") for r in man]
state = {os.path.join(SHOOT, r["file"]).replace("\\", "/"): r["cap_state"] for r in man}

for i in range(0, len(paths), CH):
    batch = paths[i:i + CH]
    for p, r in zip(batch, det.predict(batch, conf=0.15, imgsz=768, device=0, verbose=False)):
        img = cv2.imread(p); H, W = img.shape[:2]
        bs = [b for b in (r.boxes or []) if int(b.cls) == BOTTLE]
        lp = os.path.splitext(p)[0] + ".txt"
        if not bs:
            flags.append((os.path.relpath(p, SHOOT), "no_bottle_detected"))
            open(lp, "w").close()
            continue
        b = max(bs, key=lambda z: float(z.conf))
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        frac = bw * bh
        areas.append(frac)
        cid = CLS[state[p]]
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        with open(lp, "w") as f:
            f.write(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
        conf = float(b.conf)
        if conf < 0.4:
            flags.append((os.path.relpath(p, SHOOT), f"low_conf {conf:.2f}"))
        elif frac > 0.75 or frac < 0.04:
            flags.append((os.path.relpath(p, SHOOT), f"box_frac {frac:.2f}"))
        n += 1

with open(os.path.join(SHOOT, "_label_flags.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["file", "reason"]); w.writerows(flags)
ar = np.array(areas)
print(f"{n} labels | box-area frac: min {ar.min():.2f} med {np.median(ar):.2f} max {ar.max():.2f}")
print(f"{len(flags)} flagged -> _label_flags.csv")
for fl in flags:
    print("  ", fl[0], "|", fl[1])
