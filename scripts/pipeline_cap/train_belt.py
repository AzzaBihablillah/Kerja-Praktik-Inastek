"""Fine-tune cap/no_cap pada foto belt asli (cap_belt_ds, split per botol) + evaluasi vs baseline cap_v4.

env: MODEL (default cap_v4 best.pt) RUN_NAME EPOCHS IMGSZ BATCH
Evaluasi (val & test, tiap model pada imgsz yg sama):
  - mAP50 / mAP50-95 / P / R per kelas (ultralytics val)
  - top-1: deteksi ber-conf tertinggi per foto -> akurasi kelas, mean IoU thd box koreksi (ketat/tidak)
  - test per botol
"""
import os, glob, collections
from multiprocessing import freeze_support
import numpy as np
from ultralytics import YOLO

DS = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets"
DATA = f"{DS}/2_dataset_cap/cap_belt_ds/data.yaml"
BASE = f"{DS}/4_eksperimen/cap/cap_runs/cap_v4_640/weights/best.pt"
MODEL = os.environ.get("MODEL", BASE)
RUN = os.environ.get("RUN_NAME", "cap_belt_v1_ft")
EPOCHS = int(os.environ.get("EPOCHS", "80"))
IMGSZ = int(os.environ.get("IMGSZ", "960"))
BATCH = int(os.environ.get("BATCH", "8"))


def iou_xywhn(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    i = iw * ih
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def top1(model, split, imgsz):
    """per foto: deteksi conf tertinggi -> (kelas benar?, IoU thd GT). Return dict per kelas + per botol."""
    imgs = sorted(glob.glob(f"{DS}/2_dataset_cap/cap_belt_ds/images/{split}/*.jpg"))
    res = model.predict(imgs, imgsz=imgsz, conf=0.03, max_det=10, verbose=False)
    per = collections.defaultdict(lambda: [0, 0, []])      # cls -> [benar, total, ious]
    bot = collections.defaultdict(lambda: [0, 0])
    for ip, r in zip(imgs, res):
        name = os.path.basename(ip)[:-4]
        gt = [float(x) for x in open(f"{DS}/2_dataset_cap/cap_belt_ds/labels/{split}/{name}.txt").read().split()[:5]]
        c = int(gt[0]); per[c][1] += 1; bot[name[:3]][1] += 1
        if len(r.boxes) == 0:
            per[c][2].append(0.0); continue
        b = max(r.boxes, key=lambda z: float(z.conf))
        xywhn = [float(v) for v in b.xywhn[0]]
        per[c][2].append(iou_xywhn(gt[1:], xywhn))
        if int(b.cls) == c:
            per[c][0] += 1; bot[name[:3]][0] += 1
    return per, bot


def report(tag, model, imgsz):
    print(f"\n===== {tag} (imgsz {imgsz}) =====", flush=True)
    for split in ("val", "test"):
        m = model.val(data=DATA, split=split, imgsz=imgsz, batch=4, device=0, verbose=False,
                      plots=False, project=f"{DS}/4_eksperimen/cap/cap_runs/_tmp_eval", name=f"{tag}_{split}", exist_ok=True)
        per, bot = top1(model, split, imgsz)
        print(f"[{split}] mAP50 {m.box.map50:.3f}  mAP50-95 {m.box.map:.3f}  P {m.box.mp:.3f}  R {m.box.mr:.3f}", flush=True)
        for c, nm in ((0, "cap"), (1, "no_cap")):
            ok, n, ious = per[c]
            print(f"        {nm:6s} top-1 kelas benar {ok}/{n}  | mean IoU box {np.mean(ious):.3f}  (IoU>=0.5: {sum(i>=0.5 for i in ious)}/{n})", flush=True)
        if split == "test":
            print("        per botol (kelas benar/foto):", {k: f"{v[0]}/{v[1]}" for k, v in sorted(bot.items())}, flush=True)


def main():
    m = YOLO(MODEL)
    m.train(data=DATA, imgsz=IMGSZ, epochs=EPOCHS, batch=BATCH, patience=25, seed=0, device=0, workers=2,
            cache=False, close_mosaic=10, degrees=0.0, scale=0.3, translate=0.1, fliplr=0.5, flipud=0.0,
            project=f"{DS}/4_eksperimen/cap/cap_runs", name=RUN, plots=True, val=True, exist_ok=True)
    best = f"{DS}/4_eksperimen/cap/cap_runs/{RUN}/weights/best.pt"
    report("BASELINE_cap_v4", YOLO(BASE), IMGSZ)
    report("BASELINE_cap_v4", YOLO(BASE), 1280)
    report(f"BARU_{RUN}", YOLO(best), IMGSZ)
    if IMGSZ != 1280:
        report(f"BARU_{RUN}", YOLO(best), 1280)
    print("\nselesai.", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
