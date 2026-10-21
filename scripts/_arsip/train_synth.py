"""Train YOLOv8n cap/no_cap. Dataset + run name via env (TRAIN_DS / RUN_NAME),
default synth_cap_v3. Then eval on held-out test split, per-source breakdown, export OpenVINO.
"""
import os, glob, collections
from multiprocessing import freeze_support
from ultralytics import YOLO

DATASETS = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets"
WEIGHTS = os.environ.get("MODEL", f"{DATASETS}/bottle_cap_base/yolov8n.pt")
if not os.path.isabs(WEIGHTS) and os.path.exists(f"{DATASETS}/{WEIGHTS}"):
    WEIGHTS = f"{DATASETS}/{WEIGHTS}"
DS = os.environ.get("TRAIN_DS", "synth_cap_v3")
RUN = os.environ.get("RUN_NAME", f"{DS}_640")
EPOCHS = int(os.environ.get("EPOCHS", "70"))


def main():
    data = f"{DATASETS}/{DS}/data.yaml"
    m = YOLO(WEIGHTS if os.path.exists(WEIGHTS) else "yolov8n.pt")
    m.train(
        data=data, imgsz=640, epochs=EPOCHS, batch=16, patience=20,
        seed=0, device=0, workers=2, cache=False, close_mosaic=10,
        project=f"{DATASETS}/cap_runs", name=RUN, plots=True, val=True, exist_ok=True,
    )
    best = f"{DATASETS}/cap_runs/{RUN}/weights/best.pt"
    mm = YOLO(best)
    met = mm.val(data=data, split="test", project=f"{DATASETS}/cap_runs",
                 name=f"{RUN}_test", device=0, exist_ok=True)
    print("TEST metrics:", met.results_dict, flush=True)

    # per-source test breakdown (s1_ / s2_ / cv_)
    tdir = f"{DATASETS}/{DS}/images/test"
    for pref, label in (("s1_", "shoot_v1"), ("s2_", "shoot_v2"), ("cv_", "conveyor")):
        imgs = glob.glob(f"{tdir}/{pref}*.jpg")
        if not imgs:
            continue
        res = mm.predict(imgs, imgsz=640, conf=0.25, verbose=False)
        gt = tp = fp = fn = 0
        for r, ip in zip(res, imgs):
            lp = ip.replace("/images/", "/labels/")[:-4] + ".txt"
            lp = lp.replace("\\", "/")
            gts = [int(x.split()[0]) for x in open(lp) if x.strip()] if os.path.exists(lp) else []
            preds = [int(b.cls) for b in r.boxes]
            gt += len(gts)
            gc, pc = collections.Counter(gts), collections.Counter(preds)
            for c in (0, 1):
                tp += min(gc[c], pc[c]); fn += max(0, gc[c] - pc[c]); fp += max(0, pc[c] - gc[c])
        prec = tp / (tp + fp) if tp + fp else 0
        rec = tp / (tp + fn) if tp + fn else 0
        print(f"  [{label}] test imgs {len(imgs)} gt {gt} | P {prec:.3f} R {rec:.3f} (class-matched box count)", flush=True)

    try:
        mm.export(format="openvino", imgsz=640)
        print("exported OpenVINO", flush=True)
    except Exception as e:
        print("openvino export failed:", e, flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
