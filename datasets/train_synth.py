"""Train YOLOv8n cap/no_cap on synth_cap_v2 (synthetic + colour-balanced conveyor).
Then evaluate on the held-out test split and export best.pt to OpenVINO.
"""
import os
from multiprocessing import freeze_support
from ultralytics import YOLO

DATASETS = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets"
WEIGHTS = f"{DATASETS}/bottle_cap_base/yolov8n.pt"


def main():
    m = YOLO(WEIGHTS if os.path.exists(WEIGHTS) else "yolov8n.pt")
    m.train(
        data=f"{DATASETS}/synth_cap_v2/data.yaml",
        imgsz=640, epochs=100, batch=16, patience=30,
        seed=0, device=0, workers=2, cache=False, close_mosaic=10,
        project=f"{DATASETS}/cap_runs", name="synth_cap_v2_640",
        plots=True, val=True, exist_ok=True,
    )
    best = f"{DATASETS}/cap_runs/synth_cap_v2_640/weights/best.pt"
    mm = YOLO(best)
    met = mm.val(data=f"{DATASETS}/synth_cap_v2/data.yaml", split="test",
                 project=f"{DATASETS}/cap_runs", name="synth_cap_v2_640_test",
                 device=0, exist_ok=True)
    print("TEST metrics:", met.results_dict, flush=True)
    try:
        mm.export(format="openvino", imgsz=640)
        print("exported OpenVINO", flush=True)
    except Exception as e:
        print("openvino export failed:", e, flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
