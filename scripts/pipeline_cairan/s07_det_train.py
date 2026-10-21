"""Langkah 7: latih detektor per fold dari bobot ver7 (A3 = 3 kelas, B2 = 2 kelas 'ver8'). Pakai last.pt (tanpa pilih epoch).
  python s07_det_train.py --scheme A3 --folds 1,2,3,4,5 --epochs 15
"""
import argparse, os
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=["A3", "B2"])
ap.add_argument("--folds", default="1,2,3,4,5")
ap.add_argument("--epochs", type=int, default=15)
ap.add_argument("--fraction", type=float, default=1.0)
a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan")


def main():
    for k in [int(x) for x in a.folds.split(",")]:
        m = YOLO(os.path.join(ROOT, "..", "export_model", "bottle_best_ver7.pt"))
        m.train(data=f"{OUT}/det_data/{a.scheme}_f{k}.yaml", epochs=a.epochs, imgsz=640, batch=16, workers=0, seed=0, deterministic=True,
                patience=0, plots=False, verbose=False, project=f"{OUT}/runs_det", name=f"{a.scheme}_f{k}", exist_ok=True,
                hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, degrees=0, fliplr=0.5, flipud=0.0, mosaic=1.0, close_mosaic=3, fraction=a.fraction)
        print("selesai", a.scheme, k, flush=True)


if __name__ == "__main__":
    main()
