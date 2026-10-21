"""Langkah 25: PRALATIH classifier di data luar Open Food Facts (positif+negatif dari sumber yang sama, 1577 crop lipat),
lalu fine-tune di belt per fold (s03 --model <bobot ini>). Menguji satu-satunya cara memakai data luar yang belum dicoba:
bukan dicampur, tetapi tahap pertama terpisah. Belt (W16/C13/P15) TIDAK ikut pralatih -> tidak ada kebocoran ke uji.
Keluaran: out/pre_off/weights/last.pt"""
import csv, math, os, shutil
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan")


def main():
    rows = list(csv.DictReader(open(f"{OUT}/crops_ext.csv", encoding="utf-8")))
    d = f"{OUT}/tmp_pre_off"; shutil.rmtree(d, ignore_errors=True)
    n = {c: sum(r["liquid"] == c for r in rows) for c in ("berwarna", "kosong")}; big = max(n.values())
    for c in ("berwarna", "kosong"):
        os.makedirs(f"{d}/train/{c}"); os.makedirs(f"{d}/val/{c}")
    for i, r in enumerate(rows):
        c = r["liquid"]; src = f"{OUT}/{r['crop'].replace('crops/', 'crops_fold/')}"
        if i % 10 == 0:                                                      # 10% utk val (monitor saja)
            os.link(src, f"{d}/val/{c}/{r['id']}.jpg"); continue
        for j in range(math.ceil(big / n[c])):
            os.link(src, f"{d}/train/{c}/{r['id']}_{j}.jpg")
    YOLO(os.path.join(ROOT, "3_model_final/dasar/yolov8n-cls.pt")).train(data=d, epochs=30, imgsz=224, batch=32, seed=0, deterministic=True, workers=0,
                                                     verbose=False, plots=False, project=OUT, name="pre_off", exist_ok=True, patience=0,
                                                     hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, fliplr=0.5, flipud=0.5, auto_augment="", erasing=0.1)
    shutil.rmtree(d, ignore_errors=True)
    print("PRALATIH SELESAI")


if __name__ == "__main__":
    main()
