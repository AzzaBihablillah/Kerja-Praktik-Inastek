"""Langkah 10: model final (prototipe) dengan resep yang sudah dievaluasi CV, dilatih pada SEMUA data belt berlabel:
  det_ver8      : detektor 2 kelas (bottle/non_bottle) dari ver7 + data warisan (old_train, tanpa gambar 'ragu') + W16 + C13
  cls_W16       : classifier crop resep V4 (hanya W16, tanpa seperempat)          -> ambang 0.5 (lihat laporan)
  cls_W16_C13   : classifier crop resep V2 (W16 + C13 kosong sbg negatif)          -> ambang 0.2 (lebih konservatif)
P15 & old_test tetap TIDAK dilatih (dipakai hanya utk val/monitor). Angka akurasi = hasil CV, bukan dari model ini.
"""
import csv, os, shutil
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); FD = os.path.join(OUT, "final"); os.makedirs(FD, exist_ok=True)


def cls_final(name, use_c13):
    rows = list(csv.DictReader(open(f"{OUT}/crops.csv")))
    d = f"{FD}/data_{name}"
    shutil.rmtree(d, ignore_errors=True)
    tr = [r for r in rows if r["liquid"] in ("berwarna", "kosong") and (r["src"] == "W16" or (use_c13 and r["src"] == "C13"))]
    n = {c: sum(r["liquid"] == c for r in tr) for c in ("berwarna", "kosong")}; big = max(n.values())
    for c in ("berwarna", "kosong"):
        os.makedirs(f"{d}/train/{c}"); os.makedirs(f"{d}/val/{c}")
        rep = min(6, max(1, round(big / n[c])))
        for r in tr:
            if r["liquid"] == c:
                for k in range(rep):
                    os.link(f"{OUT}/{r['crop'].replace('crops/', 'crops_fold/')}", f"{d}/train/{c}/{r['id']}_{k}.jpg")
    for r in rows:                                                   # val = P15 (monitor saja)
        if r["src"] == "P15" and r["liquid"] in ("berwarna", "kosong"):
            os.link(f"{OUT}/{r['crop'].replace('crops/', 'crops_fold/')}", f"{d}/val/{r['liquid']}/{r['id']}.jpg")
    YOLO(os.path.join(ROOT, "3_model_final/dasar/yolov8n-cls.pt")).train(
        data=d, epochs=30, imgsz=224, batch=32, seed=0, deterministic=True, workers=0, verbose=False, plots=False,
        project=FD, name=name, exist_ok=True, patience=0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, fliplr=0.5, flipud=0.5,
        degrees=0, auto_augment="", erasing=0.1)
    shutil.rmtree(d, ignore_errors=True)


def det_final():
    DD = f"{OUT}/det_data"
    idx = list(csv.DictReader(open(f"{DD}/det_index.csv")))
    tr = [r["name"] for r in idx if r["tag"] == "old_train" or r["tag"].startswith("C13_") or (r["tag"].startswith("W16_f") and not r["tag"].endswith("_q"))]
    va = [r["name"] for r in idx if r["tag"] == "old_test"]
    open(f"{FD}/ver8_train.txt", "w").write("\n".join(f"{DD}/B2/images/{n}.jpg" for n in tr) + "\n")
    open(f"{FD}/ver8_val.txt", "w").write("\n".join(f"{DD}/B2/images/{n}.jpg" for n in va) + "\n")
    open(f"{FD}/ver8.yaml", "w").write(f"train: {FD}/ver8_train.txt\nval: {FD}/ver8_val.txt\nnames:\n  0: bottle\n  1: non_bottle\n")
    YOLO(os.path.join(ROOT, "..", "export_model", "bottle_best_ver7.pt")).train(
        data=f"{FD}/ver8.yaml", epochs=10, imgsz=640, batch=16, workers=0, seed=0, deterministic=True, patience=0, plots=False,
        verbose=False, project=FD, name="det_ver8", exist_ok=True, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, degrees=0, fliplr=0.5,
        flipud=0.0, mosaic=1.0, close_mosaic=3)


if __name__ == "__main__":
    cls_final("cls_W16", use_c13=False)
    cls_final("cls_W16_C13", use_c13=True)
    det_final()
    print("FINAL SELESAI")
