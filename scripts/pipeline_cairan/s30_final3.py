"""Langkah 30: model FINAL ke-3 (keputusan 21 Okt) dari SEMUA data belt berwarna: W16 (16 Okt, tanpa seperempat) + V20 (20 Okt).
  cls3_RH_all : + 3 salinan geser-hue per foto (s29)   -> resep utama
  cls3_RA_all : tanpa geser hue                        -> untuk ansambel RA+RH
Resep sama dengan validasi s28 (seimbang ceil, 30 epoch, crop lipat, seed 0). Tidak ada uji bersih:
akurasi = hasil validasi per kelompok cairan (CATATAN-PERCOBAAN-TA L-13/L-14). Keluaran: out/final3/<nama>/weights/last.pt"""
import csv, math, os, shutil
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); FD = f"{OUT}/final3"; os.makedirs(FD, exist_ok=True)


def train(name, hue):
    rows = [r for r in list(csv.DictReader(open(f"{OUT}/crops.csv"))) + list(csv.DictReader(open(f"{OUT}/v20_crops.csv")))
            if r["src"] in ("W16", "V20") and r["liquid"] in ("berwarna", "kosong")]
    items = [(f"{OUT}/crops_fold/{r['id']}.jpg", r["liquid"], r["id"]) for r in rows]
    if hue:
        items += [(f"{OUT}/crops_fold_hue/{r['id']}_h{k}.jpg", r["liquid"], f"{r['id']}_h{k}") for r in rows for k in range(3)
                  if os.path.exists(f"{OUT}/crops_fold_hue/{r['id']}_h{k}.jpg")]
    d = f"{FD}/data_{name}"; shutil.rmtree(d, ignore_errors=True)
    n = {c: sum(l == c for _, l, _ in items) for c in ("berwarna", "kosong")}; big = max(n.values())
    print(name, "latih:", n, flush=True)
    for c in ("berwarna", "kosong"):
        os.makedirs(f"{d}/train/{c}"); os.makedirs(f"{d}/val/{c}")
        for p, l, i in items:
            if l == c:
                for k in range(math.ceil(big / n[c])):
                    os.link(p, f"{d}/train/{c}/{i}_{k}.jpg")
        for p, l, i in [x for x in items if x[1] == c][:6]:            # val hanya utk format
            os.link(p, f"{d}/val/{c}/{i}.jpg")
    YOLO(os.path.join(ROOT, "3_model_final/dasar/yolov8n-cls.pt")).train(
        data=d, epochs=30, imgsz=224, batch=32, seed=0, deterministic=True, workers=0, verbose=False, plots=False,
        project=FD, name=name, exist_ok=True, patience=0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, fliplr=0.5, flipud=0.5,
        degrees=0, auto_augment="", erasing=0.1)
    shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    train("cls3_RH_all", hue=True)
    train("cls3_RA_all", hue=False)
    print("FINAL3 SELESAI")
