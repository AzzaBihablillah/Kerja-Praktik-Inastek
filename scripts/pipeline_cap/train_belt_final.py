"""Model FINAL cap/no_cap: fine-tune cap_v4 pada SEMUA 180 foto belt (30 botol).

Resep identik dengan cv_belt.py (30 epoch tetap, imgsz 960, seed 0, tanpa val/early-stop) -> angka CV 5-fold
(177/180 = 98.3%) adalah perkiraan yang sah untuk model ini. Tidak ada held-out di sini, jadi evaluasi di akhir
hanya SANITY (in-sample), BUKAN ukuran akurasi.
Output: cap_runs/cap_belt_final/weights/last.pt  (+ salinan final.pt)
env: EPOCHS (30) IMGSZ (960)
"""
import glob, os, shutil
from multiprocessing import freeze_support
from ultralytics import YOLO

DS = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets"
ALL = f"{DS}/2_dataset_cap/cap_belt_all"
BASE = f"{DS}/4_eksperimen/cap/cap_runs/cap_v4_640/weights/best.pt"
RUN = "cap_belt_final"
EPOCHS = int(os.environ.get("EPOCHS", "30"))
IMGSZ = int(os.environ.get("IMGSZ", "960"))


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build():
    if os.path.exists(ALL):
        shutil.rmtree(ALL)
    os.makedirs(f"{ALL}/images/train"); os.makedirs(f"{ALL}/labels/train")
    n = 0
    for ip in sorted(glob.glob(f"{DS}/2_dataset_cap/cap_belt_ds/images/*/*.jpg")):
        name = os.path.basename(ip)[:-4]
        sp = os.path.basename(os.path.dirname(ip))
        link(ip, f"{ALL}/images/train/{name}.jpg")
        shutil.copy2(f"{DS}/2_dataset_cap/cap_belt_ds/labels/{sp}/{name}.txt", f"{ALL}/labels/train/{name}.txt")
        n += 1
    assert n == 180, n
    # val dipakai Ultralytics hanya utk formalitas (val=False -> tidak dijalankan / tidak memilih epoch)
    open(f"{ALL}/data.yaml", "w").write(
        f"path: {ALL}\ntrain: images/train\nval: images/train\nnames:\n  0: cap\n  1: no_cap\n")
    return f"{ALL}/data.yaml"


def sanity(weights, imgsz):
    m = YOLO(weights)
    imgs = sorted(glob.glob(f"{ALL}/images/train/*.jpg"))
    res = m.predict(imgs, imgsz=imgsz, conf=0.03, max_det=10, verbose=False)
    ok = nodet = 0
    for ip, r in zip(imgs, res):
        c = 0 if "_cap_" in os.path.basename(ip) else 1
        if len(r.boxes) == 0:
            nodet += 1; continue
        b = max(r.boxes, key=lambda z: float(z.conf))
        ok += int(b.cls) == c
    print(f"[sanity in-sample, imgsz {imgsz}] kelas benar {ok}/180, tanpa deteksi {nodet}", flush=True)


def main():
    data = build()
    m = YOLO(BASE)
    m.train(data=data, imgsz=IMGSZ, epochs=EPOCHS, batch=8, patience=0, val=False, seed=0, device=0, workers=2,
            cache=False, close_mosaic=10, degrees=0.0, scale=0.3, translate=0.1, fliplr=0.5, flipud=0.0,
            project=f"{DS}/4_eksperimen/cap/cap_runs", name=RUN, plots=True, exist_ok=True)
    last = f"{DS}/4_eksperimen/cap/cap_runs/{RUN}/weights/last.pt"
    shutil.copy2(last, f"{DS}/4_eksperimen/cap/cap_runs/{RUN}/weights/final.pt")
    sanity(last, 640)
    sanity(last, 960)
    print("selesai.", flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
