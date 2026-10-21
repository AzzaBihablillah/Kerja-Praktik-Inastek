"""Langkah 23: model FINAL ke-2 -- semua data belt nyata ikut dilatih: W16 (5 botol), C13 (30 botol kosong), P15 (sesi pilot 15 Okt: 76 foto)
dan 28 foto tambahan `foto_botol` (Frestea & Fanta KOSONG, dicek visual 19 Okt).
Label P15: warna -> berwarna; kosong, bening, 'fanta'(diisi air putih) -> BUKAN berwarna (cairan tidak berwarna / kosong).
Akurasi tetap = angka validasi silang (s03/s04); model ini tidak punya uji bersih lagi.
  cls2_V4_all   : W16 + P15 + extras (tanpa C13)         -> ambang 0.5
  cls2_V2_all   : + C13                                  -> ambang 0.2-0.5 (lihat laporan)
  det_ver8b     : ver8 + P15 + extras
"""
import csv, os, shutil, sys
import cv2
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); FD = os.path.join(OUT, "final2"); os.makedirs(FD, exist_ok=True)
DD = f"{OUT}/det_data"
EXTRA_DIR = f"{ROOT}/5_data_luar/_warisan/foto_botol/foto_botol"
EXTRA_IDS = [f"foto_{i}" for i in list(range(41, 59)) + list(range(68, 78))]


def make_extras():
    """crop fold + label detektor utk 28 foto tambahan, kotak dari det_ver8 (final) conf>=0.35; tulis out/crops_extra.csv"""
    det = YOLO(f"{OUT}/final/det_ver8/weights/last.pt"); rows = []
    for n in EXTRA_IDS:
        im = cv2.imread(f"{EXTRA_DIR}/{n}.png"); H, W = im.shape[:2]
        r = det.predict(im, imgsz=640, conf=0.35, verbose=False)[0]
        bb = [b for b in r.boxes if int(b.cls) == 0]
        if not bb:
            print("tanpa kotak:", n); continue
        b = max(bb, key=lambda z: float(z.conf)); x1, y1, x2, y2 = b.xyxy[0].tolist()
        cid = f"EXTRA_{n}"
        cv2.imwrite(f"{OUT}/crops_fold/{cid}.jpg", fold_crop(im, x1, y1, x2, y2), [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(f"{DD}/img960/{cid}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        for sch in ("A3", "B2"):
            os.makedirs(f"{DD}/{sch}/images", exist_ok=True)
            dst = f"{DD}/{sch}/images/{cid}.jpg"
            if not os.path.exists(dst):
                os.link(f"{DD}/img960/{cid}.jpg", dst)
            open(f"{DD}/{sch}/labels/{cid}.txt", "w").write(f"{0 if sch == 'B2' else 0} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        rows.append(dict(id=cid, src="EXTRA", group=f"EXTRA_{n}", product="frestea/fanta kosong", liquid="kosong", level="kosong", cap="", path=f"5_data_luar/_warisan/foto_botol/foto_botol/{n}.png", crop=f"crops/{cid}.jpg"))
    with open(f"{OUT}/crops_extra.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print("extras:", len(rows))


def cls_final(name, use_c13):
    rows = list(csv.DictReader(open(f"{OUT}/crops.csv"))) + list(csv.DictReader(open(f"{OUT}/crops_extra.csv")))
    d = f"{FD}/data_{name}"
    shutil.rmtree(d, ignore_errors=True)

    def label(r):
        if r["src"] == "P15":
            return "berwarna" if r["liquid"] == "berwarna" else "kosong"          # kosong/air_bening -> bukan berwarna
        return r["liquid"] if r["liquid"] in ("berwarna", "kosong") else None
    tr = [(r, label(r)) for r in rows if r["src"] in ("W16", "P15", "EXTRA") or (use_c13 and r["src"] == "C13")]
    tr = [(r, l) for r, l in tr if l]
    n = {c: sum(l == c for _, l in tr) for c in ("berwarna", "kosong")}; big = max(n.values())
    print(name, "latih:", n)
    for c in ("berwarna", "kosong"):
        os.makedirs(f"{d}/train/{c}"); os.makedirs(f"{d}/val/{c}")
        rep = min(6, max(1, round(big / n[c])))
        for r, l in tr:
            if l == c:
                for k in range(rep):
                    os.link(f"{OUT}/{r['crop'].replace('crops/', 'crops_fold/')}", f"{d}/train/{c}/{r['id']}_{k}.jpg")
        for r, l in tr[:0]:
            pass
    for r, l in tr:                                                                    # val hanya utk format/monitor (bukan seleksi model)
        if r["src"] == "P15" and r["id"].endswith(("0", "5")):
            os.link(f"{OUT}/{r['crop'].replace('crops/', 'crops_fold/')}", f"{d}/val/{l}/{r['id']}.jpg")
    YOLO(os.path.join(ROOT, "3_model_final/dasar/yolov8n-cls.pt")).train(
        data=d, epochs=30, imgsz=224, batch=32, seed=0, deterministic=True, workers=0, verbose=False, plots=False,
        project=FD, name=name, exist_ok=True, patience=0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, fliplr=0.5, flipud=0.5,
        degrees=0, auto_augment="", erasing=0.1)
    shutil.rmtree(d, ignore_errors=True)


def det_final():
    idx = list(csv.DictReader(open(f"{DD}/det_index.csv")))
    tr = [r["name"] for r in idx if r["tag"] == "old_train" or r["tag"].startswith("C13_") or r["tag"].startswith("P15_") or (r["tag"].startswith("W16_f") and not r["tag"].endswith("_q"))]
    tr += [f"EXTRA_{n}" for n in EXTRA_IDS if os.path.exists(f"{DD}/B2/labels/EXTRA_{n}.txt")]
    va = [r["name"] for r in idx if r["tag"] == "old_test"]
    open(f"{FD}/ver8b_train.txt", "w").write("\n".join(f"{DD}/B2/images/{n}.jpg" for n in tr) + "\n")
    open(f"{FD}/ver8b_val.txt", "w").write("\n".join(f"{DD}/B2/images/{n}.jpg" for n in va) + "\n")
    open(f"{FD}/ver8b.yaml", "w").write(f"train: {FD}/ver8b_train.txt\nval: {FD}/ver8b_val.txt\nnames:\n  0: bottle\n  1: non_bottle\n")
    YOLO(os.path.join(ROOT, "..", "export_model", "bottle_best_ver7.pt")).train(
        data=f"{FD}/ver8b.yaml", epochs=10, imgsz=640, batch=16, workers=0, seed=0, deterministic=True, patience=0, plots=False,
        verbose=False, project=FD, name="det_ver8b", exist_ok=True, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, degrees=0, fliplr=0.5,
        flipud=0.0, mosaic=1.0, close_mosaic=3)


if __name__ == "__main__":
    make_extras()
    cls_final("cls2_V4_all", use_c13=False)
    cls_final("cls2_V2_all", use_c13=True)
    det_final()
    print("FINAL2 SELESAI")
