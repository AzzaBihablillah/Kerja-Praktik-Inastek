"""Langkah 29: augmentasi GESER HUE hanya pada piksel BOTOL (cairan, label, tutup) -- belt TIDAK digeser.
Ide (user, 20 Okt): warna dasar yang sudah dilihat model bisa 'disusun' jadi warna lain; uji: apakah biru dikenali tanpa contoh biru.
Diterapkan SIMETRIS ke berisi DAN kosong (pelajaran v2 cap: augmentasi asimetris membuat jalan pintas):
  berisi -> cairan bergeser warna, tetap 'berwarna';  kosong -> label/tutup bergeser warna, tetap 'kosong'
  (sekaligus mengajari: label berwarna apa pun != isi).
Masker = piksel di dalam kotak botol yang jaraknya (Lab) > THR dari warna belt di sekitar botol.
3 salinan per foto, geser hue acak dari {30,60,90,120,150} (satuan OpenCV 0-179), seed per id.
Hanya W16 dan V20 (data latih RA). Keluaran: out/crops_fold_hue/<id>_h<k>.jpg + out/hue_sheet.jpg"""
import csv, os, random, sys
import cv2, numpy as np
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); HD = f"{OUT}/crops_fold_hue"; os.makedirs(HD, exist_ok=True)
THR, K = 18.0, 3


def belt_ref(lab, x1, y1, x2, y2):
    h = y2 - y1
    top = lab[max(0, y1 - int(h * .5)):max(1, y1 - 8), x1:x2].reshape(-1, 3)
    bot = lab[min(lab.shape[0] - 1, y2 + 8):y2 + int(h * .5), x1:x2].reshape(-1, 3)
    a = np.concatenate([top, bot]); return np.median(a, axis=0) if len(a) > 50 else np.array([70, 110, 125], np.float32)


def shifted(im, box, dh):
    x1, y1, x2, y2 = [int(v) for v in box]
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = belt_ref(lab, x1, y1, x2, y2)
    m = np.zeros(im.shape[:2], bool)
    sub = np.linalg.norm(lab[y1:y2, x1:x2] - ref[None, None, :], axis=2) > THR
    m[y1:y2, x1:x2] = sub
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    h = hsv[..., 0].astype(np.int32); h[m] = (h[m] + dh) % 180; hsv[..., 0] = h.astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def main():
    idx = list(csv.DictReader(open(f"{OUT}/belt_index.csv")))
    items = []                                                     # (id, path, box)
    for i, r in enumerate(idx):
        if r["src"] == "W16" and r["liquid"] in ("berwarna", "kosong"):
            items.append((f"W16_{i:04d}", os.path.join(ROOT, r["path"]), [float(r[k]) for k in ("x1", "y1", "x2", "y2")]))
    det = YOLO(f"{OUT}/final2/det_ver8b/weights/last.pt")
    for r in csv.DictReader(open(f"{OUT}/v20_crops.csv")):
        im = cv2.imread(os.path.join(ROOT, r["path"]))
        res = det.predict(im, imgsz=640, conf=0.35, verbose=False)[0]
        bb = [b for b in res.boxes if int(b.cls) == 0]
        if bb:
            items.append((r["id"], os.path.join(ROOT, r["path"]), max(bb, key=lambda z: float(z.conf)).xyxy[0].tolist()))
    for cid, path, box in items:
        if all(os.path.exists(f"{HD}/{cid}_h{k}.jpg") for k in range(K)):
            continue
        im = cv2.imread(path); rng = random.Random(cid)
        for k, dh in enumerate(rng.sample([30, 60, 90, 120, 150], K)):
            cv2.imwrite(f"{HD}/{cid}_h{k}.jpg", fold_crop(shifted(im, box, dh), *box), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(len(items), "foto x", K, "salinan hue")
    sel = [c for c, _, _ in items][::23][:8]
    t = []
    for c in sel:
        row = [cv2.resize(cv2.imread(f"{OUT}/crops_fold/{c}.jpg"), (180, 180))] + [cv2.resize(cv2.imread(f"{HD}/{c}_h{k}.jpg"), (180, 180)) for k in range(K)]
        t.append(cv2.hconcat(row))
    cv2.imwrite(f"{OUT}/hue_sheet.jpg", cv2.vconcat(t))


if __name__ == "__main__":
    main()
