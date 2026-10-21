"""Uji F: objek TrashNet yang DITAHAN (nomor % 5 == 0, tak pernah dilatih) ditempel ke foto belt 20 Okt (cara tempel sama).
Mengukur per kategori: non-botol (kaca, kaleng, plastik lain, sampah, kertas, kardus) DITERIMA sebagai botol (makin kecil makin baik)
dan botol PET TrashNet diterima (makin besar makin baik; kontrol jalan pintas). Tiap objek ditempel 3x di foto berbeda.
  python scripts/pipeline_botol/uji_trashnet.py model1.pt model2.pt ...   -> datasets/4_eksperimen/botol/uji_trashnet/hasil.txt"""
import glob, os, random, sys
from collections import defaultdict
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taco_paste import DS, paste_object
from ver9_nonbottle import trashnet_objects, iou
from ultralytics import YOLO

OUT = f"{DS}/4_eksperimen/botol/uji_trashnet"


def build():
    if os.path.exists(f"{OUT}/meta.npy"):
        return list(np.load(f"{OUT}/meta.npy", allow_pickle=True))
    os.makedirs(OUT, exist_ok=True)
    det = YOLO(f"{DS}/3_model_final/botol_ver8b.pt"); rng = random.Random(21)
    neg, pos = trashnet_objects("uji"); objs = neg + pos
    frames = []
    for f in sorted(glob.glob(f"{DS}/1_foto_mentah/2026-10-20_water_v1/b*.png")):
        im = cv2.resize(cv2.imread(f), (960, 540), interpolation=cv2.INTER_AREA)
        bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
        if bb:
            frames.append((im, max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))))
    meta = []; i = 0
    for rep in range(3):
        for o in objs:
            im0, b = frames[i % len(frames)]; i += 1; im = im0.copy()
            pb = paste_object(im, o, [(b[0] - 15, b[1] - 15, b[2] + 15, b[3] + 15)], rng)
            if pb:
                name = f"f{rep}_{o['img']}"; cv2.imwrite(f"{OUT}/{name}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
                meta.append(dict(name=name, cat=o["cat"], pasted=[int(v) for v in pb], bottle=[int(v) for v in b]))
    np.save(f"{OUT}/meta.npy", np.array(meta, dtype=object))
    return meta


def main():
    meta = build(); L = [f"Uji F TrashNet ditahan, {len(meta)} tempelan ({len({m['cat'] for m in meta})} kategori). Sel = diterima sebagai botol."]
    cats = sorted({m["cat"] for m in meta})
    L.append(f"{'model':22s} " + " ".join(f"{c.replace('trashnet_', ''):>12s}" for c in cats) + f" {'non-botol total':>16s} {'botol asli':>11s}")
    for pt in sys.argv[1:]:
        m = YOLO(pt); acc = defaultdict(lambda: [0, 0]); ob = 0
        for mm in meta:
            r = m.predict(f"{OUT}/{mm['name']}.jpg", imgsz=640, conf=0.35, verbose=False)[0]
            d = [(int(b.cls), b.xyxy[0].tolist()) for b in r.boxes]
            a = acc[mm["cat"]]; a[0] += 1; a[1] += any(c == 0 and iou(x, mm["pasted"]) >= 0.3 for c, x in d)
            ob += any(c == 0 and iou(x, mm["bottle"]) >= 0.5 for c, x in d)
        tag = pt.replace("\\", "/").split("/")[-3] if "/weights/" in pt.replace("\\", "/") else os.path.basename(pt)[:-3]
        nn = [acc[c] for c in cats if not c.endswith("_PET")]; tot = (sum(x[1] for x in nn), sum(x[0] for x in nn))
        L.append(f"{tag:22s} " + " ".join(f"{acc[c][1]:>5d}/{acc[c][0]:<6d}" for c in cats) +
                 f" {tot[0]:>6d}/{tot[1]:<4d}{100 * tot[0] / tot[1]:4.0f}% {ob:>5d}/{len(meta)}")
        print(L[-1], flush=True)
    open(f"{OUT}/hasil.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
