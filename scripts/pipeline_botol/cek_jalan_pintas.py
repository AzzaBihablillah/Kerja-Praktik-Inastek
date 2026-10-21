"""Cek jalan pintas "objek tempelan = non_bottle": tempel BOTOL ASLI dengan cara tempel yang sama seperti negatif TACO.
Jika model baru (ver9) menolak botol tempelan jauh lebih sering daripada ver8b, ia belajar tanda tempelan, bukan bentuk objek.
  S1: potongan botol studio (rembg, alpha; 2 Sep, tanpa tangan) ditempel ke foto belt 20 Okt
  S2: botol belt 20 Okt (foto lain) dipotong dengan mask elips berhalus lalu ditempel ke foto belt 20 Okt
Metrik: botol tempelan DITERIMA sebagai bottle (conf >= 0.35, IoU >= 0.3). Semakin tinggi semakin baik.
  python scripts/pipeline_botol/cek_jalan_pintas.py model1.pt model2.pt ..."""
import glob, os, random, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taco_paste import DS, paste_object
from ultralytics import YOLO

OUT = f"{DS}/4_eksperimen/botol/cek_jalan_pintas"


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def build():
    if os.path.exists(f"{OUT}/meta.npy"):
        return list(np.load(f"{OUT}/meta.npy", allow_pickle=True))
    os.makedirs(OUT, exist_ok=True)
    det = YOLO(f"{DS}/3_model_final/botol_ver8b.pt"); rng = random.Random(11)
    # hanya botol studio bernomor GANJIL (b01, b03, ...): yang genap boleh dipakai latih ver9b
    studio = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in sorted(glob.glob(f"{DS}/_arsip/eksperimen_cap_lama/cap_shoot_v1_cutout/nohand/b*.png"))
              if int(os.path.basename(f)[1:3]) % 2 == 1]
    studio = [s for s in studio if s is not None and s.shape[2] == 4]
    fs = sorted(glob.glob(f"{DS}/1_foto_mentah/2026-10-20_water_v1/b*.png"))
    frames, bots = [], []
    for f in fs:
        im = cv2.resize(cv2.imread(f), (960, 540), interpolation=cv2.INTER_AREA)
        bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
        if bb:
            frames.append(im); bots.append(max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])))
    belt_cut = []
    for im, b in zip(frames, bots):                              # S2: elips berhalus di sekitar kotak botol
        x1, y1, x2, y2 = [int(v) for v in b]; c = im[y1:y2, x1:x2].copy(); h, w = c.shape[:2]
        a = np.zeros((h, w), np.uint8); cv2.ellipse(a, (w // 2, h // 2), (int(w * 0.52), int(h * 0.56)), 0, 0, 360, 255, -1)
        belt_cut.append(np.dstack([c, cv2.GaussianBlur(a, (9, 9), 0)]))
    meta = []
    for kind, pool in (("S1_studio", studio), ("S2_belt", belt_cut)):
        for i, (im0, b) in enumerate(zip(frames, bots)):
            im = im0.copy(); taken = [(b[0] - 15, b[1] - 15, b[2] + 15, b[3] + 15)]
            o = pool[(i * 7 + 3) % len(pool)] if kind == "S2_belt" else pool[rng.randrange(len(pool))]
            if kind == "S2_belt" and o is belt_cut[i]:
                o = pool[(i + 1) % len(pool)]
            pb = paste_object(im, dict(rgba=o), taken, rng)
            if pb:
                name = f"{kind}_{i:03d}"; cv2.imwrite(f"{OUT}/{name}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
                meta.append(dict(name=name, kind=kind, pasted=[int(v) for v in pb], bottle=[int(v) for v in b]))
    np.save(f"{OUT}/meta.npy", np.array(meta, dtype=object))
    sheet = [cv2.resize(cv2.imread(f"{OUT}/{m['name']}.jpg"), (480, 270)) for m in meta[::max(1, len(meta) // 8)][:8]]
    cv2.imwrite(f"{OUT}/contoh.jpg", np.vstack([np.hstack(sheet[i:i + 4]) for i in range(0, 8, 4)]))
    return meta


def main():
    meta = build(); lines = []
    for pt in sys.argv[1:]:
        m = YOLO(pt); res = {}
        for mm in meta:
            r = m.predict(f"{OUT}/{mm['name']}.jpg", imgsz=640, conf=0.35, verbose=False)[0]
            d = [(int(b.cls), b.xyxy[0].tolist()) for b in r.boxes]
            s = res.setdefault(mm["kind"], [0, 0, 0, 0]); s[0] += 1
            s[1] += any(c == 0 and iou(x, mm["pasted"]) >= 0.3 for c, x in d)
            s[2] += any(c == 1 and iou(x, mm["pasted"]) >= 0.3 for c, x in d)
            s[3] += any(c == 0 and iou(x, mm["bottle"]) >= 0.5 for c, x in d)
        tag = pt.replace("\\", "/").split("/")[-3] if "/weights/" in pt.replace("\\", "/") else os.path.basename(pt)
        for k, (n, acc, nb, ob) in res.items():
            lines.append(f"{tag:18s} {k:10s} botol tempelan diterima {acc:3d}/{n} = {100 * acc / n:5.1f}% | dicap non_bottle {nb:3d} | botol asli tetap {ob}/{n}")
            print(lines[-1], flush=True)
    open(f"{OUT}/hasil.txt", "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
