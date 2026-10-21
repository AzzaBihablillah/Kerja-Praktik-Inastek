"""Pra-label foto belt (primary/<tgl>_conveyor_v1) dengan cap_v4 -> YOLO txt, untuk DIKOREKSI manual.

Foto asli TIDAK diubah. Output (folder baru):
  OUT/images/*.jpg   salinan 1920x1080 (JPEG q95)
  OUT/labels/*.txt   1 box per foto: <kelas dari NAMA FILE> cx cy w h   (0=cap, 1=no_cap)
  OUT/prelabel_report.csv   per foto: sumber box, conf, kelas tebakan model, prioritas review
  OUT/review_*.jpg   lembar kontak overlay (prioritas dulu)

Kelas diambil dari nama file (..._cap_.. / ..._nocap_..) karena itu ground truth dari cara foto diambil;
model hanya memberi LOKASI box. Tebakan kelas model yang beda dicatat sebagai prioritas review.
Foto tanpa deteksi -> box penanda (posisi dari nomor foto 1/3, tengah, 2/3), ditandai PLACEHOLDER.
Tidak ada heuristik lain: box selalu dari model atau penanda yang jelas-jelas harus digeser manual.
"""
import argparse, csv, glob, os, re, statistics
import cv2, numpy as np
from ultralytics import YOLO

DS = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets"
ap = argparse.ArgumentParser()
ap.add_argument("--src", default=None, help="folder foto (default: primary/*_conveyor_v1 terbaru)")
ap.add_argument("--out", default=f"{DS}/2_dataset_cap/cap_belt_v1")
ap.add_argument("--weights", default=f"{DS}/4_eksperimen/cap/cap_runs/cap_v4_640/weights/best.pt")
ap.add_argument("--imgsz", default="1280,960,640", help="skala yang dicoba, ambil conf tertinggi")
ap.add_argument("--conf", type=float, default=0.03)
a = ap.parse_args()

src = a.src or sorted(glob.glob(f"{DS}/1_foto_mentah/*_conveyor_v1"))[-1]
files = sorted(glob.glob(f"{src}/*.png"))
files = [f for f in files if re.search(r"_(cap|nocap)_\d+\.png$", os.path.basename(f))]
assert files, f"tidak ada foto di {src}"
os.makedirs(f"{a.out}/images", exist_ok=True)
os.makedirs(f"{a.out}/labels", exist_ok=True)
model = YOLO(a.weights)
sizes = [int(s) for s in a.imgsz.split(",")]
POSX = {1: 1 / 3, 2: 0.5, 3: 2 / 3}
print(f"{len(files)} foto dari {src}\nskala {sizes}  conf>={a.conf}", flush=True)

recs = []
for i, f in enumerate(files, 1):
    name = os.path.splitext(os.path.basename(f))[0]
    m = re.match(r"b(\d+)_.*_(cap|nocap)_(\d+)$", name)
    bottle, state, pos = int(m[1]), m[2], int(m[3])
    fcls = 0 if state == "cap" else 1
    im = cv2.imread(f)
    H, W = im.shape[:2]
    cands = []                                   # (conf, cls, xyxy, imgsz)
    for s in sizes:
        r = model.predict(im, imgsz=s, conf=a.conf, max_det=10, verbose=False)[0]
        for b in r.boxes:
            cands.append((float(b.conf), int(b.cls), b.xyxy[0].cpu().numpy(), s))
    best = max(cands, key=lambda t: t[0]) if cands else None
    same = [c for c in cands if c[1] == fcls and c[0] >= 0.15]
    pick = max(same, key=lambda t: t[0]) if same else best        # utamakan kelas yang sama dgn nama file
    if pick is None:
        cx, cy, bw, bh = POSX.get(pos, 0.5), 0.54, 0.06, 0.10
        src_box, conf, mcls, isz = "PLACEHOLDER", 0.0, -1, 0
    else:
        conf, mcls, xyxy, isz = pick
        cx, cy = (xyxy[0] + xyxy[2]) / 2 / W, (xyxy[1] + xyxy[3]) / 2 / H
        bw, bh = (xyxy[2] - xyxy[0]) / W, (xyxy[3] - xyxy[1]) / H
        src_box = "model"
    cv2.imwrite(f"{a.out}/images/{name}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
    open(f"{a.out}/labels/{name}.txt", "w").write(f"{fcls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
    recs.append(dict(file=name, bottle=bottle, state=state, pos=pos, fcls=fcls, src=src_box, conf=conf,
                     mcls=mcls, imgsz=isz, cx=cx, cy=cy, w=bw, h=bh, n_cands=len(cands)))
    if i % 30 == 0:
        print(f"  {i}/{len(files)}", flush=True)

# prioritas review
mw = statistics.median(r["w"] for r in recs if r["src"] == "model") if any(r["src"] == "model" for r in recs) else 0.06
mh = statistics.median(r["h"] for r in recs if r["src"] == "model") if any(r["src"] == "model" for r in recs) else 0.10
for r in recs:
    why = []
    if r["src"] == "PLACEHOLDER":
        why.append("PLACEHOLDER")
    if r["src"] == "model" and r["mcls"] != r["fcls"]:
        why.append("kelas_model_beda")
    if r["src"] == "model" and r["conf"] < 0.5:
        why.append("conf<0.5")
    if r["src"] == "model" and (not 0.6 * mw <= r["w"] <= 1.6 * mw or not 0.6 * mh <= r["h"] <= 1.6 * mh):
        why.append("ukuran_ganjil")
    r["review"] = "|".join(why)

with open(f"{a.out}/prelabel_report.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
open(f"{a.out}/classes.txt", "w").write("cap\nno_cap\n")

# lembar kontak: yang perlu review dulu, lalu sisanya
def tile(r):
    im = cv2.imread(f"{a.out}/images/{r['file']}.jpg")
    H, W = im.shape[:2]
    x1, y1 = int((r["cx"] - r["w"] / 2) * W), int((r["cy"] - r["h"] / 2) * H)
    x2, y2 = int((r["cx"] + r["w"] / 2) * W), int((r["cy"] + r["h"] / 2) * H)
    col = (0, 200, 0) if r["fcls"] == 0 else (0, 0, 255)
    cv2.rectangle(im, (x1, y1), (x2, y2), col, 5)
    t = cv2.resize(im, (640, 360))
    cv2.putText(t, f"{r['file'][:34]}", (5, 18), 0, 0.5, (0, 255, 255), 1)
    flag = r["review"] or f"ok {r['conf']:.2f}"
    cv2.putText(t, flag, (5, 350), 0, 0.55, (0, 0, 255) if r["review"] else (255, 255, 255), 2)
    return t

order = sorted(recs, key=lambda r: (r["review"] == "", r["conf"]))
PER = 12
for si in range(0, len(order), PER):
    ch = order[si:si + PER]
    ts = [tile(r) for r in ch]
    while len(ts) % 3:
        ts.append(np.zeros_like(ts[0]))
    rows = [cv2.hconcat(ts[j:j + 3]) for j in range(0, len(ts), 3)]
    cv2.imwrite(f"{a.out}/review_{si // PER + 1:02d}.jpg", cv2.vconcat(rows), [cv2.IMWRITE_JPEG_QUALITY, 85])

n = len(recs)
cnt = lambda k: sum(1 for r in recs if k in r["review"])
print(f"\nSELESAI: {n} foto -> {a.out}")
print(f"  box dari model     : {sum(1 for r in recs if r['src']=='model')}")
print(f"  PLACEHOLDER        : {cnt('PLACEHOLDER')}")
print(f"  kelas model beda   : {cnt('kelas_model_beda')}")
print(f"  conf < 0.5         : {cnt('conf<0.5')}")
print(f"  ukuran ganjil      : {cnt('ukuran_ganjil')}")
print(f"  perlu review dulu  : {sum(1 for r in recs if r['review'])}  | median box w={mw:.3f} h={mh:.3f}")
