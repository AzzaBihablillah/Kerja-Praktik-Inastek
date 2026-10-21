"""Langkah 26: AUDIT sesi 20 Okt (botol & warna BARU) -- uji model lama TANPA latih ulang (uji paling jujur untuk warna baru).
Kotak dari det_ver8b; crop lipat; empat classifier lama: final/cls_W16 (V4), final/cls_W16_C13 (V2), final2/cls2_V4_all, final2/cls2_V2_all.
Juga: kategori ukuran dari luas kotak (bila ml ada di nama). Keluaran: out/v20_crops.csv, out/crops_fold/V20_*.jpg, out/v20_eval.txt"""
import csv, os, re, sys
import cv2, numpy as np
from ultralytics import YOLO
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); D = f"{ROOT}/1_foto_mentah/2026-10-20_water_v1"
MODELS = {"V4 (final, W16)": f"{OUT}/final/cls_W16/weights/last.pt", "V2 (final, W16+C13)": f"{OUT}/final/cls_W16_C13/weights/last.pt",
          "V4 (final2, semua belt)": f"{OUT}/final2/cls2_V4_all/weights/last.pt", "V2 (final2, semua belt)": f"{OUT}/final2/cls2_V2_all/weights/last.pt"}
A1, A2 = 240_000, 357_000


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


det = YOLO(f"{OUT}/final2/det_ver8b/weights/last.pt")
rows = []
for r in csv.DictReader(open(f"{D}/MANIFEST.csv")):
    if r["file"] == "SKIP" or r["level"] == "ref":
        continue
    im = cv2.imread(f"{D}/{r['file']}"); H, W = im.shape[:2]
    res = det.predict(im, imgsz=640, conf=0.35, verbose=False)[0]
    bb = [b for b in res.boxes if int(b.cls) == 0]
    if not bb:
        print("tanpa kotak", r["file"]); continue
    x1, y1, x2, y2 = max(bb, key=lambda z: float(z.conf)).xyxy[0].tolist()
    cid = f"V20_{len(rows):04d}"
    cv2.imwrite(f"{OUT}/crops_fold/{cid}.jpg", fold_crop(im, x1, y1, x2, y2), [cv2.IMWRITE_JPEG_QUALITY, 95])
    m = re.search(r"(\d{3,4})", r["brand"])
    rows.append(dict(id=cid, src="V20", group=f"V20_b{int(r['bottle']):02d}", product=r["brand"], liquid="berwarna" if r["level"] in ("penuh", "setengah") else "kosong",
                     level=r["level"], cap=r["cap"], path=os.path.relpath(f"{D}/{r['file']}", ROOT), crop=f"crops/{cid}.jpg",
                     ml=m.group(1) if m else "", area=round((x2 - x1) * (y2 - y1)), clipped=int(x1 < 4 or x2 > W - 4)))
with open(f"{OUT}/v20_crops.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

L = [f"Sesi 20 Okt: {len(rows)} foto, {len({r['group'] for r in rows})} botol. Ambang 0,5 (belum dikalibrasi). Model TIDAK pernah melihat botol/warna ini (kecuali botol fisik b07 = botol Sprite NM 16 Okt, isinya baru)."]
for name, path in MODELS.items():
    m = YOLO(path); cb = [i for i, v in m.names.items() if v == "berwarna"][0]
    ps = [float(p.probs.data[cb]) for p in m.predict([f"{OUT}/crops_fold/{r['id']}.jpg" for r in rows], imgsz=224, verbose=False)]
    for r, p in zip(rows, ps):
        r[name] = p
    pos = [r[name] for r in rows if r["liquid"] == "berwarna"]; neg = [r[name] for r in rows if r["liquid"] == "kosong"]
    L.append(f"\n== {name}: AUC {auc(pos, neg):.3f} | berisi->berwarna {sum(p >= .5 for p in pos)}/{len(pos)} | kosong diterima {sum(p < .5 for p in neg)}/{len(neg)}")
    for g in sorted({r["group"] for r in rows}):
        G = [r for r in rows if r["group"] == g]
        gp = [r[name] for r in G if r["liquid"] == "berwarna"]; gn = [r[name] for r in G if r["liquid"] == "kosong"]
        L.append(f"   {g[-3:]} {G[0]['product'][:24]:24s} berisi {sum(p >= .5 for p in gp)}/{len(gp)} (median p {np.median(gp) if gp else float('nan'):.2f})"
                 + (f" | kosong diterima {sum(p < .5 for p in gn)}/{len(gn)} (median p {np.median(gn):.2f})" if gn else ""))
L.append("\nUkuran (luas kotak; kecil<450, tanggung 450-800, besar>800 ml):")
for g in sorted({r["group"] for r in rows}):
    G = [r for r in rows if r["group"] == g and not r["clipped"]]
    if not G:
        continue
    a = float(np.median([r["area"] for r in G])); cat = "kecil" if a < A1 else ("tanggung" if a < A2 else "besar")
    ml = G[0]["ml"]; truth = ("kecil" if int(ml) < 450 else ("tanggung" if int(ml) <= 800 else "besar")) if ml else "?"
    L.append(f"   {g[-3:]} {G[0]['product'][:24]:24s} luas {a / 1000:.0f}k -> {cat} | dari nama: {ml or '?'} ml -> {truth}")
open(f"{OUT}/v20_eval.txt", "w").write("\n".join(L) + "\n"); print("\n".join(L))
