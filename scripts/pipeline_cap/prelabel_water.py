"""Pra-label tutup untuk foto sesi cairan 16 & 20 Okt (217 foto, 21 botol) -> dikoreksi manual di Roboflow.
Tujuan: menambah data tutup HITAM (Coca-Cola Zero, Powerade) dan sesi lain ke model tutup (CATATAN C-9).

Foto asli TIDAK diubah. Output datasets/2_dataset_cap/cap_water_v1/:
  images/*.jpg (1920x1080 JPEG q95), labels/*.txt (YOLO, 0=cap 1=no_cap), classes.txt, data.yaml
  prelabel_report.csv (per foto: sumber box, conf, kelas tebakan model, alasan review), review_*.jpg (lembar kontak, prioritas dulu)
Kelas dari NAMA FILE (capon -> cap, capoff -> no_cap). Model hanya memberi LOKASI:
  kandidat = cap_belt_final + cap_v4, skala 1280/960/640, hanya kandidat yang pusatnya di dalam kotak botol (botol_ver8b).
  Tanpa kandidat -> PLACEHOLDER di TENGAH badan botol (pasti salah tempat, wajib digeser ke tutup/leher).
Definisi box sama dengan set 13 Okt: cap = tutup; no_cap = leher + cincin (tanpa tutup), box ketat.
  python scripts/pipeline_cap/prelabel_water.py
"""
import csv, glob, os, re, statistics
import cv2, numpy as np
from ultralytics import YOLO

DS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
SESI = ["2026-10-16_water_v1", "2026-10-20_water_v1"]
OUT = f"{DS}/2_dataset_cap/cap_water_v1"
CAP_MODELS = [f"{DS}/3_model_final/cap_belt_final.pt", f"{DS}/3_model_final/cap_v4_640.pt"]
SIZES = [1280, 960, 640]
HITAM = ("cola-zero", "cocacola-zero", "powerade")             # tutup hitam (C-9): selalu ditinjau


def main():
    os.makedirs(f"{OUT}/images", exist_ok=True); os.makedirs(f"{OUT}/labels", exist_ok=True)
    det = YOLO(f"{DS}/3_model_final/botol_ver8b.pt"); caps = [YOLO(p) for p in CAP_MODELS]
    files = []
    for s in SESI:
        files += [f for f in sorted(glob.glob(f"{DS}/1_foto_mentah/{s}/b*.png")) if re.search(r"_cap(on|off)_\d+\.png$", f)]
    print(len(files), "foto", flush=True)
    recs = []
    for i, f in enumerate(files, 1):
        sesi = os.path.basename(os.path.dirname(f))[5:10]          # 10-16 / 10-20
        base = os.path.splitext(os.path.basename(f))[0]
        name = f"s{sesi.replace('-', '')}_{base}"                  # awalan sesi: nomor botol b01.. berulang antar sesi
        fcls = 0 if "_capon_" in base else 1
        im = cv2.imread(f); H, W = im.shape[:2]
        rb = det.predict(im, imgsz=640, conf=0.25, verbose=False)[0]
        bb = [b for b in rb.boxes if int(b.cls) == 0]
        bx = [float(v) for v in max(bb, key=lambda z: float(z.conf)).xyxy[0]] if bb else [0, 0, W, H]
        mx, my = 0.08 * (bx[2] - bx[0]), 0.08 * (bx[3] - bx[1])
        cands = []
        for mi, m in enumerate(caps):
            for s in SIZES:
                for b in m.predict(im, imgsz=s, conf=0.03, max_det=10, verbose=False)[0].boxes:
                    x = b.xyxy[0].tolist(); cx, cy = (x[0] + x[2]) / 2, (x[1] + x[3]) / 2
                    if bx[0] - mx <= cx <= bx[2] + mx and bx[1] - my <= cy <= bx[3] + my:
                        cands.append((float(b.conf), int(b.cls), x, s, mi))
        same = [c for c in cands if c[1] == fcls and c[0] >= 0.15]
        pick = max(same, key=lambda t: t[0]) if same else (max(cands, key=lambda t: t[0]) if cands else None)
        if pick is None:
            cxy = ((bx[0] + bx[2]) / 2 / W, (bx[1] + bx[3]) / 2 / H); wh = (0.06, 0.12)
            src, conf, mcls, isz, mname = "PLACEHOLDER", 0.0, -1, 0, ""
        else:
            conf, mcls, x, isz, mi = pick
            cxy = ((x[0] + x[2]) / 2 / W, (x[1] + x[3]) / 2 / H); wh = ((x[2] - x[0]) / W, (x[3] - x[1]) / H)
            src, mname = "model", ["cap_belt_final", "cap_v4"][mi]
        cv2.imwrite(f"{OUT}/images/{name}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
        open(f"{OUT}/labels/{name}.txt", "w").write(f"{fcls} {cxy[0]:.6f} {cxy[1]:.6f} {wh[0]:.6f} {wh[1]:.6f}\n")
        recs.append(dict(file=name, sesi=sesi, asli=os.path.relpath(f, DS), fcls=fcls, src=src, model=mname, conf=round(conf, 3),
                         mcls=mcls, imgsz=isz, cx=cxy[0], cy=cxy[1], w=wh[0], h=wh[1], n_cands=len(cands),
                         hitam=any(h in base for h in HITAM)))
        if i % 30 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    mod = [r for r in recs if r["src"] == "model"]
    mw = statistics.median(r["w"] for r in mod); mh = statistics.median(r["h"] for r in mod)
    for r in recs:
        why = []
        if r["src"] == "PLACEHOLDER":
            why.append("PLACEHOLDER")
        if r["src"] == "model" and r["mcls"] != r["fcls"]:
            why.append("kelas_model_beda")
        if r["src"] == "model" and r["conf"] < 0.5:
            why.append("conf<0.5")
        if r["src"] == "model" and (not 0.5 * mw <= r["w"] <= 2 * mw or not 0.5 * mh <= r["h"] <= 2 * mh):
            why.append("ukuran_ganjil")
        if r["hitam"]:
            why.append("tutup_hitam")
        r["review"] = "|".join(why)
    with open(f"{OUT}/prelabel_report.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(recs[0].keys())); w.writeheader(); w.writerows(recs)
    open(f"{OUT}/classes.txt", "w").write("cap\nno_cap\n")
    open(f"{OUT}/data.yaml", "w").write("train: images\nval: images\nnc: 2\nnames: ['cap', 'no_cap']\n")

    def tile(r):
        im = cv2.imread(f"{OUT}/images/{r['file']}.jpg"); H, W = im.shape[:2]
        x1, y1 = int((r["cx"] - r["w"] / 2) * W), int((r["cy"] - r["h"] / 2) * H)
        x2, y2 = int((r["cx"] + r["w"] / 2) * W), int((r["cy"] + r["h"] / 2) * H)
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 200, 0) if r["fcls"] == 0 else (0, 0, 255), 5)
        t = cv2.resize(im, (640, 360))
        cv2.putText(t, r["file"][:44], (5, 18), 0, 0.5, (0, 255, 255), 1)
        cv2.putText(t, ("cap " if r["fcls"] == 0 else "no_cap ") + (r["review"] or f"ok {r['conf']:.2f}"), (5, 350), 0, 0.5,
                    (0, 0, 255) if r["review"] else (255, 255, 255), 2)
        return t
    order = sorted(recs, key=lambda r: (r["review"] == "", "PLACEHOLDER" not in r["review"], r["conf"]))
    for si in range(0, len(order), 12):
        ts = [tile(r) for r in order[si:si + 12]]
        while len(ts) % 3:
            ts.append(np.zeros_like(ts[0]))
        cv2.imwrite(f"{OUT}/review_{si // 12 + 1:02d}.jpg", cv2.vconcat([cv2.hconcat(ts[j:j + 3]) for j in range(0, len(ts), 3)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    cnt = lambda k: sum(k in r["review"] for r in recs)
    print(f"\nSELESAI {len(recs)} foto -> {OUT}\n  box model {len(mod)} | PLACEHOLDER {cnt('PLACEHOLDER')} | kelas beda {cnt('kelas_model_beda')} | "
          f"conf<0.5 {cnt('conf<0.5')} | ukuran ganjil {cnt('ukuran_ganjil')} | tutup hitam {cnt('tutup_hitam')} | "
          f"perlu review {sum(bool(r['review']) for r in recs)} | median box w={mw:.3f} h={mh:.3f}")


if __name__ == "__main__":
    main()
