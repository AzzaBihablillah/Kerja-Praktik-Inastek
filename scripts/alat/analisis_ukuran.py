"""Kategori ukuran dari kamera: luas kotak vs PANJANG botol (sisi panjang kotak) vs akar luas. Bahan keputusan & TA.
Data: foto belt yang ml-nya ada di nama file (13, 16, 20 Okt). Kategori benar: kecil <= 350 ml, tanggung 351-800, besar > 800.
Kotak = botol_ver9c; foto yang kotaknya terpotong tepi frame dibuang. Fitur dinyatakan setara frame 1920 px.
Per botol diambil median. Batas dipilih di data latih lalu diuji LEAVE-ONE-BOTTLE-OUT (batas tidak melihat botol uji).
  python scripts/alat/analisis_ukuran.py   -> datasets/4_eksperimen/ukuran/analisis_ukuran.txt + .csv"""
import csv, glob, os, re
from collections import defaultdict
import cv2, numpy as np
from ultralytics import YOLO

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS = os.path.join(REPO, "datasets"); OUT = os.path.join(DS, "4_eksperimen", "ukuran"); os.makedirs(OUT, exist_ok=True)
SESI = ["2026-10-13_conveyor_v1", "2026-10-16_water_v1", "2026-10-20_water_v1"]
# Koreksi ml dari user (23 Okt): nama file 16 Okt b01 "sprite-nipis-mint-250" sebenarnya botol 390 ml.
ML_KOREKSI = {("10-16", "b01"): 390}


def kategori(ml):
    return "kecil" if ml <= 350 else ("tanggung" if ml <= 800 else "besar")


def ml_of(name):
    tok = name.split("_")[1]
    m = re.search(r"(\d{3,4})", tok)
    return int(m.group(1)) if m and 150 <= int(m.group(1)) <= 2000 else None


def main():
    det = YOLO(f"{DS}/3_model_final/botol_ver9c.pt"); per = defaultdict(list)
    for s in SESI:
        for f in sorted(glob.glob(f"{DS}/1_foto_mentah/{s}/b*.png")):
            n = os.path.basename(f); ml = ml_of(n)
            if ml is None:
                continue
            im = cv2.imread(f); H, W = im.shape[:2]; k = 1920 / W
            bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
            if not bb:
                continue
            x1, y1, x2, y2 = max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            if x1 < 4 or x2 > W - 4:
                continue
            w, h = (x2 - x1) * k, (y2 - y1) * k
            ml = ML_KOREKSI.get((s[5:10], n.split("_")[0]), ml)
            per[(s[5:10], n.split("_")[0], n.split("_")[1], ml)].append((w * h, max(w, h), min(w, h)))
    rows = [dict(sesi=key[0], botol=key[1], nama=key[2], ml=key[3], kat=kategori(key[3]),
                 luas=float(np.median([v[0] for v in vs])), panjang=float(np.median([v[1] for v in vs])),
                 lebar=float(np.median([v[2] for v in vs])), n=len(vs)) for key, vs in per.items()]
    with open(f"{OUT}/analisis_ukuran.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    order = {"kecil": 0, "tanggung": 1, "besar": 2}

    def fit(train, feat):
        """batas t1<t2 meminimalkan salah di train (kandidat = titik tengah nilai berurutan)."""
        v = sorted(r[feat] for r in train); cand = [(a + b) / 2 for a, b in zip(v, v[1:])] or v
        best = None
        for i, t1 in enumerate(cand):
            for t2 in cand[i:]:
                err = sum(order[r["kat"]] != (0 if r[feat] < t1 else (1 if r[feat] < t2 else 2)) for r in train)
                if best is None or err < best[0]:
                    best = (err, t1, t2)
        return best[1], best[2]

    L = [f"{len(rows)} botol (ml dari nama file; {sum(r['n'] for r in rows)} foto). Kategori: kecil<=350, tanggung 351-800, besar>800 ml.",
         "Uji leave-one-bottle-out (batas dipilih tanpa botol uji).", ""]
    for feat in ("luas", "panjang", "lebar"):
        ok = 0; salah = []
        for i, r in enumerate(rows):
            t1, t2 = fit(rows[:i] + rows[i + 1:], feat)
            p = 0 if r[feat] < t1 else (1 if r[feat] < t2 else 2)
            if p == order[r["kat"]]:
                ok += 1
            else:
                salah.append(f"{r['nama']} {r['ml']}ml->{['kecil', 'tanggung', 'besar'][p]}")
        t1, t2 = fit(rows, feat)
        L.append(f"{feat:8s}: benar {ok}/{len(rows)} | batas (semua data) {t1:,.0f} / {t2:,.0f} | salah: {', '.join(salah) or '-'}")
    L.append("\nPer botol (urut ml):")
    for r in sorted(rows, key=lambda r: r["ml"]):
        L.append(f"  {r['sesi']} {r['botol']} {r['nama'][:22]:22s} {r['ml']:5d} ml {r['kat']:8s} luas {r['luas'] / 1000:6.0f}k  panjang {r['panjang']:5.0f}  lebar {r['lebar']:4.0f}")
    open(f"{OUT}/analisis_ukuran.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
