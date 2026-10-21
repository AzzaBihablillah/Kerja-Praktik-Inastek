"""Langkah 24: cek JALAN PINTAS SESI. Bisakah model membedakan sesi pemotretan (13 Okt vs 16 Okt) tanpa melihat isi cairan?
  (a) potongan BOTOL KOSONG saja: C13 (13 Okt, 30 botol) vs W16 kosong (16 Okt, 5 botol)
  (b) potongan BELT KOSONG saja (pita belt di atas/bawah botol, tanpa botol)
CV 5 fold per botol (W16 botol k + C13 grup k), yolov8n-cls 30 epoch, kelas diseimbangkan. AUC tinggi = sesi mudah dikenali
= positif (semua 16 Okt) vs negatif (dominan 13 Okt) bisa dipisah tanpa melihat cairan.
Keluaran: out/session_check.json + .txt"""
import csv, json, math, os, shutil, sys
import cv2, numpy as np
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); BG = f"{OUT}/crops_bg"; os.makedirs(BG, exist_ok=True)


def auc(pos, neg):
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def main():
    idx = list(csv.DictReader(open(f"{OUT}/belt_index.csv")))
    rows = [r for r in csv.DictReader(open(f"{OUT}/crops.csv")) if (r["src"] == "C13") or (r["src"] == "W16" and r["liquid"] == "kosong")]
    # potongan belt kosong: pita di atas kotak botol (atau di bawah bila tak cukup), 224x224
    for i, r in enumerate(idx):
        cid = f"{r['src']}_{i:04d}"
        if r["src"] not in ("C13", "W16") or os.path.exists(f"{BG}/{cid}.jpg"):
            continue
        im = cv2.imread(os.path.join(ROOT, r["path"])); H, W = im.shape[:2]
        x1, y1, x2, y2 = [int(float(r[k])) for k in ("x1", "y1", "x2", "y2")]
        cx = (x1 + x2) // 2; s = 220
        top = (max(0, y1 - 10 - s), y1 - 10); bot = (y2 + 10, min(H, y2 + 10 + s))
        ya, yb = top if top[1] - top[0] >= bot[1] - bot[0] else bot
        xa = int(np.clip(cx - s // 2, 0, W - s))
        patch = im[ya:yb, xa:xa + s]
        cv2.imwrite(f"{BG}/{cid}.jpg", cv2.resize(patch, (224, 224)), [cv2.IMWRITE_JPEG_QUALITY, 95])
    c13g = sorted({r["group"] for r in rows if r["src"] == "C13"}, key=lambda g: int(g.split("_b")[1]))
    c13f = {g: (i % 5) + 1 for i, g in enumerate(c13g)}
    fold = lambda r: int(r["group"][-2:]) if r["src"] == "W16" else c13f[r["group"]]
    lab = lambda r: "s16" if r["src"] == "W16" else "s13"
    res = {}
    for variant, path in (("botol_kosong", lambda r: f"{OUT}/{r['crop'].replace('crops/', 'crops_fold/')}"), ("belt_saja", lambda r: f"{BG}/{r['id']}.jpg")):
        probs = []
        for k in range(1, 6):
            d = f"{OUT}/tmp_sess/{variant}_f{k}"; shutil.rmtree(d, ignore_errors=True)
            tr = [r for r in rows if fold(r) != k]
            n = {c: sum(lab(r) == c for r in tr) for c in ("s13", "s16")}; big = max(n.values())
            for c in ("s13", "s16"):
                os.makedirs(f"{d}/train/{c}"); os.makedirs(f"{d}/val/{c}")
                rep = math.ceil(big / n[c])
                for r in tr:
                    if lab(r) == c:
                        for j in range(rep):
                            os.link(path(r), f"{d}/train/{c}/{r['id']}_{j}.jpg")
            te = [r for r in rows if fold(r) == k]
            for r in te:
                os.link(path(r), f"{d}/val/{lab(r)}/{r['id']}.jpg")
            YOLO(os.path.join(ROOT, "3_model_final/dasar/yolov8n-cls.pt")).train(data=d, epochs=30, imgsz=224, batch=32, seed=0, deterministic=True, workers=0,
                                                             verbose=False, plots=False, project=f"{OUT}/runs_sess", name=f"{variant}_f{k}", exist_ok=True,
                                                             patience=0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, fliplr=0.5, flipud=0.5, auto_augment="", erasing=0.1)
            m = YOLO(f"{OUT}/runs_sess/{variant}_f{k}/weights/last.pt"); c16 = [i for i, v in m.names.items() if v == "s16"][0]
            for r, p in zip(te, m.predict([path(r) for r in te], imgsz=224, verbose=False)):
                probs.append((lab(r), float(p.probs.data[c16])))
            shutil.rmtree(d, ignore_errors=True)
        pos = [p for l, p in probs if l == "s16"]; neg = [p for l, p in probs if l == "s13"]
        acc = (sum(p >= .5 for p in pos) + sum(p < .5 for p in neg)) / len(probs)
        res[variant] = dict(auc=round(auc(pos, neg), 3), akurasi=round(acc, 3), n16=len(pos), n13=len(neg),
                            s16_benar=sum(p >= .5 for p in pos), s13_benar=sum(p < .5 for p in neg))
        print(variant, res[variant], flush=True)
    json.dump(res, open(f"{OUT}/session_check.json", "w"), indent=1)
    open(f"{OUT}/session_check.txt", "w").write("\n".join(f"{k}: {v}" for k, v in res.items()) + "\n")
    print("SESI SELESAI")


if __name__ == "__main__":
    main()
