"""Model merek langkah 2: model DASAR dari foto belt kita saja (label produk divalidasi user 23 Okt: 8 mitra, 29 bukan).
Kelas: 'mitra' (semua produk Coca-Cola Company) vs 'bukan'. Input: potongan kotak botol (botol_ver9c), sumbu panjang mendatar,
TIDAK dilipat (label di tengah botol tidak terbelah), letterbox abu 114 ke 256x256. YOLOv8n-cls, 30 epoch, seimbang, seed 0.
Dua cara CV (5 fold):
  F = botol/sesi ditahan (produk sudah dikenal, foto/botol baru)  -> kondisi nyata di mesin
  P = PRODUK ditahan (produk tidak pernah dilihat)                 -> apakah model mengenali ciri Coca-Cola, bukan hafalan produk
  python scripts/pipeline_merek/m02_cv_dasar.py   -> datasets/4_eksperimen/merek/cv_dasar.jsonl + cv_dasar_report.txt"""
import csv, json, math, os, shutil, sys
from collections import defaultdict
import cv2, numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
ROOT = os.path.join(REPO, "datasets")
OUT = f"{ROOT}/4_eksperimen/merek"; CR = f"{OUT}/crops"; S = 256
LABEL_USER = {"botol_17_tak_dikenal": "B"}                    # validasi user 23 Okt (lainnya = usulan, semua ACC)
from rvm_kp.merek import brand_crop                  # satu sumber: dipakai juga oleh app/ (kandidat) dan Jetson


def rows():
    R = list(csv.DictReader(open(f"{OUT}/foto.csv")))
    for i, r in enumerate(R):
        r["id"] = f"M{i:04d}"; lab = LABEL_USER.get(r["produk"], r["usul"])
        r["kelas"] = "mitra" if lab == "M" else "bukan"
        r["grup_F"] = (r["produk"] + "_scan") if r["sesi"] == "scan" else f"{r['sesi']}_{r['botol']}"
        r["grup_P"] = r["produk"]
    return R


def make_crops(R):
    from ultralytics import YOLO
    os.makedirs(CR, exist_ok=True); det = YOLO(f"{ROOT}/3_model_final/botol_ver9c.pt"); keep = []
    for r in R:
        dst = f"{CR}/{r['id']}.jpg"
        if not os.path.exists(dst):
            im = cv2.imread(f"{ROOT}/{r['path']}")
            bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
            if not bb:
                continue
            cv2.imwrite(dst, brand_crop(im, *max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))), [cv2.IMWRITE_JPEG_QUALITY, 95])
        keep.append(r)
    return keep


def folds(R, key, K=5):
    cnt = defaultdict(int)
    for r in R:
        cnt[r[key]] += 1
    F, load = {}, [0] * K
    # sebar kelompok mitra dulu agar tiap fold punya produk mitra & bukan
    order = sorted(cnt, key=lambda g: (-(any(r["kelas"] == "mitra" and r[key] == g for r in R)), -cnt[g]))
    for g in order:
        k = min(range(K), key=lambda i: load[i]); F[g] = k; load[k] += cnt[g]
    return F


def train(items, d, project, name, epochs=30, cache=False):
    from ultralytics import YOLO
    shutil.rmtree(d, ignore_errors=True)
    cls = sorted({l for _, l, _ in items})               # mitra/bukan (CV) atau coca_cola/non_coca_cola (model final)
    n = {c: sum(l == c for _, l, _ in items) for c in cls}; big = max(n.values())
    for c in cls:
        os.makedirs(f"{d}/train/{c}"); os.makedirs(f"{d}/val/{c}")
        for p, l, i in items:
            if l == c:
                for k in range(math.ceil(big / n[c])):
                    os.link(p, f"{d}/train/{c}/{i}_{k}.jpg")
        for p, l, i in [x for x in items if x[1] == c][:6]:
            os.link(p, f"{d}/val/{c}/{i}.jpg")
    YOLO(os.path.join(ROOT, "3_model_final/dasar/yolov8n-cls.pt")).train(
        data=d, epochs=epochs, imgsz=S, cache=cache, batch=32, seed=0, deterministic=True, workers=0, verbose=False, plots=False, project=project,
        name=name, exist_ok=True, patience=0, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, fliplr=0.0, flipud=0.5, degrees=0, auto_augment="", erasing=0.1)
    shutil.rmtree(d, ignore_errors=True)
    return f"{project}/{name}/weights/last.pt"


def main():
    from ultralytics import YOLO
    R = make_crops(rows()); res = f"{OUT}/cv_dasar.jsonl"
    done = {(json.loads(l)["skema"], json.loads(l)["fold"]) for l in open(res)} if os.path.exists(res) else set()
    print(len(R), "foto;", sum(r["kelas"] == "mitra" for r in R), "mitra", flush=True)
    for skema, key in (("F", "grup_F"), ("P", "grup_P")):
        F = folds(R, key)
        for k in range(5):
            if (skema, k) in done:
                continue
            tr = [r for r in R if F[r[key]] != k]; te = [r for r in R if F[r[key]] == k]
            pt = train([(f"{CR}/{r['id']}.jpg", r["kelas"], r["id"]) for r in tr], f"{OUT}/tmp/{skema}{k}", f"{OUT}/runs", f"dasar_{skema}{k}")
            m = YOLO(pt); cm = [i for i, v in m.names.items() if v == "mitra"][0]
            ps = [float(p.probs.data[cm]) for p in m.predict([f"{CR}/{r['id']}.jpg" for r in te], imgsz=S, verbose=False)]
            with open(res, "a") as fh:
                for r, p in zip(te, ps):
                    fh.write(json.dumps(dict(skema=skema, fold=k, id=r["id"], produk=r["produk"], sesi=r["sesi"], kelas=r["kelas"], p=round(p, 4))) + "\n")
            print(f"{skema} fold {k}: latih {len(tr)}, uji {len(te)}", flush=True)
    report(res)


def report(res):
    R = [json.loads(l) for l in open(res) if l.strip()]
    L = ["Model merek DASAR (foto belt kita, label tervalidasi). Ambang 0,5. mitra = semua produk Coca-Cola Company.", ""]
    for skema, judul in (("F", "F: produk dikenal, botol/sesi baru (kondisi mesin)"), ("P", "P: produk TIDAK pernah dilihat")):
        rr = [r for r in R if r["skema"] == skema]
        mt = [r for r in rr if r["kelas"] == "mitra"]; bk = [r for r in rr if r["kelas"] == "bukan"]
        L.append(f"== {judul}: mitra terdeteksi {sum(r['p'] >= .5 for r in mt)}/{len(mt)} | bukan-mitra benar {sum(r['p'] < .5 for r in bk)}/{len(bk)}")
        per = defaultdict(list)
        for r in rr:
            per[(r["kelas"], r["produk"])].append(r["p"])
        for (kl, pr), ps in sorted(per.items()):
            ok = sum((p >= .5) == (kl == "mitra") for p in ps)
            if kl == "mitra" or ok < len(ps):
                L.append(f"   {'MITRA' if kl == 'mitra' else 'bukan'} {pr:24s} benar {ok}/{len(ps)}")
        L.append("")
    open(f"{OUT}/cv_dasar_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
