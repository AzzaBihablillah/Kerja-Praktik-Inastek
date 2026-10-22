"""Langkah 32: apakah data 22 Okt (botol berlabel penuh + frame diam aplikasi di belt) memperbaiki model cairan?
CV 5 fold per KELOMPOK (kelompok cairan lama s28 + kelompok botol fisik baru s31). Per fold, dua resep, semuanya sama kecuali data:
  RH : data lama saja (W16 + V20) + geser-hue          (= resep model aktif cairan_RH)
  RN : data lama + data baru 22 Okt (V22 + S22) + geser-hue
Uji = semua foto kelompok yang ditahan (lama & baru). Seimbang (ceil), 30 epoch, crop lipat, seed 0 (sama dgn s28/s30).
  python scripts/pipeline_cairan/s32_cv_rn.py          -> out/cv_rn.jsonl + out/cv_rn_report.txt (resumable)
  python scripts/pipeline_cairan/s32_cv_rn.py final    -> out/final4/cls4_RN_all -> datasets/3_model_final/cairan_RN.pt"""
import csv, json, math, os, shutil, sys
import numpy as np
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); RES = f"{OUT}/cv_rn.jsonl"; K = 5
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
LIQ = {"V20_b01": "powerade", "V20_b07": "powerade", "V20_b14": "powerade", "V20_b02": "fanta_anggur", "V20_b08": "fanta_anggur",
       "V20_b03": "soy", "V20_b09": "soy", "V20_b04": "minutemaid", "V20_b05": "cola_vanila", "V20_b06": "cola_vanila",
       "V20_b10": "nutri_jeruk", "V20_b12": "nutri_jeruk", "V20_b15": "nutri_jeruk", "V20_b11": "nutri_hijau", "V20_b13": "nutri_hijau",
       "V20_b16": "sunlight"}


def rows_all():
    old = [r for r in list(csv.DictReader(open(f"{OUT}/crops.csv"))) + list(csv.DictReader(open(f"{OUT}/v20_crops.csv")))
           if r["src"] in ("W16", "V20") and r["liquid"] in ("berwarna", "kosong")]
    for r in old:
        r["kel"] = LIQ.get(r["group"], r["group"]); r["baru"] = 0
    new = list(csv.DictReader(open(f"{OUT}/crops22.csv")))
    for r in new:
        r["kel"] = r["group"]; r["baru"] = 1
    return old + new


def folds(rows):
    cnt = {}
    for r in rows:
        cnt[r["kel"]] = cnt.get(r["kel"], 0) + 1
    F, load = {}, [0] * K
    for g in sorted(cnt, key=lambda g: -cnt[g]):                 # kelompok besar dulu, ke fold paling ringan
        k = min(range(K), key=lambda i: load[i]); F[g] = k; load[k] += cnt[g]
    return F, load


def items_of(rows):
    it = [(f"{OUT}/crops_fold/{r['id']}.jpg", r["liquid"], r["id"]) for r in rows]
    it += [(f"{OUT}/crops_fold_hue/{r['id']}_h{k}.jpg", r["liquid"], f"{r['id']}_h{k}") for r in rows for k in range(3)
           if os.path.exists(f"{OUT}/crops_fold_hue/{r['id']}_h{k}.jpg")]
    return it


def train(items, d, project, name):
    shutil.rmtree(d, ignore_errors=True)
    n = {c: sum(l == c for _, l, _ in items) for c in ("berwarna", "kosong")}; big = max(n.values())
    for c in ("berwarna", "kosong"):
        os.makedirs(f"{d}/train/{c}"); os.makedirs(f"{d}/val/{c}")
        for p, l, i in items:
            if l == c:
                for k in range(math.ceil(big / n[c])):
                    os.link(p, f"{d}/train/{c}/{i}_{k}.jpg")
        for p, l, i in [x for x in items if x[1] == c][:6]:          # val hanya utk format, bukan seleksi
            os.link(p, f"{d}/val/{c}/{i}.jpg")
    YOLO(os.path.join(ROOT, "3_model_final/dasar/yolov8n-cls.pt")).train(
        data=d, epochs=30, imgsz=224, batch=32, seed=0, deterministic=True, workers=0, verbose=False, plots=False,
        project=project, name=name, exist_ok=True, patience=0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, fliplr=0.5, flipud=0.5,
        degrees=0, auto_augment="", erasing=0.1)
    shutil.rmtree(d, ignore_errors=True)
    return f"{project}/{name}/weights/last.pt"


def auc(pos, neg):
    if not pos or not neg:
        return float("nan")
    pos, neg = np.array(pos), np.array(neg)
    return float(((pos[:, None] > neg[None, :]).sum() + 0.5 * (pos[:, None] == neg[None, :]).sum()) / (len(pos) * len(neg)))


def report():
    R = [json.loads(l) for l in open(RES) if l.strip()]
    L = ["CV 5 fold per kelompok (cairan lama + botol fisik baru 22 Okt). Ambang 0,5. RH = data lama saja; RN = + data 22 Okt.", ""]
    subs = [("SEMUA", lambda r: True), ("data LAMA (W16+V20, 16/20 Okt)", lambda r: r["baru"] == 0),
            ("V22 botol label penuh (capture 22 Okt)", lambda r: r["src"] == "V22"), ("S22 frame diam aplikasi di belt", lambda r: r["src"] == "S22")]
    for name, f in subs:
        L.append(f"== {name}")
        for rc in ("RH", "RN"):
            rr = [x for x in R if x["resep"] == rc and f(x)]
            pos = [x["p"] for x in rr if x["liquid"] == "berwarna"]; neg = [x["p"] for x in rr if x["liquid"] == "kosong"]
            L.append(f"   {rc}: AUC {auc(pos, neg):.3f} | berisi terdeteksi {sum(p >= .5 for p in pos)}/{len(pos)} | kosong benar {sum(p < .5 for p in neg)}/{len(neg)}")
    L.append("\nPer kelompok BARU (berisi terdeteksi / kosong benar), RH -> RN:")
    for g in sorted({x["kel"] for x in R if x["baru"]}):
        s = []
        for rc in ("RH", "RN"):
            rr = [x for x in R if x["resep"] == rc and x["kel"] == g]
            pos = [x["p"] for x in rr if x["liquid"] == "berwarna"]; neg = [x["p"] for x in rr if x["liquid"] == "kosong"]
            s.append(f"{sum(p >= .5 for p in pos)}/{len(pos)} | {sum(p < .5 for p in neg)}/{len(neg)}")
        L.append(f"   {g:20s} {s[0]:>14s}  ->  {s[1]}")
    open(f"{OUT}/cv_rn_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    rows = rows_all(); F, load = folds(rows)
    print(len(rows), "foto,", len(F), "kelompok; beban fold", load, flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        pt = train(items_of(rows), f"{OUT}/final4/data_RN", f"{OUT}/final4", "cls4_RN_all")
        shutil.copy2(pt, f"{ROOT}/3_model_final/cairan_RN.pt"); print("FINAL ->", f"{ROOT}/3_model_final/cairan_RN.pt"); return
    done = {(json.loads(l)["fold"], json.loads(l)["resep"]) for l in open(RES)} if os.path.exists(RES) else set()
    for k in range(K):
        te = [r for r in rows if F[r["kel"]] == k]
        for rc in ("RH", "RN"):
            if (k, rc) in done:
                continue
            tr = [r for r in rows if F[r["kel"]] != k and (rc == "RN" or r["baru"] == 0)]
            pt = train(items_of(tr), f"{OUT}/tmp_rn/{rc}_{k}", f"{OUT}/runs_rn", f"{rc}_f{k}")
            m = YOLO(pt); cb = [i for i, v in m.names.items() if v == "berwarna"][0]
            ps = [float(p.probs.data[cb]) for p in m.predict([f"{OUT}/crops_fold/{r['id']}.jpg" for r in te], imgsz=224, verbose=False)]
            with open(RES, "a") as fh:
                for r, p in zip(te, ps):
                    fh.write(json.dumps(dict(fold=k, resep=rc, id=r["id"], src=r["src"], kel=r["kel"], baru=r["baru"], liquid=r["liquid"], p=round(p, 4))) + "\n")
            print(f"fold {k} {rc}: latih {len(tr)}, uji {len(te)}", flush=True)
    report()


if __name__ == "__main__":
    main()
