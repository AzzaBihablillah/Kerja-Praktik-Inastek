"""Langkah 3: classifier crop (YOLOv8n-cls) 'berwarna' vs 'kosong', CV 5 fold per BOTOL.

Fold k: uji = botol W16 ke-k (semua levelnya, termasuk seperempat utk dilaporkan) + 6 botol C13 grup k.
        latih = sisa W16 (tanpa seperempat) + sisa C13 [+ WEB Fanta]. P15 TIDAK pernah dilatih -> uji lintas sesi.
Tanpa pemilihan epoch (pakai last.pt). Augmentasi tanpa geser hue dan tanpa RandAugment (warna = label).
Kelas minoritas digandakan agar seimbang.

  python s03_cls_cv.py --name V2_fold_belt --crops crops_fold [--web] [--no_c13] [--epochs 30]
Keluaran: out/cls_preds.jsonl (satu baris per foto uji per fold per varian)
"""
import argparse, csv, json, os, shutil
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--crops", default="crops_fold")
ap.add_argument("--web", action="store_true")
ap.add_argument("--no_c13", action="store_true")
ap.add_argument("--ext", action="store_true", help="tambah crop Open Food Facts (pos dan neg dari sumber yang sama; label dari jenis minuman, belum divalidasi)")
ap.add_argument("--glare", action="store_true", help="tambah salinan crop berkilau sintetis (s11) utk data LATIH")
ap.add_argument("--soda", action="store_true", help="tambah 600 crop botol penuh RF100 soda-bottles (s12) sbg positif latih")
ap.add_argument("--old", action="store_true", help="tambah crop data warisan (pilah visual Claude) ke data latih")
ap.add_argument("--epochs", type=int, default=30)
ap.add_argument("--imgsz", type=int, default=224)
ap.add_argument("--model", default="3_model_final/dasar/yolov8n-cls.pt")
ap.add_argument("--folds", default="1,2,3,4,5")
a = ap.parse_args()

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan")
rows = list(csv.DictReader(open(f"{OUT}/crops.csv")))
if a.old:
    rows += list(csv.DictReader(open(f"{OUT}/crops_old.csv")))
if a.soda:
    rows += list(csv.DictReader(open(f"{OUT}/crops_soda.csv")))
if a.ext:
    rows += list(csv.DictReader(open(f"{OUT}/crops_ext.csv", encoding="utf-8")))
c13_groups = sorted({r["group"] for r in rows if r["src"] == "C13"}, key=lambda g: int(g.split("_b")[1]))
c13_fold = {g: (i % 5) + 1 for i, g in enumerate(c13_groups)}            # b01->1, b02->2, ... b06->1 ...


def crop_path(r):
    return os.path.join(OUT, r["crop"].replace("crops/", a.crops + "/"))


def fold_of(r):
    if r["src"] == "W16":
        return int(r["group"][-2:])
    if r["src"] == "C13":
        return c13_fold[r["group"]]
    return 0                                                                # WEB / P15: tidak punya fold


def build(fold, d):
    if os.path.exists(d):
        shutil.rmtree(d)
    tr = []
    for r in rows:
        if r["liquid"] not in ("berwarna", "kosong"):
            continue
        if r["src"] == "P15" or (r["src"] == "WEB" and not a.web) or (r["src"] == "C13" and a.no_c13) or (r["src"] == "OLD" and not a.old) or (r["src"] == "SODA" and not a.soda) or (r["src"] == "EXT" and not a.ext):
            continue
        if fold_of(r) == fold:
            continue
        tr.append(r)
    n = {c: sum(1 for r in tr if r["liquid"] == c) for c in ("berwarna", "kosong")}
    big = max(n.values())
    for c in ("berwarna", "kosong"):
        os.makedirs(f"{d}/train/{c}", exist_ok=True); os.makedirs(f"{d}/val/{c}", exist_ok=True)
        items = [r for r in tr if r["liquid"] == c]
        rep = min(6, max(1, round(big / max(1, len(items)))))
        for r in items:
            srcs = [("", crop_path(r))] + ([("g", os.path.join(OUT, "crops_fold_glare", os.path.basename(crop_path(r))))] if a.glare else [])
            for tag, pth in srcs:
                for k in range(rep):
                    os.link(pth, f"{d}/train/{c}/{r['id']}{tag}_{k}.jpg")
    for r in rows:                                                          # val = uji fold ini (hanya utk memenuhi format; tak dipakai memilih model)
        if fold_of(r) == fold and r["liquid"] in ("berwarna", "kosong") and r["src"] in ("W16", "C13"):
            os.link(crop_path(r), f"{d}/val/{r['liquid']}/{r['id']}.jpg")
    return n


def main():
    pred_file = f"{OUT}/cls_preds.jsonl"
    for fold in [int(x) for x in a.folds.split(",")]:
        d = os.path.join(OUT, "tmp_cls", f"{a.name}_f{fold}")
        n = build(fold, d)
        print(f"== {a.name} fold {fold}: latih {n}", flush=True)
        m = YOLO(os.path.join(ROOT, a.model))
        m.train(data=d, epochs=a.epochs, imgsz=a.imgsz, batch=32, seed=0, deterministic=True, workers=0, verbose=False, plots=False,
                project=os.path.join(OUT, "runs_cls"), name=f"{a.name}_f{fold}", exist_ok=True, patience=0,
                hsv_h=0.0, hsv_s=0.2, hsv_v=0.3, fliplr=0.5, flipud=0.5, degrees=0, auto_augment="", erasing=0.1)
        wt = os.path.join(OUT, "runs_cls", f"{a.name}_f{fold}", "weights", "last.pt")
        pm = YOLO(wt)
        test = [r for r in rows if (fold_of(r) == fold and r["src"] in ("W16", "C13")) or r["src"] == "P15"]
        with open(pred_file, "a") as fh:
            for i in range(0, len(test), 64):
                batch = test[i:i + 64]
                res = pm.predict([crop_path(r) for r in batch], imgsz=a.imgsz, verbose=False)
                for r, p in zip(batch, res):
                    names = p.names; probs = p.probs.data.tolist()
                    pb = probs[[k for k, v in names.items() if v == "berwarna"][0]]
                    fh.write(json.dumps(dict(variant=a.name, fold=fold, id=r["id"], src=r["src"], group=r["group"], liquid=r["liquid"],
                                             level=r["level"], p_berwarna=round(pb, 4))) + "\n")
        shutil.rmtree(d, ignore_errors=True)
    print("selesai", a.name)


if __name__ == "__main__":                                  # wajib di Windows: worker dataloader meng-impor ulang skrip ini
    main()
