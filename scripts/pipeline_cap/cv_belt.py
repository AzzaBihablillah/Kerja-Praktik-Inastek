"""Grouped 5-fold cross-validation (per BOTOL) untuk cap/no_cap di foto belt.

30 botol -> 5 fold x 6 botol. Tiap fold: latih dari cap_v4 pada 24 botol, uji pada 6 botol yg ditahan.
Epoch TETAP (tanpa val/early-stop/pemilihan epoch -> tidak ada bias seleksi); model uji = last.pt.
Warna tutup disebar rata antar fold (putih b17/b21 di fold berbeda).
Per fold & model dicatat per-foto: kelas top-1 benar?, IoU thd box koreksi. Pembanding: baseline cap_v4 (tanpa fine-tune).
Output: datasets/4_eksperimen/cap/_cv_belt/cv_records.jsonl, cv_summary.txt. Resumable (fold yg sudah ada dilewati).
env: EPOCHS (30) IMGSZ (960)
"""
import glob, json, math, os, shutil, collections
from multiprocessing import freeze_support
import numpy as np
from ultralytics import YOLO

DS = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets"
CV = f"{DS}/4_eksperimen/cap/_cv_belt"
BASE = f"{DS}/4_eksperimen/cap/cap_runs/cap_v4_640/weights/best.pt"
EPOCHS = int(os.environ.get("EPOCHS", "30"))
IMGSZ = int(os.environ.get("IMGSZ", "960"))
K = 5
# kelompok warna tutup (diukur dari box koreksi) -> disebar round-robin ke fold
GROUPS = [[17, 21], [6, 13, 23], [16, 24, 25], [15, 26], [5, 14, 20], [11, 12, 22, 27, 28, 29],
          [1, 2, 3, 4, 7, 8, 9, 10, 18, 19, 30]]
FOLD = {}
i = 0
for g in GROUPS:
    for b in sorted(g):
        FOLD[b] = i % K
        i += 1
assert len(FOLD) == 30 and all(sum(1 for v in FOLD.values() if v == k) == 6 for k in range(K))


def link(src, dst):
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(k):
    d = f"{CV}/fold{k}"
    if os.path.exists(d):
        shutil.rmtree(d)
    for s in ("train", "test"):
        os.makedirs(f"{d}/images/{s}"); os.makedirs(f"{d}/labels/{s}")
    for ip in sorted(glob.glob(f"{DS}/2_dataset_cap/cap_belt_ds/images/*/*.jpg")):
        name = os.path.basename(ip)[:-4]
        sp = "test" if FOLD[int(name[1:3])] == k else "train"
        link(ip, f"{d}/images/{sp}/{name}.jpg")
        orig_split = os.path.basename(os.path.dirname(ip))             # train/val/test asal di cap_belt_ds
        shutil.copy2(f"{DS}/2_dataset_cap/cap_belt_ds/labels/{orig_split}/{name}.txt", f"{d}/labels/{sp}/{name}.txt")
    open(f"{d}/data.yaml", "w").write(
        f"path: {d}\ntrain: images/train\nval: images/test\ntest: images/test\nnames:\n  0: cap\n  1: no_cap\n")
    return d


def iou(a, b):
    ax1, ay1, ax2, ay2 = a[0] - a[2] / 2, a[1] - a[3] / 2, a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1, bx2, by2 = b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2
    iw, ih = max(0, min(ax2, bx2) - max(ax1, bx1)), max(0, min(ay2, by2) - max(ay1, by1))
    i = iw * ih
    return i / (a[2] * a[3] + b[2] * b[3] - i + 1e-9)


def evaluate(model, d, tag, k, imgsz, with_map):
    imgs = sorted(glob.glob(f"{d}/images/test/*.jpg"))
    res = model.predict(imgs, imgsz=imgsz, conf=0.03, max_det=10, verbose=False)
    recs = []
    for ip, r in zip(imgs, res):
        name = os.path.basename(ip)[:-4]
        gt = [float(x) for x in open(f"{d}/labels/test/{name}.txt").read().split()[:5]]
        rec = dict(fold=k, model=tag, imgsz=imgsz, file=name, bottle=int(name[1:3]), cls=int(gt[0]),
                   pred=-1, conf=0.0, iou=0.0, correct=False)
        if len(r.boxes):
            b = max(r.boxes, key=lambda z: float(z.conf))
            rec.update(pred=int(b.cls), conf=float(b.conf), iou=iou(gt[1:], [float(v) for v in b.xywhn[0]]))
            rec["correct"] = rec["pred"] == rec["cls"]
        recs.append(rec)
    if with_map:
        m = model.val(data=f"{d}/data.yaml", split="test", imgsz=imgsz, batch=4, device=0, verbose=False, plots=False,
                      project=f"{CV}/_val", name=f"{tag}_f{k}", exist_ok=True)
        for r in recs:
            r["map50"], r["map5095"] = float(m.box.map50), float(m.box.map)
    return recs


def main():
    os.makedirs(CV, exist_ok=True)
    out = f"{CV}/cv_records.jsonl"
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["fold"] for l in open(out) if l.strip()}
    print("fold -> botol:", {k: sorted(b for b, v in FOLD.items() if v == k) for k in range(K)}, flush=True)
    for k in range(K):
        if k in done:
            print(f"fold {k} sudah ada, dilewati", flush=True); continue
        d = build(k)
        m = YOLO(BASE)
        m.train(data=f"{d}/data.yaml", imgsz=IMGSZ, epochs=EPOCHS, batch=8, patience=0, val=False, seed=0, device=0,
                workers=2, cache=False, close_mosaic=10, degrees=0.0, scale=0.3, translate=0.1, fliplr=0.5, flipud=0.0,
                project=f"{CV}/runs", name=f"fold{k}", plots=False, exist_ok=True, verbose=False)
        last = f"{CV}/runs/fold{k}/weights/last.pt"
        recs = []
        recs += evaluate(YOLO(BASE), d, "baseline_v4", k, IMGSZ, True)
        recs += evaluate(YOLO(BASE), d, "baseline_v4", k, 640, False)
        recs += evaluate(YOLO(last), d, "finetune", k, IMGSZ, True)
        recs += evaluate(YOLO(last), d, "finetune", k, 640, False)
        with open(out, "a") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        for tag in ("baseline_v4", "finetune"):
            rr = [r for r in recs if r["model"] == tag and r["imgsz"] == IMGSZ]
            print(f"  fold {k} {tag:12s} imgsz {IMGSZ}: benar {sum(r['correct'] for r in rr)}/{len(rr)}", flush=True)
        shutil.rmtree(d, ignore_errors=True)               # hemat disk (hard link; data asli aman)
    summarize(out)


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def summarize(out):
    recs = [json.loads(l) for l in open(out) if l.strip()]
    lines = []
    for imgsz in sorted({r["imgsz"] for r in recs}, reverse=True):
        for tag in ("baseline_v4", "finetune"):
            rr = [r for r in recs if r["model"] == tag and r["imgsz"] == imgsz]
            if not rr:
                continue
            n = len(rr); ok = sum(r["correct"] for r in rr); lo, hi = wilson(ok, n)
            lines.append(f"\n[{tag} | imgsz {imgsz}] semua {n} foto: benar {ok}/{n} = {100*ok/n:.1f}%  (95% CI {100*lo:.1f}-{100*hi:.1f}%)")
            for c, nm in ((0, "cap"), (1, "no_cap")):
                x = [r for r in rr if r["cls"] == c]; o = sum(r["correct"] for r in x); lo, hi = wilson(o, len(x))
                lines.append(f"    {nm:6s}: {o}/{len(x)} = {100*o/len(x):.1f}% (CI {100*lo:.1f}-{100*hi:.1f}%) | mean IoU {np.mean([r['iou'] for r in x]):.3f} | IoU>=0.5 {sum(r['iou']>=0.5 for r in x)}/{len(x)}")
            folds = collections.defaultdict(list)
            for r in rr:
                folds[r["fold"]].append(r["correct"])
            lines.append("    per fold: " + " ".join(f"f{k}={sum(v)}/{len(v)}" for k, v in sorted(folds.items())))
            if imgsz == IMGSZ and "map50" in rr[0]:
                mp = {}
                for r in rr:
                    mp[r["fold"]] = (r["map50"], r["map5095"])
                a = np.array(list(mp.values()))
                lines.append(f"    mAP50 per fold mean {a[:,0].mean():.3f} (sd {a[:,0].std():.3f}) | mAP50-95 mean {a[:,1].mean():.3f} (sd {a[:,1].std():.3f})")
            bad = [f"{r['file']}(pred={'cap' if r['pred']==0 else 'no_cap' if r['pred']==1 else 'none'},{r['conf']:.2f})" for r in rr if not r["correct"]]
            lines.append(f"    salah ({len(bad)}): " + ", ".join(bad[:30]))
    txt = "\n".join(lines)
    open(f"{CV}/cv_summary.txt", "w", encoding="utf-8").write(txt)
    print("\n===== RINGKASAN CV =====" + txt, flush=True)


if __name__ == "__main__":
    freeze_support()
    main()
