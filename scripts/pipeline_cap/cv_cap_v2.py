"""Model tutup v2: tambah foto sesi 16 & 20 Okt (termasuk tutup HITAM: Coca-Cola Zero, Powerade) ke data 13 Okt.
Data: Roboflow cap-nocap-belt-v1 versi 2 (397 foto, kotak dikoreksi user 21 Okt) -> datasets/2_dataset_cap/_rf_export_v2/.

CV 5 fold per MEREK (bukan per foto/botol): semua botol satu merek di fold yang sama, karena botol isi ulang 20 Okt bisa
botol fisik yang sama dengan set 13 Okt (mis. Aqua, Pristine, Vica). Coca-Cola Zero dan Powerade di fold berbeda.
Per fold, dua model dari cap_v4, resep IDENTIK dengan cv_belt.py (30 epoch tetap, imgsz 960, tanpa val/early-stop, last.pt):
  hanya13 : latih hanya foto 13 Okt (cara lama)        gabung : latih 13 Okt + 16/20 Okt
Uji (foto merek yang ditahan): (1) full-frame 640, kelas kotak terbaik; (2) ROI profil nano: kotak botol dari botol_ver8b,
kanvas 320 skala 1/3 (rvm_kp.roi_canvas), kelas kotak terbaik; tanpa deteksi = salah.
  python scripts/pipeline_cap/cv_cap_v2.py          (CV, resumable)
  python scripts/pipeline_cap/cv_cap_v2.py final    (model final dari 397 foto -> datasets/3_model_final/cap_belt_v2.pt)
Keluaran: datasets/4_eksperimen/cap/_cv_cap_v2/ (cv_records.jsonl, cv_summary.txt)."""
import glob, json, math, os, re, shutil, sys
from collections import defaultdict
from multiprocessing import freeze_support

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
DS = f"{REPO}/datasets"
SRC = f"{DS}/2_dataset_cap/_rf_export_v2/train"
CV = f"{DS}/4_eksperimen/cap/_cv_cap_v2"
BASE = f"{DS}/4_eksperimen/cap/cap_runs/cap_v4_640/weights/best.pt"
K = 5
HITAM = ("cokezero", "powerade")
ALIAS = [(r"cola-zero|cocacola-zero", "cokezero"), (r"powerade", "powerade"), (r"lemin|le-mineral", "lemin"), (r"aqua(?!viva)", "aqua"),
         (r"aquviva", "aquviva"), (r"crystalin", "crystalin"), (r"pristine", "pristine"), (r"superindo", "superindo"), (r"vica", "vica"),
         (r"pocari", "pocari"), (r"tehpucuk|teh-pucuk", "tehpucuk"), (r"fanta", "fanta"), (r"sprite", "sprite"), (r"frestea", "frestea"),
         (r"bigcola", "bigcola"), (r"cola-kecil", "cocacola"), (r"minutemaid", "minutemaid"), (r"sunbay", "sunbay"), (r"cleo", "cleo"),
         (r"isoplus", "isoplus"), (r"pocari", "pocari"), (r"vit\d", "vit"), (r"prima", "prima"), (r"nipis", "bignipis")]


def items():
    out = []
    for ip in sorted(glob.glob(f"{SRC}/images/*.jpg")):
        n = re.sub(r"_jpg\.rf\.[0-9a-f]+", "", os.path.basename(ip))[:-4]
        lp = f"{SRC}/labels/{os.path.basename(ip)[:-4]}.txt"
        if n.startswith("s10"):
            sesi, prod = n[1:5], n.split("_")[2]
        else:
            sesi, prod = "1013", n.split("_")[1]
        g = next((a for pat, a in ALIAS if re.search(pat, prod)), prod)
        cls = int(open(lp).read().split()[0])
        out.append(dict(name=n, img=ip, lbl=lp, sesi=sesi, merek=g, cls=cls))
    return out


def folds(its):
    cnt = defaultdict(int)
    for it in its:
        cnt[it["merek"]] += 1
    F = {"cokezero": 0, "powerade": 1}; load = [0] * K
    load[0] += cnt["cokezero"]; load[1] += cnt["powerade"]
    for g in sorted(cnt, key=lambda g: -cnt[g]):
        if g in F:
            continue
        k = min(range(K), key=lambda i: load[i]); F[g] = k; load[k] += cnt[g]
    return F, load


def link(s, d):
    try:
        os.link(s, d)
    except OSError:
        shutil.copy2(s, d)


def build(d, train_its):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(f"{d}/images/train"); os.makedirs(f"{d}/labels/train")
    for it in train_its:
        link(it["img"], f"{d}/images/train/{it['name']}.jpg"); shutil.copy2(it["lbl"], f"{d}/labels/train/{it['name']}.txt")
    open(f"{d}/data.yaml", "w").write(f"path: {d}\ntrain: images/train\nval: images/train\nnames:\n  0: cap\n  1: no_cap\n")


def train(d, name):
    from ultralytics import YOLO
    m = YOLO(BASE)
    m.train(data=f"{d}/data.yaml", imgsz=960, epochs=30, batch=8, patience=0, val=False, seed=0, device=0, workers=0, cache=False,
            close_mosaic=10, degrees=0.0, scale=0.3, translate=0.1, fliplr=0.5, flipud=0.0, project=f"{CV}/runs", name=name,
            plots=False, exist_ok=True, verbose=False)
    return f"{CV}/runs/{name}/weights/last.pt"


def bottle_boxes(its):
    """kotak botol (botol_ver8b) per foto, di-cache; dipakai mode ROI."""
    p = f"{CV}/bottle_boxes.json"
    if os.path.exists(p):
        return json.load(open(p))
    import cv2
    from ultralytics import YOLO
    det = YOLO(f"{DS}/3_model_final/botol_ver8b.pt"); out = {}
    for it in its:
        r = det.predict(cv2.imread(it["img"]), imgsz=640, conf=0.35, verbose=False)[0]
        bb = [b.xyxy[0].tolist() for b in r.boxes if int(b.cls) == 0]
        out[it["name"]] = max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])) if bb else None
    json.dump(out, open(p, "w")); return out


def evaluate(pt, test_its, boxes, tag, k):
    import cv2
    from ultralytics import YOLO
    from rvm_kp.analyzer import roi_canvas
    m = YOLO(pt); recs = []
    for it in test_its:
        im = cv2.imread(it["img"]); W = im.shape[1]
        r = m.predict(im, imgsz=640, conf=0.03, max_det=10, verbose=False)[0]
        full = int(max(r.boxes, key=lambda z: float(z.conf)).cls) if len(r.boxes) else -1
        roi = -1; bx = boxes.get(it["name"])
        if bx:
            cv, _ = roi_canvas(im, *bx, 320, (1 / 3) * 1920 / W, 0.08)
            rr = m.predict(cv, imgsz=320, conf=0.25, verbose=False)[0]
            roi = int(max(rr.boxes, key=lambda z: float(z.conf)).cls) if len(rr.boxes) else -1
        recs.append(dict(fold=k, model=tag, name=it["name"], sesi=it["sesi"], merek=it["merek"], cls=it["cls"],
                         full=full, roi=roi, hitam=it["merek"] in HITAM))
    return recs


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = k / n; den = 1 + z * z / n; c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def summarize(out):
    recs = [json.loads(l) for l in open(out) if l.strip()]
    L = ["CV 5 fold per merek, model tutup (cap/no_cap). Benar = kelas kotak terbaik == label; tanpa deteksi = salah.", ""]
    subsets = [("SEMUA", lambda r: True), ("sesi 13 Okt", lambda r: r["sesi"] == "1013"),
               ("sesi 16/20 Okt", lambda r: r["sesi"] != "1013"), ("TUTUP HITAM (Coke Zero, Powerade)", lambda r: r["hitam"]),
               ("16/20 Okt tanpa tutup hitam", lambda r: r["sesi"] != "1013" and not r["hitam"]),
               ("label no_cap", lambda r: r["cls"] == 1), ("label cap", lambda r: r["cls"] == 0)]
    for mode in ("full", "roi"):
        L.append(f"== mode {'full-frame 640' if mode == 'full' else 'ROI area botol 320 (profil nano)'}")
        for sname, f in subsets:
            row = f"  {sname:36s}"
            for tag in ("hanya13", "gabung"):
                rr = [r for r in recs if r["model"] == tag and f(r)]
                ok = sum(r[mode] == r["cls"] for r in rr); lo, hi = wilson(ok, len(rr))
                row += f" | {tag}: {ok:3d}/{len(rr):<3d} {100 * ok / max(1, len(rr)):5.1f}% (CI {100 * lo:4.1f}-{100 * hi:4.1f})"
            L.append(row)
        L.append("")
    L.append("Per merek (mode ROI): hanya13 -> gabung")
    for g in sorted({r["merek"] for r in recs}):
        a = [r for r in recs if r["merek"] == g and r["model"] == "hanya13"]; b = [r for r in recs if r["merek"] == g and r["model"] == "gabung"]
        L.append(f"  {g:12s} n={len(a):3d}  {sum(r['roi'] == r['cls'] for r in a):3d} -> {sum(r['roi'] == r['cls'] for r in b):3d}")
    open(f"{CV}/cv_summary.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L), flush=True)


def main():
    os.makedirs(CV, exist_ok=True)
    its = items(); F, load = folds(its)
    print(f"{len(its)} foto, {len({i['merek'] for i in its})} merek; beban fold {load}", flush=True)
    for k in range(K):
        print(f"  fold {k}: {sorted(g for g, v in F.items() if v == k)}", flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        d = f"{CV}/data_final"; build(d, its); pt = train(d, "final")
        shutil.copy2(pt, f"{DS}/3_model_final/cap_belt_v2.pt"); shutil.rmtree(d, ignore_errors=True)
        print("FINAL ->", f"{DS}/3_model_final/cap_belt_v2.pt"); return
    boxes = bottle_boxes(its)
    out = f"{CV}/cv_records.jsonl"
    done = {(json.loads(l)["fold"], json.loads(l)["model"]) for l in open(out) if l.strip()} if os.path.exists(out) else set()
    for k in range(K):
        test = [i for i in its if F[i["merek"]] == k]
        for tag, tr in (("hanya13", [i for i in its if F[i["merek"]] != k and i["sesi"] == "1013"]),
                        ("gabung", [i for i in its if F[i["merek"]] != k])):
            if (k, tag) in done:
                continue
            d = f"{CV}/data_f{k}_{tag}"; build(d, tr)
            print(f"fold {k} {tag}: latih {len(tr)} foto, uji {len(test)}", flush=True)
            recs = evaluate(train(d, f"f{k}_{tag}"), test, boxes, tag, k)
            with open(out, "a") as fh:
                for r in recs:
                    fh.write(json.dumps(r) + "\n")
            print(f"  benar full {sum(r['full'] == r['cls'] for r in recs)}/{len(recs)}, roi {sum(r['roi'] == r['cls'] for r in recs)}/{len(recs)}", flush=True)
            shutil.rmtree(d, ignore_errors=True)
    summarize(out)


if __name__ == "__main__":
    freeze_support()
    main()
