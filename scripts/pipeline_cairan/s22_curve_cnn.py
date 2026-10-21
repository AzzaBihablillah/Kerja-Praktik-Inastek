"""Langkah 22: kurva belajar classifier CNN: berapa botol BERWARNA nyata yang diperlukan?
Untuk tiap botol uji b (W16, 5 botol) dan k = 1..4 botol latih dari 4 botol sisanya (subset acak, seed tetap):
  latih (resep V2, crop lipat, 15 epoch): botol latih W16 (berwarna+kosong, tanpa 1/4) + C13 kosong KECUALI fold-nya botol b
  uji   : botol b (berwarna recall, kosong spesifisitas), C13 fold botol b (spesifisitas), P15 (recall berwarna; spesifisitas kosong)
Subset per k: k=1 -> 2, k=2 -> 2, k=3 -> 2, k=4 -> 1  (35 latihan total). Keluaran: out/curve_cnn.jsonl (dapat dilanjutkan).
"""
import csv, itertools, json, os, random, shutil
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); RES = f"{OUT}/curve_cnn.jsonl"
EPOCHS = 15


def main():
    rows = list(csv.DictReader(open(f"{OUT}/crops.csv")))
    c13g = sorted({r["group"] for r in rows if r["src"] == "C13"}, key=lambda g: int(g.split("_b")[1]))
    c13f = {g: (i % 5) + 1 for i, g in enumerate(c13g)}
    cp = lambda r: f"{OUT}/{r['crop'].replace('crops/', 'crops_fold/')}"
    W = {b: [r for r in rows if r["src"] == "W16" and int(r["group"][-2:]) == b and r["liquid"] in ("berwarna", "kosong")] for b in range(1, 6)}
    done = set()
    if os.path.exists(RES):
        done = {(j["test"], tuple(j["train"])) for j in map(json.loads, open(RES))}
    rng = random.Random(0)
    plan = []
    for b in range(1, 6):
        others = [x for x in range(1, 6) if x != b]
        for k, n in ((1, 2), (2, 2), (3, 2), (4, 1)):
            subs = list(itertools.combinations(others, k)); rng.shuffle(subs)
            plan += [(b, s) for s in subs[:n]]
    for b, tr_b in plan:
        if (b, tr_b) in done:
            continue
        d = f"{OUT}/tmp_curve/t{b}_{''.join(map(str, tr_b))}"; shutil.rmtree(d, ignore_errors=True)
        tr = [(r, r["liquid"]) for x in tr_b for r in W[x]] + [(r, "kosong") for r in rows if r["src"] == "C13" and c13f[r["group"]] != b]
        n = {c: sum(l == c for _, l in tr) for c in ("berwarna", "kosong")}; big = max(n.values())
        for c in ("berwarna", "kosong"):
            os.makedirs(f"{d}/train/{c}"); os.makedirs(f"{d}/val/{c}")
            rep = min(6, max(1, round(big / n[c])))
            for r, l in tr:
                if l == c:
                    for i in range(rep):
                        os.link(cp(r), f"{d}/train/{c}/{r['id']}_{i}.jpg")
        for r in W[b][:4]:
            os.link(cp(r), f"{d}/val/{r['liquid']}/{r['id']}.jpg")
        for c in ("berwarna", "kosong"):
            if not os.listdir(f"{d}/val/{c}"):
                os.link(cp(tr[0][0]), f"{d}/val/{c}/dummy.jpg")
        m = YOLO(os.path.join(ROOT, "3_model_final/dasar/yolov8n-cls.pt"))
        m.train(data=d, epochs=EPOCHS, imgsz=224, batch=32, seed=0, deterministic=True, workers=0, verbose=False, plots=False,
                project=f"{OUT}/runs_curve", name=f"t{b}_{''.join(map(str, tr_b))}", exist_ok=True, patience=0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,
                fliplr=0.5, flipud=0.5, degrees=0, auto_augment="", erasing=0.1)
        pm = YOLO(f"{OUT}/runs_curve/t{b}_{''.join(map(str, tr_b))}/weights/last.pt")
        cb = [i for i, v in pm.names.items() if v == "berwarna"][0]

        def pr(rs):
            if not rs:
                return []
            res = pm.predict([cp(r) for r in rs], imgsz=224, verbose=False)
            return [float(p.probs.data[cb]) for p in res]
        wp = [(r, p) for r, p in zip(W[b], pr(W[b]))]
        pos = [p for r, p in wp if r["liquid"] == "berwarna"]; neg = [p for r, p in wp if r["liquid"] == "kosong"]
        c13t = [r for r in rows if r["src"] == "C13" and c13f[r["group"]] == b]; cps = pr(c13t)
        p15 = [r for r in rows if r["src"] == "P15"]; p15p = pr(p15)
        pw = [p for r, p in zip(p15, p15p) if r["liquid"] == "berwarna"]; pn = [p for r, p in zip(p15, p15p) if r["liquid"] != "berwarna"]
        rec = dict(test=b, train=list(tr_b), k=len(tr_b), w16_recall=[sum(p >= .5 for p in pos), len(pos)], w16_spec=[sum(p < .5 for p in neg), len(neg)],
                   c13_spec=[sum(p < .5 for p in cps), len(cps)], p15_recall=[sum(p >= .5 for p in pw), len(pw)], p15_spec=[sum(p < .5 for p in pn), len(pn)])
        open(RES, "a").write(json.dumps(rec) + "\n")
        shutil.rmtree(d, ignore_errors=True)
        print("selesai", b, tr_b, rec["w16_recall"], flush=True)
    print("CURVE SELESAI")


if __name__ == "__main__":
    main()
