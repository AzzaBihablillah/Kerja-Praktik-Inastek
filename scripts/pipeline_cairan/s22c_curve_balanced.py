"""Langkah 22c: kurva belajar CNN DIULANG dengan benar (memperbaiki cacat s22):
  - kelas SEIMBANG (positif digandakan ceil(neg/pos), tanpa batas 6x)  -> ketidakseimbangan tidak lagi berubah menurut k
  - 30 epoch (sama dgn resep CV utama)
  - probabilitas disimpan -> AUC (tanpa ambang) selain recall/spesifisitas @0.5
Rencana subset SAMA dengan s22 (seed 0): tiap botol uji b, k=1..4 botol berwarna latih (2/2/2/1 subset) = 35 latihan.
Latih: botol latih W16 (berwarna+kosong, tanpa 1/4) + C13 di luar fold botol b. Uji: botol b, C13 fold b, P15.
Keluaran: out/curve_cnn_bal.jsonl (dapat dilanjutkan)."""
import csv, itertools, json, math, os, random, shutil
from ultralytics import YOLO

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); RES = f"{OUT}/curve_cnn_bal.jsonl"
EPOCHS = 30


def main():
    rows = list(csv.DictReader(open(f"{OUT}/crops.csv")))
    c13g = sorted({r["group"] for r in rows if r["src"] == "C13"}, key=lambda g: int(g.split("_b")[1]))
    c13f = {g: (i % 5) + 1 for i, g in enumerate(c13g)}
    cp = lambda r: f"{OUT}/{r['crop'].replace('crops/', 'crops_fold/')}"
    W = {b: [r for r in rows if r["src"] == "W16" and int(r["group"][-2:]) == b and r["liquid"] in ("berwarna", "kosong")] for b in range(1, 6)}
    done = set()
    if os.path.exists(RES):
        done = {(j["test"], tuple(j["train"])) for j in map(json.loads, open(RES))}
    rng = random.Random(0); plan = []
    for b in range(1, 6):
        others = [x for x in range(1, 6) if x != b]
        for k, n in ((1, 2), (2, 2), (3, 2), (4, 1)):
            subs = list(itertools.combinations(others, k)); rng.shuffle(subs)
            plan += [(b, s) for s in subs[:n]]
    for b, tr_b in plan:
        if (b, tr_b) in done:
            continue
        tag = f"t{b}_{''.join(map(str, tr_b))}"
        d = f"{OUT}/tmp_curve_bal/{tag}"; shutil.rmtree(d, ignore_errors=True)
        tr = [(r, r["liquid"]) for x in tr_b for r in W[x]] + [(r, "kosong") for r in rows if r["src"] == "C13" and c13f[r["group"]] != b]
        n = {c: sum(l == c for _, l in tr) for c in ("berwarna", "kosong")}; big = max(n.values())
        reps = {c: math.ceil(big / n[c]) for c in n}
        for c in ("berwarna", "kosong"):
            os.makedirs(f"{d}/train/{c}"); os.makedirs(f"{d}/val/{c}")
            for r, l in tr:
                if l == c:
                    for i in range(reps[c]):
                        os.link(cp(r), f"{d}/train/{c}/{r['id']}_{i}.jpg")
        for r in W[b][:4]:
            os.link(cp(r), f"{d}/val/{r['liquid']}/{r['id']}.jpg")
        for c in ("berwarna", "kosong"):
            if not os.listdir(f"{d}/val/{c}"):
                os.link(cp(tr[0][0]), f"{d}/val/{c}/dummy.jpg")
        YOLO(os.path.join(ROOT, "3_model_final/dasar/yolov8n-cls.pt")).train(
            data=d, epochs=EPOCHS, imgsz=224, batch=32, seed=0, deterministic=True, workers=0, verbose=False, plots=False,
            project=f"{OUT}/runs_curve_bal", name=tag, exist_ok=True, patience=0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,
            fliplr=0.5, flipud=0.5, degrees=0, auto_augment="", erasing=0.1)
        pm = YOLO(f"{OUT}/runs_curve_bal/{tag}/weights/last.pt"); cb = [i for i, v in pm.names.items() if v == "berwarna"][0]
        pr = lambda rs: [float(p.probs.data[cb]) for p in pm.predict([cp(r) for r in rs], imgsz=224, verbose=False)] if rs else []
        c13t = [r for r in rows if r["src"] == "C13" and c13f[r["group"]] == b]
        p15 = [r for r in rows if r["src"] == "P15"]
        wp = pr(W[b]); cp13 = pr(c13t); pp = pr(p15)
        rec = dict(test=b, train=list(tr_b), k=len(tr_b), latih_pos=n["berwarna"] * reps["berwarna"], latih_neg=n["kosong"] * reps["kosong"],
                   w16=[(r["liquid"], round(p, 4)) for r, p in zip(W[b], wp)], c13=[round(p, 4) for p in cp13],
                   p15=[(r["liquid"], round(p, 4)) for r, p in zip(p15, pp)])
        open(RES, "a").write(json.dumps(rec) + "\n")
        shutil.rmtree(d, ignore_errors=True)
        print("selesai", tag, flush=True)
    print("CURVE BAL SELESAI")


if __name__ == "__main__":
    main()
