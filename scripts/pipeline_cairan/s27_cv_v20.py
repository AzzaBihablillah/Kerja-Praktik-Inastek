"""Langkah 27: latih ulang + validasi silang per botol dengan sesi 20 Okt (botol baru).
Grup uji = 13 botol berwarna: W16 b01..b05 (16 Okt) + V20 b01..b08 (20 Okt). Satu model per grup (leave-one-bottle-out).
Resep:
  RA = W16 + V20 (positif DAN negatif dari dua sesi, tanpa C13)  -> lebih sedikit jalan pintas sesi
  RB = RA + C13 kosong (negatif merek lain, 13 Okt), C13 dibagi 5 fold mengikuti urutan grup
Kelas seimbang (ceil), 30 epoch, crop lipat. Level seperempat W16 tidak dilatih (tetap diprediksi sbg info).
Prediksi disimpan: botol uji, C13 (RA: semua; RB: fold yang tidak dilatih), P15 (sesi pilot).
  python s27_cv_v20.py --recipe RA|RB      -> out/cv_v20_<recipe>.jsonl (dapat dilanjutkan)"""
import argparse, csv, json, math, os, shutil
from ultralytics import YOLO

ap = argparse.ArgumentParser(); ap.add_argument("--recipe", required=True, choices=["RA", "RB"]); a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); RES = f"{OUT}/cv_v20_{a.recipe}.jsonl"


def main():
    rows = list(csv.DictReader(open(f"{OUT}/crops.csv"))) + list(csv.DictReader(open(f"{OUT}/v20_crops.csv")))
    cp = lambda r: f"{OUT}/{r['crop'].replace('crops/', 'crops_fold/')}"
    groups = sorted({r["group"] for r in rows if r["src"] in ("W16", "V20")})
    c13g = sorted({r["group"] for r in rows if r["src"] == "C13"}, key=lambda g: int(g.split("_b")[1]))
    c13f = {g: (i % 5) for i, g in enumerate(c13g)}
    done = {json.loads(l)["test"] for l in open(RES)} if os.path.exists(RES) else set()
    for i, g in enumerate(groups):
        if g in done:
            continue
        cf = i % 5
        tr = [r for r in rows if r["src"] in ("W16", "V20") and r["group"] != g and r["liquid"] in ("berwarna", "kosong")]
        if a.recipe == "RB":
            tr += [r for r in rows if r["src"] == "C13" and c13f[r["group"]] != cf]
        d = f"{OUT}/tmp_v20/{a.recipe}_{g}"; shutil.rmtree(d, ignore_errors=True)
        n = {c: sum(r["liquid"] == c for r in tr) for c in ("berwarna", "kosong")}; big = max(n.values())
        for c in ("berwarna", "kosong"):
            os.makedirs(f"{d}/train/{c}"); os.makedirs(f"{d}/val/{c}")
            for r in tr:
                if r["liquid"] == c:
                    for k in range(math.ceil(big / n[c])):
                        os.link(cp(r), f"{d}/train/{c}/{r['id']}_{k}.jpg")
        te = [r for r in rows if r["group"] == g]
        for c in ("berwarna", "kosong"):                                  # val hanya utk format
            src = [r for r in te if r["liquid"] == c] or [r for r in tr if r["liquid"] == c][:1]
            for r in src[:4]:
                os.link(cp(r), f"{d}/val/{c}/{r['id']}.jpg")
        YOLO(os.path.join(ROOT, "3_model_final/dasar/yolov8n-cls.pt")).train(
            data=d, epochs=30, imgsz=224, batch=32, seed=0, deterministic=True, workers=0, verbose=False, plots=False,
            project=f"{OUT}/runs_v20", name=f"{a.recipe}_{g}", exist_ok=True, patience=0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,
            fliplr=0.5, flipud=0.5, degrees=0, auto_augment="", erasing=0.1)
        m = YOLO(f"{OUT}/runs_v20/{a.recipe}_{g}/weights/last.pt"); cb = [k for k, v in m.names.items() if v == "berwarna"][0]
        pr = lambda rs: [round(float(p.probs.data[cb]), 4) for p in m.predict([cp(r) for r in rs], imgsz=224, verbose=False)] if rs else []
        c13 = [r for r in rows if r["src"] == "C13" and (a.recipe == "RA" or c13f[r["group"]] == cf)]
        p15 = [r for r in rows if r["src"] == "P15"]
        rec = dict(test=g, test_rows=[(r["id"], r["liquid"], r["level"]) for r in te], test_p=pr(te),
                   c13=[(r["group"], p) for r, p in zip(c13, pr(c13))], p15=[(r["liquid"], p) for r, p in zip(p15, pr(p15))])
        open(RES, "a").write(json.dumps(rec) + "\n")
        shutil.rmtree(d, ignore_errors=True)
        print("selesai", a.recipe, g, flush=True)
    print("CV V20 SELESAI", a.recipe)


if __name__ == "__main__":
    main()
