"""Langkah 28: validasi silang yang JUJUR -- dikelompokkan per CAIRAN (botol isi ulang dengan cairan yang sama diuji bersamaan).
Botol berwarna: W16 b01..b05 (16 Okt) + V20 b01..b16 (20 Okt) = 21 botol, 13 kelompok cairan (keterangan user 20 Okt):
  powerade      : V20 b01 (asli), b07 (botol Sprite), b14 (botol Vica)          [b14 biru: diasumsikan Powerade]
  fanta_anggur  : V20 b02 (asli), b08 (botol Teh Pucuk)
  soy           : V20 b03 (soybean), b09 (soybean + Nutrisari sirsak)
  minutemaid    : V20 b04
  cola_vanila   : V20 b05 (asli), b06 (botol Aqua)                              [b06 diasumsikan Coke Zero vanila]
  nutri_jeruk   : V20 b10, b12, b15 (Nutrisari jeruk peras)
  nutri_hijau   : V20 b11, b13 (Nutrisari anggur hijau)
  sunlight      : V20 b16 (air sabun Sunlight)
  W16 b01..b05  : masing-masing satu kelompok (Sprite NM, Coke Zero, Frestea, Fanta-label kuning, Coca-Cola kecil)
Resep RA = W16+V20 (dua sesi, pos+neg), RB = RA + C13 kosong (5 fold). Seimbang (ceil), 30 epoch, crop lipat.
  python s28_cv_liquidgroup.py --recipe RA|RB   -> out/cv_lg_<recipe>.jsonl"""
import argparse, csv, json, math, os, shutil
from ultralytics import YOLO

ap = argparse.ArgumentParser(); ap.add_argument("--recipe", required=True, choices=["RA", "RB", "RC", "RH"]); a = ap.parse_args()
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); RES = f"{OUT}/cv_lg_{a.recipe}.jsonl"
LIQ = {"V20_b01": "powerade", "V20_b07": "powerade", "V20_b14": "powerade", "V20_b02": "fanta_anggur", "V20_b08": "fanta_anggur",
       "V20_b03": "soy", "V20_b09": "soy", "V20_b04": "minutemaid", "V20_b05": "cola_vanila", "V20_b06": "cola_vanila",
       "V20_b10": "nutri_jeruk", "V20_b12": "nutri_jeruk", "V20_b15": "nutri_jeruk", "V20_b11": "nutri_hijau", "V20_b13": "nutri_hijau",
       "V20_b16": "sunlight"}


# RC = RA + HANYA botol kosong 13 Okt berlabel penuh/berwarna (dipilih visual dari lembar C13), bukan semua 30
FULLLABEL = {"C13_b06", "C13_b11", "C13_b12", "C13_b13", "C13_b15", "C13_b17", "C13_b21", "C13_b23", "C13_b25", "C13_b26", "C13_b27", "C13_b29"}


def liq(g):
    return LIQ.get(g, g)                                                  # W16 botol = kelompoknya sendiri


def main():
    rows = list(csv.DictReader(open(f"{OUT}/crops.csv"))) + list(csv.DictReader(open(f"{OUT}/v20_crops.csv")))
    cp = lambda r: f"{OUT}/{r['crop'].replace('crops/', 'crops_fold/')}"
    lgroups = sorted({liq(r["group"]) for r in rows if r["src"] in ("W16", "V20")})
    c13g = sorted({r["group"] for r in rows if r["src"] == "C13"}, key=lambda g: int(g.split("_b")[1]))
    c13f = {g: (i % 5) for i, g in enumerate(c13g)}
    done = {json.loads(l)["test"] for l in open(RES)} if os.path.exists(RES) else set()
    for i, L in enumerate(lgroups):
        if L in done:
            continue
        cf = i % 5
        tr = [r for r in rows if r["src"] in ("W16", "V20") and liq(r["group"]) != L and r["liquid"] in ("berwarna", "kosong")]
        if a.recipe == "RB":
            tr += [r for r in rows if r["src"] == "C13" and c13f[r["group"]] != cf]
        if a.recipe == "RH":                                              # RA + salinan geser-hue (piksel botol saja, s29)
            tr += [dict(r, id=f"{r['id']}_h{k}", crop=f"crops_fold_hue/{r['id']}_h{k}.jpg") for r in list(tr) for k in range(3)
                   if os.path.exists(f"{OUT}/crops_fold_hue/{r['id']}_h{k}.jpg")]
        if a.recipe == "RC":
            tr += [r for r in rows if r["src"] == "C13" and r["group"] in FULLLABEL and c13f[r["group"]] != cf]
        d = f"{OUT}/tmp_lg/{a.recipe}_{L}"; shutil.rmtree(d, ignore_errors=True)
        n = {c: sum(r["liquid"] == c for r in tr) for c in ("berwarna", "kosong")}; big = max(n.values())
        for c in ("berwarna", "kosong"):
            os.makedirs(f"{d}/train/{c}"); os.makedirs(f"{d}/val/{c}")
            for r in tr:
                if r["liquid"] == c:
                    for k in range(math.ceil(big / n[c])):
                        os.link(cp(r), f"{d}/train/{c}/{r['id']}_{k}.jpg")
        te = [r for r in rows if r["src"] in ("W16", "V20") and liq(r["group"]) == L]
        for c in ("berwarna", "kosong"):
            src = [r for r in te if r["liquid"] == c] or [r for r in tr if r["liquid"] == c][:1]
            for r in src[:4]:
                os.link(cp(r), f"{d}/val/{c}/{r['id']}.jpg")
        YOLO(os.path.join(ROOT, "3_model_final/dasar/yolov8n-cls.pt")).train(
            data=d, epochs=30, imgsz=224, batch=32, seed=0, deterministic=True, workers=0, verbose=False, plots=False,
            project=f"{OUT}/runs_lg", name=f"{a.recipe}_{L}", exist_ok=True, patience=0, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,
            fliplr=0.5, flipud=0.5, degrees=0, auto_augment="", erasing=0.1)
        m = YOLO(f"{OUT}/runs_lg/{a.recipe}_{L}/weights/last.pt"); cb = [k for k, v in m.names.items() if v == "berwarna"][0]
        pr = lambda rs: [round(float(p.probs.data[cb]), 4) for p in m.predict([cp(r) for r in rs], imgsz=224, verbose=False)] if rs else []
        c13 = [r for r in rows if r["src"] == "C13" and (a.recipe == "RA" or c13f[r["group"]] == cf)]   # RB/RC: hanya fold yang tidak dilatih
        p15 = [r for r in rows if r["src"] == "P15"]
        rec = dict(test=L, test_rows=[(r["id"], r["liquid"], r["level"], r["group"]) for r in te], test_p=pr(te),
                   c13=[(r["group"], p) for r, p in zip(c13, pr(c13))], p15=[(r["liquid"], p) for r, p in zip(p15, pr(p15))])
        open(RES, "a").write(json.dumps(rec) + "\n")
        shutil.rmtree(d, ignore_errors=True)
        print("selesai", a.recipe, L, flush=True)
    print("CV LG SELESAI", a.recipe)


if __name__ == "__main__":
    main()
