"""Model merek langkah 6: apakah data tambahan TERAUDIT membantu? Fold & ujian SAMA dengan m02 (hanya foto belt kita);
data tambahan hanya masuk ke bagian latih tiap fold.
  varian D = belt + warisan brand2 + Open Food Facts ; B = belt + warisan ; C = belt + OFF
Label tambahan = audit user 23 Okt (audit_final.csv: label 'buang' dibuang). Kelas internal mitra/bukan = coca_cola/non_coca_cola.
  python scripts/pipeline_merek/m06_cv_campur.py [D B C] -> datasets/4_eksperimen/merek/cv_campur.jsonl + cv_campur_report.txt"""
import csv, json, os, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m02_cv_dasar import OUT, CR, S, rows, make_crops, folds, train

VAR = {"E": ("warisan", "off"), "D": ("warisan", "off"), "B": ("warisan",), "C": ("off",)}
BOBOT = {"E": 6}            # E = D dengan foto belt diulang 6x (belt ~ setara jumlah data tambahan)
EP = 12                      # data ~6x lebih banyak per epoch -> 12 epoch = langkah latih ~2,4x model dasar
MAP = {"coca_cola": "mitra", "non_coca_cola": "bukan"}


def extra(src):
    return [(r["path"], MAP[r["label"]], r["id"]) for r in csv.DictReader(open(f"{OUT}/audit_final.csv", encoding="utf-8"))
            if r["sumber"] in src and r["label"] in MAP]


def main(vs):
    from ultralytics import YOLO
    R = make_crops(rows()); res = f"{OUT}/cv_campur.jsonl"
    done = {(j["var"], j["skema"], j["fold"]) for j in map(json.loads, open(res))} if os.path.exists(res) else set()
    for v, skema in vs:
        X = extra(VAR[v]); key = "grup_" + skema
        if True:
            F = folds(R, key)
            for k in range(5):
                if (v, skema, k) in done:
                    continue
                tr = [r for r in R if F[r[key]] != k]; te = [r for r in R if F[r[key]] == k]
                items = [(f"{CR}/{r['id']}.jpg", r["kelas"], f"{r['id']}r{j}") for r in tr for j in range(BOBOT.get(v, 1))] + X
                pt = train(items, f"{OUT}/tmp/{v}{skema}{k}", f"{OUT}/runs", f"campur_{v}{skema}{k}", epochs=EP)
                m = YOLO(pt); cm = [i for i, n in m.names.items() if n == "mitra"][0]
                ps = [float(p.probs.data[cm]) for p in m.predict([f"{CR}/{r['id']}.jpg" for r in te], imgsz=S, verbose=False)]
                with open(res, "a") as fh:
                    for r, p in zip(te, ps):
                        fh.write(json.dumps(dict(var=v, skema=skema, fold=k, id=r["id"], produk=r["produk"], kelas=r["kelas"], p=round(p, 4))) + "\n")
                print(f"{v} {skema} fold {k}: latih {len(items)} (tambahan {len(X)}), uji {len(te)}", flush=True)
        report()


def report():
    A = [json.loads(l) for l in open(f"{OUT}/cv_dasar.jsonl")]
    for a in A:
        a["var"] = "A"
    A += [json.loads(l) for l in open(f"{OUT}/cv_campur.jsonl")]
    nama = {"A": "dasar (belt saja)", "B": "+ data lama", "C": "+ Open Food Facts", "D": "+ data lama + OFF", "E": "D, belt x6"}
    L = ["Model merek coca_cola vs non_coca_cola. Uji HANYA foto belt kita, fold sama. Ambang 0,5.", ""]
    for skema, judul in (("F", "F: produk dikenal, botol/sesi baru"), ("P", "P: produk TIDAK pernah dilihat di belt")):
        L.append(f"== {judul}")
        prods = sorted({(a["kelas"], a["produk"]) for a in A if a["skema"] == skema and a["kelas"] == "mitra"})
        for v in "ABCDE":
            rr = [a for a in A if a["var"] == v and a["skema"] == skema]
            if not rr:
                continue
            mt = [a for a in rr if a["kelas"] == "mitra"]; bk = [a for a in rr if a["kelas"] == "bukan"]
            per = defaultdict(list)
            for a in mt:
                per[a["produk"]].append(a["p"] >= .5)
            L.append(f"  {v} {nama[v]:22s} coca {sum(a['p'] >= .5 for a in mt):3d}/{len(mt)} | non benar {sum(a['p'] < .5 for a in bk):3d}/{len(bk)} | "
                     + " ".join(f"{p[:10]} {sum(per[p])}/{len(per[p])}" for _, p in prods))
        L.append("")
    open(f"{OUT}/cv_campur_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main([(a[0], a[1]) for a in (sys.argv[1:] or ["DF", "DP", "BF", "CF", "BP", "CP"])])
