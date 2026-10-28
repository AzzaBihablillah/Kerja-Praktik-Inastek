"""ver11 = resep ver10, tetapi foto non-botol NYATA 22 Okt diberi KOTAK kelas non_bottle (bukan gambar kosong/latar).
Alasan (26 Okt): app menolak benda asing lewat kelas non_bottle; ver10 dilatih dengan non-botol sebagai latar sehingga TIDAK
pernah mengeluarkan non_bottle (0/86) -> deteksi benda asing hilang bila ver10 diaktifkan.
Kotak non_bottle = usulan ver9c (kotak paling yakin) DIAUDIT user 27 Okt (https://claude.ai/artifact/HWEc2Tc8bkcMBLuvBmVgKs):
81 pas, 5 'kotak salah' (tidak dipakai). Botol semprot & Listerine IKUT sebagai non-botol (keputusan user 27 Okt: botol
non-minuman ditolak). Botol positif & kelompok = sama dengan ver10 (nyata22/items.json).
Anti-bias (permintaan user 27 Okt): plastik remas = 35/81 foto non-botol -> bobot per JENIS benda disamakan (rep_seimbang);
dinilai juga di data LAIN yang tak dilatih: TrashNet uji (kaca/kaleng/kertas/kardus/sampah, ditempel ke belt), TACO uji,
cek jalan pintas (botol tempelan), old_test (botol) -> scripts/pipeline_botol/ver11_uji_lain.py.
CV 5 fold per BENDA. Per item dicatat: terima (ada kotak bottle conf>=0,35) dan nb (ada kotak non_bottle conf>=0,35).
Pembanding: ver9c apa adanya (tidak pernah melihat foto ini) + hasil CV ver10 per item (ver10/cv.jsonl, kecuali semprot/Listerine).
  python scripts/pipeline_botol/ver11_nonbotol.py          CV -> datasets/4_eksperimen/botol/ver11/cv.jsonl + cv_report.txt
  python scripts/pipeline_botol/ver11_nonbotol.py final    latih semua -> datasets/3_model_final/botol_ver11_kandidat.pt"""
import csv, json, os, shutil, sys
from collections import defaultdict

DS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
V10 = f"{DS}/4_eksperimen/botol/ver10"; NY10 = f"{V10}/nyata22"
OUT = f"{DS}/4_eksperimen/botol/ver11"; NY = f"{OUT}/nyata22"
NB = f"{DS}/1_foto_mentah/2026-10-22_nonbottle_v1"
AUD = f"{DS}/4_eksperimen/nonbotol_audit"
V9C = f"{DS}/3_model_final/botol_ver9c.pt"; V7 = os.path.join(DS, "..", "export_model", "bottle_best_ver7.pt")
K, REP_NEG, REP_POS, REP_MAX = 5, 5, 2, 12
SALAH = {"N005", "N018", "N020", "N076", "N077"}          # audit user 27 Okt: kotak tidak pas -> tidak dipakai
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ver10_nyata import group_of_neg, folds                 # kelompok per benda & pembagian fold sama dengan ver10


def build():
    if os.path.exists(f"{NY}/items.json"):
        return json.load(open(f"{NY}/items.json"))
    import cv2
    os.makedirs(f"{NY}/images", exist_ok=True); os.makedirs(f"{NY}/labels", exist_ok=True)
    items = []
    for it in json.load(open(f"{NY10}/items.json")):              # botol positif: sama dengan ver10
        if it["jenis"] != "botol":
            continue
        if not os.path.exists(f"{NY}/images/{it['name']}"):
            os.link(f"{NY10}/images/{it['name']}", f"{NY}/images/{it['name']}")
        shutil.copy2(f"{NY10}/labels/{it['name'][:-4]}.txt", f"{NY}/labels/{it['name'][:-4]}.txt")
        items.append(it)
    for r in csv.DictReader(open(f"{AUD}/usulan.csv", encoding="utf-8")):
        if r["id"] in SALAH or r["cls_usul"] == "-":
            continue
        n = r["file"]; name = "nb_" + n
        H, W = cv2.imread(f"{NB}/{n}").shape[:2]
        x1, y1, x2, y2 = (float(r[k]) for k in ("x1", "y1", "x2", "y2"))
        if not os.path.exists(f"{NY}/images/{name}"):
            os.link(f"{NB}/{n}", f"{NY}/images/{name}")
        open(f"{NY}/labels/{name[:-4]}.txt", "w").write(f"1 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        items.append(dict(name=name, kel=group_of_neg(n), jenis="non_botol", kat=n.rsplit("_", 1)[0].rstrip("-0123456789")))
    json.dump(items, open(f"{NY}/items.json", "w"), indent=0)
    return items


def rep_seimbang(extra):
    """Ulangan per foto non-botol agar tiap JENIS benda berbobot total sama (plastik remas 35 foto tidak mendominasi
    -> model tidak belajar 'non-botol = plastik kresek'). Total bobot non-botol = REP_NEG x jumlah foto (sama dengan ver10);
    dibatasi 1..REP_MAX per foto agar jenis dengan sedikit foto tidak dihafal."""
    nb = [it for it in extra if it["jenis"] == "non_botol"]
    cnt = defaultdict(int)
    for it in nb:
        cnt[it["kat"]] += 1
    w_kat = REP_NEG * len(nb) / max(1, len(cnt))
    return {it["name"]: max(1, min(REP_MAX, round(w_kat / cnt[it["kat"]]))) for it in nb}


def train(extra, name):
    from ultralytics import YOLO
    base = [l.strip() for l in open(f"{DS}/4_eksperimen/botol/ver9c_train.txt") if l.strip()]
    rep = rep_seimbang(extra)
    tr = base + [f"{NY}/images/{it['name']}" for it in extra for _ in range(rep[it["name"]] if it["jenis"] == "non_botol" else REP_POS)]
    kat = defaultdict(int)
    for it in extra:
        if it["jenis"] == "non_botol":
            kat[it["kat"]] += rep[it["name"]]
    print(name, "salinan non-botol per jenis:", dict(sorted(kat.items(), key=lambda z: -z[1])), flush=True)
    open(f"{OUT}/{name}_train.txt", "w").write("\n".join(tr) + "\n")
    open(f"{OUT}/{name}.yaml", "w").write(f"train: {OUT}/{name}_train.txt\nval: {DS}/4_eksperimen/botol/old_test.txt\nnames:\n  0: bottle\n  1: non_bottle\n")
    YOLO(V7).train(data=f"{OUT}/{name}.yaml", epochs=10, imgsz=640, batch=16, workers=0, seed=0, deterministic=True, patience=0, plots=False,
                   verbose=False, project=OUT, name=name, exist_ok=True, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, degrees=0, fliplr=0.5,
                   flipud=0.0, mosaic=1.0, close_mosaic=3)
    return f"{OUT}/{name}/weights/last.pt"


def predict(pt, its):
    from ultralytics import YOLO
    m = YOLO(pt); out = []
    for it in its:
        r = m.predict(f"{NY}/images/{it['name']}", imgsz=640, conf=0.35, verbose=False)[0]
        c = [int(b.cls) for b in r.boxes]
        out.append((0 in c, 1 in c))
    return out


def report():
    R = [json.loads(l) for l in open(f"{OUT}/cv.jsonl") if l.strip()]
    v10 = {}
    if os.path.exists(f"{V10}/cv.jsonl"):
        for l in open(f"{V10}/cv.jsonl"):
            j = json.loads(l)
            if j["model"] == "ver10":
                v10[j["name"]] = j["terima"]
    L = ["ver11 (non-botol nyata DIBERI kotak non_bottle) — CV 5 fold per benda, foto NYATA 22 Okt. Ambang 0,35.",
         "Non-botol: 'diterima' = ada kotak bottle (BURUK), 'ditandai' = ada kotak non_bottle (dibutuhkan app untuk menolak benda asing).",
         "Botol: 'terdeteksi' = ada kotak bottle, 'salah non_bottle' = botol juga diberi kotak non_bottle.", ""]
    kats = sorted({r["kat"] for r in R if r["jenis"] == "non_botol"})
    for m in ("ver9c", "ver11"):
        rr = [r for r in R if r["model"] == m]
        nb = [r for r in rr if r["jenis"] == "non_botol"]; bt = [r for r in rr if r["jenis"] == "botol"]
        L.append(f"{m:6s} non-botol diterima {sum(r['terima'] for r in nb)}/{len(nb)}, DITANDAI non_bottle {sum(r['nb'] for r in nb)}/{len(nb)}, "
                 f"tanpa reaksi {sum(not r['terima'] and not r['nb'] for r in nb)}/{len(nb)} | botol terdeteksi {sum(r['terima'] for r in bt)}/{len(bt)}, "
                 f"botol salah non_bottle {sum(r['nb'] for r in bt)}/{len(bt)}")
        L.append("        diterima/ditandai per benda: " + ", ".join(
            f"{k} {sum(r['terima'] for r in nb if r['kat'] == k)}/{sum(r['nb'] for r in nb if r['kat'] == k)}/{sum(1 for r in nb if r['kat'] == k)}" for k in kats))
    nb11 = [r for r in R if r["model"] == "ver11" and r["jenis"] == "non_botol" and r["name"] in v10]
    if nb11:
        L.append(f"\nPembanding ver10 (CV-nya sendiri, item yang sama, tanpa semprot/Listerine): non-botol diterima ver10 "
                 f"{sum(v10[r['name']] for r in nb11)}/{len(nb11)} vs ver11 {sum(r['terima'] for r in nb11)}/{len(nb11)}; ver10 tidak pernah menandai non_bottle.")
    open(f"{OUT}/cv_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    os.makedirs(OUT, exist_ok=True)
    items = build(); F, load = folds(items)
    print(len(items), "foto nyata (", sum(i["jenis"] == "non_botol" for i in items), "non-botol berkotak ),", len({i["kel"] for i in items}),
          "kelompok; beban fold", load, flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        pt = train(items, "final"); shutil.copy2(pt, f"{DS}/3_model_final/botol_ver11_kandidat.pt")
        print("FINAL ->", f"{DS}/3_model_final/botol_ver11_kandidat.pt"); return
    res = f"{OUT}/cv.jsonl"
    done = {(json.loads(l)["fold"], json.loads(l)["model"]) for l in open(res)} if os.path.exists(res) else set()
    if (-1, "ver9c") not in done:
        with open(res, "a") as fh:
            for it, (t, n) in zip(items, predict(V9C, items)):
                fh.write(json.dumps(dict(fold=-1, model="ver9c", **it, terima=bool(t), nb=bool(n))) + "\n")
    for k in range(K):
        if (k, "ver11") in done:
            continue
        te = [i for i in items if F[i["kel"]] == k]
        pt = train([i for i in items if F[i["kel"]] != k], f"cv_f{k}")
        with open(res, "a") as fh:
            for it, (t, n) in zip(te, predict(pt, te)):
                fh.write(json.dumps(dict(fold=k, model="ver11", **it, terima=bool(t), nb=bool(n))) + "\n")
        print(f"fold {k}: uji {len(te)} selesai", flush=True)
    report()


if __name__ == "__main__":
    main()
