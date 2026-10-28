"""ver12 = resep ver11 + foto belt NYATA sesi 27 Okt (Kiosk Capturer, 3 frame diam per benda).
Kotak = usulan ver11 per frame, DIAUDIT user 28 Okt (https://claude.ai/artifact/Fh3joDUr8T3YuyfTAfPGhA; 3 'salah' tidak dipakai).
Kelas mengikuti SESI (sesi6 = non_bottle, sesi1-5 = bottle). Deteksi lain di frame (kotak kuning ver11) setelah dicek 28 Okt
ternyata pantulan cahaya di belt (tepi kiri), rangka/benda gelap di ujung belt (tepi kanan) dan sekali tangan, BUKAN botol
terpotong -> dibiarkan TANPA label = contoh negatif (model belajar itu bukan benda). TUTUP_ABU=True mengembalikan cara lama.
Uji: CV 2 fold per PASANGAN SESI (botol fisik yang sama dipakai di sesi1&2 dan sesi3&4, jadi dipasangkan):
  fold A uji sesi1+2+5 (Coca & air), fold B uji sesi3+4+6 (non-Coca & bukan botol). Data 22 Okt (ver11) selalu ikut latih.
  Dinilai per FRAME asli (tanpa tutup abu): botol terdeteksi (kotak bottle conf>=0,35 IoU>=0,5 dg kotak audit), bukan botol
  ditandai (kotak non_bottle), botol salah non_bottle, dan jumlah deteksi tambahan (alarm palsu) per frame. Pembanding: ver11.
  python scripts/pipeline_botol/ver12_belt27.py          CV -> datasets/4_eksperimen/botol/ver12/cv.jsonl + cv_report.txt
  python scripts/pipeline_botol/ver12_belt27.py final    latih semua -> datasets/3_model_final/botol_ver12_kandidat.pt"""
import csv, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ver11_nonbotol as v11

DS = v11.DS; OUT = f"{DS}/4_eksperimen/botol/ver12"; NY = f"{OUT}/belt27"
AUD = f"{DS}/4_eksperimen/ver12_audit"; REPO = os.path.dirname(DS)
SALAH = {"V53B214", "V60B33", "V60B45"}                     # audit user 28 Okt
TUTUP_ABU = False
FOLD = {"sesi1": 0, "sesi2": 0, "sesi5": 0, "sesi3": 1, "sesi4": 1, "sesi6": 1}
v11.OUT = OUT


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def build27():
    if os.path.exists(f"{NY}/items.json"):
        return json.load(open(f"{NY}/items.json"))
    import cv2
    os.makedirs(f"{NY}/images", exist_ok=True); os.makedirs(f"{NY}/labels", exist_ok=True); items = []
    for r in csv.DictReader(open(f"{AUD}/usulan.csv", encoding="utf-8")):
        if r["id"] in SALAH:
            continue
        cls = 1 if r["kelas"] == "non_bottle" else 0
        for k, (f, main, lain) in enumerate(json.loads(r["kotak"])):
            if not main:
                continue
            im = cv2.imread(os.path.join(REPO, f)); H, W = im.shape[:2]
            for z in lain:                                     # tutup abu deteksi lain (tidak bertumpuk dg kotak utama)
                x1, y1, x2, y2 = (int(round(v)) for v in z)
                if TUTUP_ABU and iou(z, main) < 0.3:
                    im[max(0, y1):y2, max(0, x1):x2] = 114
            name = f"b27_{r['id']}_{k + 1}"
            cv2.imwrite(f"{NY}/images/{name}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 95])
            x1, y1, x2, y2 = main
            open(f"{NY}/labels/{name}.txt", "w").write(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
            items.append(dict(name=name + ".jpg", asli=f, kotak=main, sesi=r["sesi"], obj=r["id"], jenis="non_botol" if cls else "botol"))
    json.dump(items, open(f"{NY}/items.json", "w"), indent=0)
    return items


def train(old, new, name):
    """Resep ver11 (rep_seimbang untuk non-botol 22 Okt) + frame belt 27 Okt x REP_POS (botol & bukan botol)."""
    from ultralytics import YOLO
    base = [l.strip() for l in open(f"{DS}/4_eksperimen/botol/ver9c_train.txt") if l.strip()]
    rep = v11.rep_seimbang(old)
    tr = base + [f"{v11.NY}/images/{it['name']}" for it in old for _ in range(rep[it["name"]] if it["jenis"] == "non_botol" else v11.REP_POS)]
    tr += [f"{NY}/images/{it['name']}" for it in new for _ in range(v11.REP_POS)]
    print(name, "latih", len(tr), "baris (", len(new), "frame belt 27 Okt )", flush=True)
    open(f"{OUT}/{name}_train.txt", "w").write("\n".join(tr) + "\n")
    open(f"{OUT}/{name}.yaml", "w").write(f"train: {OUT}/{name}_train.txt\nval: {DS}/4_eksperimen/botol/old_test.txt\nnames:\n  0: bottle\n  1: non_bottle\n")
    YOLO(v11.V7).train(data=f"{OUT}/{name}.yaml", epochs=10, imgsz=640, batch=16, workers=0, seed=0, deterministic=True, patience=0, plots=False,
                       verbose=False, project=OUT, name=name, exist_ok=True, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, degrees=0, fliplr=0.5,
                       flipud=0.0, mosaic=1.0, close_mosaic=3)
    return f"{OUT}/{name}/weights/last.pt"


def evaluate(pt, its, model, fold):
    from ultralytics import YOLO
    m = YOLO(pt); rows = []
    for it in its:
        bx = [(int(b.cls), [float(v) for v in b.xyxy[0]]) for b in m.predict(os.path.join(REPO, it["asli"]), imgsz=640, conf=0.35, verbose=False)[0].boxes]
        hit = [c for c, b in bx if iou(b, it["kotak"]) >= 0.5]
        rows.append(dict(fold=fold, model=model, name=it["name"], sesi=it["sesi"], obj=it["obj"], jenis=it["jenis"],
                         bottle=0 in hit, nb=1 in hit, lain=sum(iou(b, it["kotak"]) < 0.3 for _, b in bx)))
    return rows


def report():
    R = [json.loads(l) for l in open(f"{OUT}/cv.jsonl") if l.strip()]
    L = ["ver12 (ver11 + frame belt 27 Okt) — CV 2 fold per pasangan sesi, dinilai per FRAME asli, ambang 0,35, IoU>=0,5 dg kotak audit.", ""]
    for f, nm in ((0, "fold A: uji sesi1+2+5"), (1, "fold B: uji sesi3+4+6")):
        L.append(nm)
        for m in ("ver11", "ver12"):
            rr = [r for r in R if r["model"] == m and r["fold"] == f]
            bt = [r for r in rr if r["jenis"] == "botol"]; nb = [r for r in rr if r["jenis"] == "non_botol"]
            s = f"  {m}: botol terdeteksi {sum(r['bottle'] for r in bt)}/{len(bt)}, botol salah non_bottle {sum(r['nb'] for r in bt)}/{len(bt)}"
            if nb:
                s += f" | bukan botol DITANDAI {sum(r['nb'] for r in nb)}/{len(nb)}, bukan botol diterima {sum(r['bottle'] for r in nb)}/{len(nb)}"
            s += f" | frame dg deteksi tambahan {sum(r['lain'] > 0 for r in rr)}/{len(rr)}"
            L.append(s)
    open(f"{OUT}/cv_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    os.makedirs(OUT, exist_ok=True)
    old = v11.build(); new = build27()
    print(len(old), "foto 22 Okt +", len(new), "frame belt 27 Okt (", sum(i["jenis"] == "non_botol" for i in new), "bukan botol )", flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        pt = train(old, new, "final"); shutil.copy2(pt, f"{DS}/3_model_final/botol_ver12_kandidat.pt")
        print("FINAL ->", f"{DS}/3_model_final/botol_ver12_kandidat.pt"); return
    res = f"{OUT}/cv.jsonl"
    done = {(json.loads(l)["fold"], json.loads(l)["model"]) for l in open(res)} if os.path.exists(res) else set()
    for k in (0, 1):
        te = [i for i in new if FOLD[i["sesi"]] == k]
        if (k, "ver11") not in done:
            with open(res, "a") as fh:
                for r in evaluate(f"{DS}/3_model_final/botol_ver11_kandidat.pt", te, "ver11", k):
                    fh.write(json.dumps(r) + "\n")
        if (k, "ver12") in done:
            continue
        pt = train(old, [i for i in new if FOLD[i["sesi"]] != k], f"cv_f{k}")
        with open(res, "a") as fh:
            for r in evaluate(pt, te, "ver12", k):
                fh.write(json.dumps(r) + "\n")
        print(f"fold {k}: uji {len(te)} frame selesai", flush=True)
    report()


if __name__ == "__main__":
    main()
