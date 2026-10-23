"""Langkah 34 (keputusan user 23 Okt: botol berisi AIR BENING juga harus DITOLAK): definisi baru 'berisi apa pun' vs 'kosong'.
Kelas model tetap bernama 'berwarna'/'kosong' (kompatibel dengan rvm_kp), tetapi artinya kini 'berisi' vs 'kosong'.
Data air bening di belt (dulu dibuang dari latih, kini = berisi):
  V22 b01 (Mizone penuh, cairan bening) + b03 (Mizone setengah) -> kelompok V22_mizone (botol fisik yang sama dgn Mizone kosong)
  S22 frame diam aplikasi berlabel 'berisi air bening' (Le Minerale, Cheers, Coca-Cola isi air) -> kelompok GROUP_SCAN
Data luar tervalidasi user (luar_bersih_relabel.csv): berwarna & air_bening -> berisi, kosong -> kosong.
Potongan KETAT, tanpa geser-hue (sama dengan s33). CV 5 fold per kelompok (fold dihitung ulang karena kelompok bertambah):
  B0 = data kita (+ air bening belt sebagai berisi)      B1 = B0 + data luar tervalidasi
  python scripts/pipeline_cairan/s34_berisi_bening.py   -> out/cv_bening.jsonl + out/cv_bening_report.txt"""
import csv, glob, json, os, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop
from s32_cv_rn import rows_all, folds, train, auc, OUT, ROOT
from s33_luar_ketat import TD
from s31_data_22okt import GROUP_SCAN

DV = f"{ROOT}/1_foto_mentah/2026-10-22_water_v1"; DS_ = f"{ROOT}/1_foto_mentah/2026-10-22_scan_app"


def bening_rows():
    from ultralytics import YOLO
    det = YOLO(f"{ROOT}/3_model_final/botol_ver9c.pt"); src = []
    for f in sorted(glob.glob(f"{DV}/b*.png")):
        t = os.path.basename(f).split("_")
        if "bening" in t[2] and t[3] in ("penuh", "setengah"):
            src.append((f, "V22_mizone"))
    for i, r in enumerate(csv.DictReader(open(f"{DS_}/SCAN.csv")), 1):
        if "bening" in (r["label_manual"] or ""):
            for k in (1, 2, 3):
                f = f"{DS_}/s{r['sesi']}_b{r['botol']}_{r['waktu']}_{k}.png"
                if os.path.exists(f):
                    src.append((f, GROUP_SCAN[i]))
    rows = []
    for j, (f, g) in enumerate(src):
        cid = f"B22_{j:04d}"; dst = f"{TD}/{cid}.jpg"
        if not os.path.exists(dst):
            im = cv2.imread(f)
            bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
            if not bb:
                continue
            cv2.imwrite(dst, fold_crop(im, *max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])), pad=0.0), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append(dict(id=cid, src="B22", kel=g, baru=1, liquid="berwarna"))
    return rows


def main():
    from ultralytics import YOLO
    rows = [r for r in rows_all() if os.path.exists(f"{TD}/{r['id']}.jpg")] + bening_rows()
    F, load = folds(rows)
    print(len(rows), "foto,", len(F), "kelompok; beban", load, "| air bening belt:", sum(r["src"] == "B22" for r in rows), flush=True)
    rel = list(csv.DictReader(open(f"{OUT}/luar_bersih_relabel.csv", encoding="utf-8")))
    ext = [dict(id=r["id"], liquid="kosong" if r["label_user"] == "kosong" else "berwarna") for r in rel
           if r["label_user"] in ("berwarna", "air_bening", "kosong")]
    res = f"{OUT}/cv_bening.jsonl"
    done = {(json.loads(l)["fold"], json.loads(l)["resep"]) for l in open(res)} if os.path.exists(res) else set()
    for k in range(5):
        te = [r for r in rows if F[r["kel"]] == k]
        for rc in ("B0", "B1"):
            if (k, rc) in done:
                continue
            items = [(f"{TD}/{r['id']}.jpg", r["liquid"], r["id"]) for r in rows if F[r["kel"]] != k]
            if rc == "B1":
                items += [(f"{TD}/{e['id']}.jpg", e["liquid"], e["id"]) for e in ext]
            pt = train(items, f"{OUT}/tmp_bening/{rc}_{k}", f"{OUT}/runs_bening", f"{rc}_f{k}")
            m = YOLO(pt); cb = [i for i, v in m.names.items() if v == "berwarna"][0]
            ps = [float(p.probs.data[cb]) for p in m.predict([f"{TD}/{r['id']}.jpg" for r in te], imgsz=224, verbose=False)]
            with open(res, "a") as fh:
                for r, p in zip(te, ps):
                    fh.write(json.dumps(dict(fold=k, resep=rc, id=r["id"], src=r["src"], kel=r["kel"], baru=r["baru"], liquid=r["liquid"], p=round(p, 4))) + "\n")
            print(f"fold {k} {rc}: latih {len(items)}, uji {len(te)}", flush=True)
    R = [json.loads(l) for l in open(res) if l.strip()]
    L = ["Definisi baru (user 23 Okt): BERISI apa pun (berwarna atau bening) vs KOSONG. B0 = data kita (+ air bening belt), B1 = + data luar tervalidasi (bening = berisi).", ""]
    for name, f in (("SEMUA", lambda r: True), ("cairan berwarna lama 16/20 Okt", lambda r: r["baru"] == 0),
                    ("AIR BENING di belt (Mizone, Le Minerale, Cheers, Coca-Cola isi air)", lambda r: r["src"] == "B22"),
                    ("V22 berlabel penuh", lambda r: r["src"] == "V22"), ("S22 frame belt (non-bening)", lambda r: r["src"] == "S22")):
        L.append(f"== {name}")
        for rc in ("B0", "B1"):
            rr = [x for x in R if x["resep"] == rc and f(x)]
            pos = [x["p"] for x in rr if x["liquid"] == "berwarna"]; neg = [x["p"] for x in rr if x["liquid"] == "kosong"]
            a = auc(pos, neg) if pos and neg else float("nan")
            L.append(f"   {rc}: AUC {a:.3f} | berisi terdeteksi {sum(p >= .5 for p in pos)}/{len(pos)} | kosong benar {sum(p < .5 for p in neg)}/{len(neg)}")
    open(f"{OUT}/cv_bening_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
