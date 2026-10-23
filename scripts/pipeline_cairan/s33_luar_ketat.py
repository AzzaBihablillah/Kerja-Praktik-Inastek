"""Langkah 33 (usul user 23 Okt): apakah data luar MEMBANTU bila potongannya BAGUS dan ketat (tanpa tepi latar)?
Data luar = Open Food Facts (1577 potongan, DUA kelas: berwarna 696 / kosong 881, label s14). RF100 soda (hanya berwarna) &
warisan (label belum diverifikasi) TIDAK dipakai (pasti jadi jalan pintas).
Hapus latar (rembg) DITOLAK: pada botol PET bening badan botol ikut terhapus (scratchpad rembg_belt.jpg) -> pakai potongan kotak KETAT.
Saringan kualitas data luar (kotak dari botol_ver9c pada foto asli):
  conf botol >= 0,6; kotak tidak menyentuh tepi foto (>= 2%); luas kotak 8-85% foto; sisi panjang/pendek >= 1,6 (bukan tampak atas);
  ketajaman (variansi Laplacian potongan abu) >= 60.
Potongan KETAT: fold_crop dengan pad 0 (data kita juga dibuat ulang ketat, kotak botol_ver9c).
CV: fold sama dengan s32 (data kita). Data luar hanya di LATIH, tidak pernah diuji. Resep tanpa geser-hue:
  T0 = data kita (ketat)   T1 = + luar berwarna/kosong (label user 23 Okt)   T2 = T1 + air bening luar sebagai 'kosong' (tidak berwarna)
  python scripts/pipeline_cairan/s33_luar_ketat.py siapkan   -> crops_fold_tight/ + luar_bersih.csv + LEMBAR-LUAR-BERSIH.jpg
  python scripts/pipeline_cairan/s33_luar_ketat.py cv        -> cv_tight.jsonl + cv_tight_report.txt"""
import csv, json, os, random, shutil, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop
from s32_cv_rn import rows_all, folds, train, auc, OUT, ROOT

TD = f"{OUT}/crops_fold_tight"


def fix(p):
    p = p.replace("\\", "/")
    for a, b in (("primary/", "1_foto_mentah/"), ("_dl_ext/", "5_data_luar/_dl_ext/")):
        if p.startswith(a):
            p = b + p[len(a):]
    return os.path.join(ROOT, p)


def best_box(det, im, conf=0.35):
    bb = [(float(b.conf), b.xyxy[0].tolist()) for b in det.predict(im, imgsz=640, conf=conf, verbose=False)[0].boxes if int(b.cls) == 0]
    return max(bb, key=lambda z: (z[1][2] - z[1][0]) * (z[1][3] - z[1][1])) if bb else None


def siapkan():
    from ultralytics import YOLO
    os.makedirs(TD, exist_ok=True); det = YOLO(f"{ROOT}/3_model_final/botol_ver9c.pt")
    ours = rows_all(); miss = 0
    for r in ours:                                                     # data kita: potongan ketat
        dst = f"{TD}/{r['id']}.jpg"
        if os.path.exists(dst):
            continue
        im = cv2.imread(fix(r["path"]))
        b = best_box(det, im) if im is not None else None
        if b is None:
            miss += 1; continue
        cv2.imwrite(dst, fold_crop(im, *b[1], pad=0.0), [cv2.IMWRITE_JPEG_QUALITY, 95])
    print("data kita tanpa kotak/foto:", miss, flush=True)
    keep = []; why = {}
    for r in csv.DictReader(open(f"{OUT}/crops_ext.csv", encoding="utf-8", errors="replace")):
        im = cv2.imread(fix(r["path"]))
        if im is None:
            why["foto"] = why.get("foto", 0) + 1; continue
        H, W = im.shape[:2]; b = best_box(det, im, 0.6)
        if b is None:
            why["conf<0,6"] = why.get("conf<0,6", 0) + 1; continue
        x1, y1, x2, y2 = b[1]; w, h = x2 - x1, y2 - y1
        if min(x1, y1) < 0.02 * min(W, H) or x2 > W * 0.98 or y2 > H * 0.98:
            why["menyentuh tepi"] = why.get("menyentuh tepi", 0) + 1; continue
        if not 0.08 <= w * h / (W * H) <= 0.85:
            why["luas"] = why.get("luas", 0) + 1; continue
        if max(w, h) / max(1, min(w, h)) < 1.6:
            why["bukan bentuk botol"] = why.get("bukan bentuk botol", 0) + 1; continue
        c = cv2.cvtColor(im[int(y1):int(y2), int(x1):int(x2)], cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(c, cv2.CV_64F).var() < 60:
            why["buram"] = why.get("buram", 0) + 1; continue
        cv2.imwrite(f"{TD}/{r['id']}.jpg", fold_crop(im, *b[1], pad=0.0), [cv2.IMWRITE_JPEG_QUALITY, 95])
        keep.append(dict(id=r["id"], liquid=r["liquid"], product=r["product"]))
    with open(f"{OUT}/luar_bersih.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["id", "liquid", "product"]); w.writeheader(); w.writerows(keep)
    from collections import Counter
    print("Open Food Facts lolos saringan:", len(keep), Counter(k["liquid"] for k in keep), "| dibuang:", why, flush=True)
    rng = random.Random(1); T = 150; rowsimg = []
    for title, lst in (("DATA KITA (ketat)", [f"{TD}/{r['id']}.jpg" for r in ours if os.path.exists(f"{TD}/{r['id']}.jpg")]),
                       ("LUAR berwarna (lolos saringan)", [f"{TD}/{k['id']}.jpg" for k in keep if k["liquid"] == "berwarna"]),
                       ("LUAR kosong (lolos saringan)", [f"{TD}/{k['id']}.jpg" for k in keep if k["liquid"] == "kosong"])):
        fs = rng.sample(lst, min(14, len(lst)))
        ims = [cv2.resize(cv2.imread(f), (T, T)) for f in fs]
        while len(ims) < 14:
            ims.append(np.full((T, T, 3), 200, np.uint8))
        bar = np.full((34, T * 7, 3), 245, np.uint8); cv2.putText(bar, title, (8, 24), 0, 0.7, (20, 20, 20), 2)
        rowsimg += [bar, np.hstack(ims[:7]), np.hstack(ims[7:])]
    cv2.imwrite(f"{OUT}/LEMBAR-LUAR-BERSIH.jpg", np.vstack(rowsimg), [cv2.IMWRITE_JPEG_QUALITY, 85])


def cv():
    from ultralytics import YOLO
    rows = [r for r in rows_all() if os.path.exists(f"{TD}/{r['id']}.jpg")]; F, load = folds(rows)
    # label AKHIR dari validasi user 23 Okt (luar_bersih_relabel.csv kolom label_user): berwarna / kosong / air_bening / buang
    rel = list(csv.DictReader(open(f"{OUT}/luar_bersih_relabel.csv", encoding="utf-8")))
    ext1 = [dict(id=r["id"], liquid=r["label_user"]) for r in rel if r["label_user"] in ("berwarna", "kosong")]
    ext2 = ext1 + [dict(id=r["id"], liquid="kosong") for r in rel if r["label_user"] == "air_bening"]   # air bening = tidak berwarna
    res = f"{OUT}/cv_tight.jsonl"
    done = {(json.loads(l)["fold"], json.loads(l)["resep"]) for l in open(res)} if os.path.exists(res) else set()
    for k in range(5):
        te = [r for r in rows if F[r["kel"]] == k]
        for rc in ("T0", "T1", "T2"):
            if (k, rc) in done:
                continue
            items = [(f"{TD}/{r['id']}.jpg", r["liquid"], r["id"]) for r in rows if F[r["kel"]] != k]
            if rc in ("T1", "T2"):
                items += [(f"{TD}/{e['id']}.jpg", e["liquid"], e["id"]) for e in (ext1 if rc == "T1" else ext2)]
            pt = train(items, f"{OUT}/tmp_tight/{rc}_{k}", f"{OUT}/runs_tight", f"{rc}_f{k}")
            m = YOLO(pt); cb = [i for i, v in m.names.items() if v == "berwarna"][0]
            ps = [float(p.probs.data[cb]) for p in m.predict([f"{TD}/{r['id']}.jpg" for r in te], imgsz=224, verbose=False)]
            with open(res, "a") as fh:
                for r, p in zip(te, ps):
                    fh.write(json.dumps(dict(fold=k, resep=rc, id=r["id"], src=r["src"], kel=r["kel"], baru=r["baru"], liquid=r["liquid"], p=round(p, 4))) + "\n")
            print(f"fold {k} {rc}: latih {len(items)}, uji {len(te)}", flush=True)
    R = [json.loads(l) for l in open(res) if l.strip()]
    L = ["Uji usul user: data luar (Open Food Facts) dengan potongan BAGUS & KETAT. T0 = data kita saja, T1 = + luar tervalidasi user (14 berwarna, 4 kosong), T2 = T1 + 15 air bening sebagai tidak berwarna.", ""]
    for name, f in (("SEMUA", lambda r: True), ("data lama 16/20 Okt", lambda r: r["baru"] == 0), ("V22 label penuh", lambda r: r["src"] == "V22"),
                    ("S22 frame belt", lambda r: r["src"] == "S22")):
        L.append(f"== {name}")
        for rc in ("T0", "T1", "T2"):
            rr = [x for x in R if x["resep"] == rc and f(x)]
            pos = [x["p"] for x in rr if x["liquid"] == "berwarna"]; neg = [x["p"] for x in rr if x["liquid"] == "kosong"]
            L.append(f"   {rc}: AUC {auc(pos, neg):.3f} | berisi terdeteksi {sum(p >= .5 for p in pos)}/{len(pos)} | kosong benar {sum(p < .5 for p in neg)}/{len(neg)}")
    open(f"{OUT}/cv_tight_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    {"siapkan": siapkan, "cv": cv}[sys.argv[1]]()
