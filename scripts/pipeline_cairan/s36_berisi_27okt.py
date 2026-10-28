"""Langkah 36: latih ulang model cairan 'berisi apa pun' (resep s35) + data belt NYATA 27 Okt (Kiosk Capturer).
Data baru (label kelompok dari user, isi pasti: user sendiri yang mengisi/mengosongkan):
  sesi1-4 (botol kosong, 72 botol) -> 'kosong'; sesi5 (botol berisi air bening penuh/sedikit, 35 botol) -> 'berwarna' (= berisi).
Kelompok CV = SESI (bukan botol): botol fisik yang sama kemungkinan dipakai ulang di beberapa sesi (mis. sesi1 & sesi2) ->
pengelompokan per sesi mencegah kebocoran. Tiap fold menguji SATU sesi 27 Okt yang tidak dilatih (+ kelompok lama seperti biasa).
Pembanding pada sesi 27 Okt yang sama: cairan_berisi_kandidat (s35, tak pernah melihat data 27 Okt) dan cairan_RH (aktif).
Keputusan per botol = median p dari 3 frame diam (sama dengan aplikasi), ambang 0,5.
  python scripts/pipeline_cairan/s36_berisi_27okt.py         CV -> out/cv_27okt.jsonl + out/cv_27okt_report.txt
  python scripts/pipeline_cairan/s36_berisi_27okt.py final   -> datasets/3_model_final/cairan_berisi_v2_kandidat.pt (kandidat lama TIDAK ditimpa)"""
import csv, glob, json, os, random, shutil, statistics, sys
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop
from s29_hueaug import shifted, K as KH
from s32_cv_rn import rows_all, folds, items_of, train, OUT, ROOT

T27 = f"{ROOT}/1_foto_mentah/2026-10-27_scan_app"; HD = f"{OUT}/crops_fold_hue"; RES = f"{OUT}/cv_27okt.jsonl"


def rows_s35():
    """Baris air bening 22 Okt dari s35 (crops22b.csv, sudah ada potongannya)."""
    rr = list(csv.DictReader(open(f"{OUT}/crops22b.csv")))
    for r in rr:
        r["kel"] = r["group"]; r["baru"] = 1
    return rr


def rows_27okt():
    """Potongan frame diam 27 Okt (+3 salinan geser-hue). id T27_<sesi>_<botol>_<frame>."""
    p = f"{OUT}/crops27.csv"
    if os.path.exists(p):
        return list(csv.DictReader(open(p)))
    from ultralytics import YOLO
    det = YOLO(f"{ROOT}/3_model_final/botol_ver9c.pt"); rows = []
    for d in sorted(glob.glob(f"{T27}/sesi[1-5]_*")):
        ses = os.path.basename(d).split("_")[0]
        for r in csv.DictReader(open(f"{d}/SCAN.csv")):
            if not r.get("harapan") or r["alasan"] == "bukan_botol" or r.get("k_tutup") == "buang_audit":
                continue
            liq = "kosong" if r["k_isi"] == "kosong" else ("berwarna" if r["k_isi"] == "air_bening" else None)
            if liq is None:
                continue
            for k in (1, 2, 3):
                f = f"{d}/s{r['sesi']}_b{r['botol']}_{r['waktu']}_{k}.png"
                if not os.path.exists(f):
                    continue
                im = cv2.imread(f)
                bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
                if not bb:
                    continue
                box = max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])); cid = f"T27_{r['sesi']}_{r['botol']}_{k}"
                cv2.imwrite(f"{OUT}/crops_fold/{cid}.jpg", fold_crop(im, *box), [cv2.IMWRITE_JPEG_QUALITY, 95])
                rng = random.Random(cid)
                for j, dh in enumerate(rng.sample([30, 60, 90, 120, 150], KH)):
                    cv2.imwrite(f"{HD}/{cid}_h{j}.jpg", fold_crop(shifted(im, box, dh), *box), [cv2.IMWRITE_JPEG_QUALITY, 95])
                rows.append(dict(id=cid, src="T27", group=ses, kel="T27_" + ses, baru=1, product=r.get("k_produk", ""),
                                 liquid=liq, botol=f"{r['sesi']}_{r['botol']}", path=os.path.relpath(f, ROOT)))
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def score(pt, rows):
    """p(berisi) per potongan untuk daftar baris (frame) -> dict id -> p."""
    from ultralytics import YOLO
    m = YOLO(pt); ci = [i for i, v in m.names.items() if v == "berwarna"][0]; out = {}
    for s in range(0, len(rows), 64):
        ch = rows[s:s + 64]
        for r, p in zip(ch, m.predict([f"{OUT}/crops_fold/{r['id']}.jpg" for r in ch], imgsz=224, verbose=False)):
            out[r["id"]] = float(p.probs.data[ci])
    return out


def per_botol(rows, P):
    g = {}
    for r in rows:
        if r["id"] in P:
            g.setdefault(r["botol"], [r["liquid"], r["group"], []])[2].append(P[r["id"]])
    return {b: (liq, ses, statistics.median(ps)) for b, (liq, ses, ps) in g.items()}


def report():
    J = [json.loads(l) for l in open(RES)]
    L = ["Model cairan 'berisi' + data 27 Okt — CV 5 fold per SESI 27 Okt (uji = sesi tak dilatih), keputusan = median 3 frame, ambang 0,5.", ""]
    for m in ("RH_aktif", "berisi_lama", "berisi_v2"):
        jj = [j for j in J if j["model"] == m]
        kos = [j for j in jj if j["liquid"] == "kosong"]; isi = [j for j in jj if j["liquid"] == "berwarna"]
        L.append(f"{m:12s} kosong salah 'berisi' {sum(j['p'] >= .5 for j in kos):3d}/{len(kos)} | air terdeteksi {sum(j['p'] >= .5 for j in isi):3d}/{len(isi)}"
                 + "   per sesi: " + ", ".join(f"{s} {sum((j['p'] >= .5) == (j['liquid'] == 'berwarna') for j in jj if j['sesi'] == s)}/"
                                             f"{sum(1 for j in jj if j['sesi'] == s)}" for s in sorted({j['sesi'] for j in jj})))
    lama = [j for j in J if j["model"] == "v2_kelompok_lama"]
    if lama:
        L.append(f"\nKelompok LAMA (regresi, frame): berisi_v2 benar {sum((j['p'] >= .5) == (j['liquid'] == 'berwarna') for j in lama)}/{len(lama)} "
                 f"(berwarna {sum(j['p'] >= .5 for j in lama if j['liquid'] == 'berwarna')}/{sum(j['liquid'] == 'berwarna' for j in lama)}, "
                 f"kosong {sum(j['p'] < .5 for j in lama if j['liquid'] == 'kosong')}/{sum(j['liquid'] == 'kosong' for j in lama)})")
    open(f"{OUT}/cv_27okt_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    base = rows_all() + rows_s35(); new = rows_27okt(); rows = base + new
    print(len(base), "baris lama +", len(new), "frame 27 Okt (", len({r['botol'] for r in new}), "botol )", flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        pt = train(items_of(rows), f"{OUT}/final6/data", f"{OUT}/final6", "berisi_v2_all")
        shutil.copy2(pt, f"{ROOT}/3_model_final/cairan_berisi_v2_kandidat.pt"); print("FINAL ->", f"{ROOT}/3_model_final/cairan_berisi_v2_kandidat.pt"); return
    F, load = folds(rows); print("beban fold", load, "| sesi 27 Okt per fold:", {g: F[g] for g in sorted({r['kel'] for r in new})}, flush=True)
    done = {(json.loads(l)["fold"], json.loads(l)["model"]) for l in open(RES)} if os.path.exists(RES) else set()
    if (-1, "berisi_lama") not in done:                                          # pembanding: tak pernah melihat 27 Okt
        with open(RES, "a") as fh:
            for m, pt in (("berisi_lama", f"{ROOT}/3_model_final/cairan_berisi_kandidat.pt"), ("RH_aktif", f"{ROOT}/3_model_final/cairan_RH.pt")):
                for b, (liq, ses, p) in per_botol(new, score(pt, new)).items():
                    fh.write(json.dumps(dict(fold=-1, model=m, botol=b, sesi=ses, liquid=liq, p=round(p, 4))) + "\n")
    for k in range(5):
        if (k, "berisi_v2") in done:
            continue
        tr = [r for r in rows if F[r["kel"]] != k]; te_new = [r for r in new if F[r["kel"]] == k]; te_old = [r for r in base if F[r["kel"]] == k]
        pt = train(items_of(tr), f"{OUT}/cv27_data/f{k}", f"{OUT}/cv27", f"f{k}")
        P = score(pt, te_new + te_old)
        with open(RES, "a") as fh:
            for b, (liq, ses, p) in per_botol(te_new, P).items():
                fh.write(json.dumps(dict(fold=k, model="berisi_v2", botol=b, sesi=ses, liquid=liq, p=round(p, 4))) + "\n")
            for r in te_old:
                if r["id"] in P:
                    fh.write(json.dumps(dict(fold=k, model="v2_kelompok_lama", id=r["id"], liquid=r["liquid"], p=round(P[r["id"]], 4))) + "\n")
        print(f"fold {k}: uji {len({r['botol'] for r in te_new})} botol 27 Okt + {len(te_old)} frame lama", flush=True)
    report()


if __name__ == "__main__":
    main()
