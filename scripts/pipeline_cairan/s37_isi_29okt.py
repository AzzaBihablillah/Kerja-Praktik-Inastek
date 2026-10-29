"""Langkah 37: model ISI v3 ('berisi apa pun' vs kosong) dari SEMUA data: resep s35/s36 + belt 29 Okt (uji tingkat isi).
Data 29 Okt (`1_foto_mentah/2026-10-29_scan_app/sesi_*`, label kelompok dari user, isi pasti):
  kosong (tanpa/bertutup) -> 'kosong';  air_1-2, air_1-6, air_penuh -> 'berwarna' (= berisi). Sesi salah di `_dibuang_user/` tidak dipakai.
Validasi PER HARI (botol & sesi yang tidak pernah dilatih):
  A: latih tanpa 29 Okt -> uji 29 Okt   = model s36 `cairan_berisi_v2_kandidat.pt` (dilatih s.d. 27 Okt; tak dilatih ulang)
  B: latih tanpa 27 Okt (+29 Okt)       -> uji 27 Okt
Pembanding pada botol yang sama: cairan_RH (aktif) dan cairan_berisi_kandidat (s35). Per botol = median p 3 frame, ambang 0,5.
Catatan kebocoran: botol fisik yang sama mungkin dipakai di 27 dan 29 Okt -> 'botol baru' tidak sepenuhnya terjamin.
  python scripts/pipeline_cairan/s37_isi_29okt.py         CV per hari -> out/cv_29okt.jsonl + out/cv_29okt_report.txt
  python scripts/pipeline_cairan/s37_isi_29okt.py final   -> datasets/3_model_final/cairan_isi_v3_kandidat.pt (model lain TIDAK ditimpa)"""
import csv, glob, json, os, random, shutil, sys
from collections import defaultdict
import cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop
from s29_hueaug import shifted, K as KH
from s32_cv_rn import rows_all, items_of, train, OUT, ROOT
from s36_berisi_27okt import rows_s35, rows_27okt, score, per_botol

T29 = f"{ROOT}/1_foto_mentah/2026-10-29_scan_app"; HD = f"{OUT}/crops_fold_hue"; RES = f"{OUT}/cv_29okt.jsonl"
M = f"{ROOT}/3_model_final"
ISI = {"kosong": "kosong", "air_1-2": "berwarna", "air_1-6": "berwarna", "air_penuh": "berwarna"}


def rows_29okt():
    """Potongan frame diam 29 Okt (+3 salinan geser-hue). id T29_<sesi>_<botol>_<frame>."""
    p = f"{OUT}/crops29.csv"
    if os.path.exists(p):
        return list(csv.DictReader(open(p)))
    from ultralytics import YOLO
    det = YOLO(f"{M}/botol_ver9c.pt"); rows = []
    for d in sorted(glob.glob(f"{T29}/sesi_*")):
        if not os.path.exists(f"{d}/SCAN.csv"):
            continue
        for r in csv.DictReader(open(f"{d}/SCAN.csv")):
            liq = ISI.get(r.get("k_isi", ""))
            if liq is None or r["alasan"] == "bukan_botol":
                continue
            for k in (1, 2, 3):
                f = f"{d}/s{r['sesi']}_b{r['botol']}_{r['waktu']}_{k}.png"
                if not os.path.exists(f):
                    continue
                im = cv2.imread(f)
                bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
                if not bb:
                    continue
                box = max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])); cid = f"T29_{r['sesi']}_{r['botol']}_{k}"
                cv2.imwrite(f"{OUT}/crops_fold/{cid}.jpg", fold_crop(im, *box), [cv2.IMWRITE_JPEG_QUALITY, 95])
                rng = random.Random(cid)
                for j, dh in enumerate(rng.sample([30, 60, 90, 120, 150], KH)):
                    cv2.imwrite(f"{HD}/{cid}_h{j}.jpg", fold_crop(shifted(im, box, dh), *box), [cv2.IMWRITE_JPEG_QUALITY, 95])
                prod = "coca" if r.get("k_produk", "").lower().startswith("coca") else "non"
                rows.append(dict(id=cid, src="T29", group=f"S{r['sesi']}", kel=f"T29_S{r['sesi']}", baru=1, liquid=liq,
                                 level=r["k_isi"], tutup=r["k_tutup"], produk=prod, botol=f"29_{r['sesi']}_{r['botol']}",
                                 path=os.path.relpath(f, ROOT)))
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return rows


def write(fold, model, rows, P, meta):
    with open(RES, "a") as fh:
        for b, (liq, ses, p) in per_botol(rows, P).items():
            fh.write(json.dumps(dict(fold=fold, model=model, botol=b, liquid=liq, p=round(p, 4), **meta.get(b, {}))) + "\n")


def report():
    J = [json.loads(l) for l in open(RES)]
    L = ["Model ISI v3 — validasi PER HARI (uji = hari yang tidak dilatih), per botol median 3 frame, ambang 0,5.", ""]
    for hari, fold in (("29 Okt (uji tingkat isi)", "A29"), ("27 Okt (air acak)", "B27")):
        L.append(f"== Uji {hari}")
        grp = sorted({j.get("level", j["liquid"]) for j in J if j["fold"] == fold})
        for m in ("RH_aktif", "s35", "v3"):
            jj = [j for j in J if j["fold"] == fold and j["model"] == m]
            if not jj:
                continue
            s = []
            for g in grp:
                x = [j for j in jj if j.get("level", j["liquid"]) == g]
                ok = sum((j["p"] >= .5) == (j["liquid"] == "berwarna") for j in x)
                s.append(f"{g} {ok}/{len(x)}")
            L.append(f"  {m:9s} " + " | ".join(s))
        L.append("")
    open(f"{OUT}/cv_29okt_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    base = rows_all() + rows_s35(); r27 = rows_27okt(); r29 = rows_29okt()
    print(len(base), "baris lama +", len(r27), "frame 27 Okt +", len(r29), "frame 29 Okt (", len({r['botol'] for r in r29}), "botol )", flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        pt = train(items_of(base + r27 + r29), f"{OUT}/final7_data", f"{OUT}/final7", "isi_v3_all")
        shutil.copy2(pt, f"{M}/cairan_isi_v3_kandidat.pt"); print("FINAL ->", f"{M}/cairan_isi_v3_kandidat.pt"); return
    meta29 = {r["botol"]: dict(level=r["level"], tutup=r["tutup"], produk=r["produk"]) for r in r29}
    meta27 = {r["botol"]: dict(level="air_acak" if r["liquid"] == "berwarna" else "kosong") for r in r27}
    done = {(json.loads(l)["fold"], json.loads(l)["model"]) for l in open(RES)} if os.path.exists(RES) else set()
    for m, pt in (("RH_aktif", f"{M}/cairan_RH.pt"), ("s35", f"{M}/cairan_berisi_kandidat.pt")):
        for fold, rows, meta in (("A29", r29, meta29), ("B27", r27, meta27)):
            if (fold, m) not in done:
                write(fold, m, rows, score(pt, rows), meta)
    if ("A29", "v3") not in done:                     # model s36 = dilatih base + 27 Okt (tanpa 29 Okt)
        write("A29", "v3", r29, score(f"{M}/cairan_berisi_v2_kandidat.pt", r29), meta29)
    if ("B27", "v3") not in done:
        pt = train(items_of(base + r29), f"{OUT}/cv29_data/b27", f"{OUT}/cv29", "b27")
        write("B27", "v3", r27, score(pt, r27), meta27)
    report()


if __name__ == "__main__":
    main()
