"""Model merek langkah 9: resep A + frame belt 27 Okt (m08) + frame belt 29 Okt (uji tingkat isi).
Label 29 Okt = label kelompok produk per sesi dari user: 'coca cola co.' -> coca_cola, 'NON coca cola co.' -> non_coca_cola;
sesi berlabel campur (galon 2, penuh maksimal) TIDAK dipakai (merek per botol tak tercatat). Sesi `_dibuang_user/` tidak dipakai.
Validasi PER HARI (botol & sesi yang tidak dilatih):
  A: latih tanpa 29 Okt -> uji 29 Okt  = model m08 `merek_coca_v2_kandidat.pt` (dipakai bonus di demo; tak dilatih ulang)
  B: latih tanpa 27 Okt (+29 Okt)      -> uji 27 Okt
Pembanding: m07 `merek_coca_kandidat.pt`. Per botol = rata-rata p_coca 3 frame (rvm_kp.merek.Merek), ambang 0,5.
Catatan: 29 Okt memakai botol fisik yang sama berulang (25 Coca, 50 non) di beberapa sesi -> dihitung per botol per sesi.
  python scripts/pipeline_merek/m09_belt29.py         -> datasets/4_eksperimen/merek/cv_belt29.jsonl + cv_belt29_report.txt
  python scripts/pipeline_merek/m09_belt29.py final   -> datasets/3_model_final/merek_coca_v3_kandidat.pt (model lain TIDAK ditimpa)"""
import csv, glob, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m02_cv_dasar import ROOT, OUT, CR, rows, train, brand_crop
from m08_belt27 import NAMA, rows27, score
import cv2

T29 = f"{ROOT}/1_foto_mentah/2026-10-29_scan_app"; RES = f"{OUT}/cv_belt29.jsonl"; M = f"{ROOT}/3_model_final"


def rows29():
    p = f"{OUT}/crops29.csv"
    if os.path.exists(p):
        return list(csv.DictReader(open(p)))
    from ultralytics import YOLO
    det = YOLO(f"{M}/botol_ver9c.pt"); out = []
    for d in sorted(glob.glob(f"{T29}/sesi_*")):
        if not os.path.exists(f"{d}/SCAN.csv"):
            continue
        for r in csv.DictReader(open(f"{d}/SCAN.csv")):
            pr = r.get("k_produk", "").strip().lower()
            lab = "coca_cola" if pr.startswith("coca") else ("non_coca_cola" if pr.startswith("non") else None)
            if lab is None or r["alasan"] == "bukan_botol":
                continue
            for k in (1, 2, 3):
                f = f"{d}/s{r['sesi']}_b{r['botol']}_{r['waktu']}_{k}.png"
                if not os.path.exists(f):
                    continue
                im = cv2.imread(f)
                bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
                if not bb:
                    continue
                cid = f"B29_S{r['sesi']}B{r['botol']}_{k}"
                cv2.imwrite(f"{CR}/{cid}.jpg", brand_crop(im, *max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))), [cv2.IMWRITE_JPEG_QUALITY, 95])
                out.append(dict(id=cid, botol=f"S{r['sesi']}B{r['botol']}", sesi=f"S{r['sesi']}", label=lab,
                                tutup=r["k_tutup"], isi=r["k_isi"]))
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0])); w.writeheader(); w.writerows(out)
    return out


def report():
    J = [json.loads(l) for l in open(RES)]
    L = ["Merek v3 — validasi PER HARI (uji = hari yang tidak dilatih), per botol rata-rata 3 frame, ambang 0,5.", ""]
    for fold, nm in (("A29", "uji 29 Okt"), ("B27", "uji 27 Okt")):
        L.append(f"== {nm}")
        for m in ("m07", "m08_v2", "v3"):
            jj = [j for j in J if j["fold"] == fold and j["model"] == m]
            if not jj:
                continue
            co = [j for j in jj if j["label"] == "coca_cola"]; no = [j for j in jj if j["label"] == "non_coca_cola"]
            cot = [j for j in co if j.get("tutup") == "bertutup"]; con = [j for j in co if j.get("tutup") == "tanpa"]
            L.append(f"  {m:7s} benar {sum((j['p'] >= .5) == (j['label'] == 'coca_cola') for j in jj)}/{len(jj)} | Coca dikenali "
                     f"{sum(j['p'] >= .5 for j in co)}/{len(co)} (bertutup {sum(j['p'] >= .5 for j in cot)}/{len(cot)}, tanpa "
                     f"{sum(j['p'] >= .5 for j in con)}/{len(con)}) | non salah bonus {sum(j['p'] >= .5 for j in no)}/{len(no)}")
        L.append("")
    open(f"{OUT}/cv_belt29_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    old = [(f"{CR}/{r['id']}.jpg", NAMA[r["kelas"]], r["id"]) for r in rows() if os.path.exists(f"{CR}/{r['id']}.jpg")]
    r27 = rows27(); r29 = rows29()
    it = lambda rr: [(f"{CR}/{r['id']}.jpg", r["label"], r["id"]) for r in rr]
    print(len(old), "foto lama +", len(r27), "frame 27 Okt +", len(r29), "frame 29 Okt", flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        pt = train(old + it(r27) + it(r29), f"{OUT}/tmp/final_v3", f"{OUT}/runs", "final_coca_v3")
        shutil.copy2(pt, f"{M}/merek_coca_v3_kandidat.pt"); print("FINAL ->", f"{M}/merek_coca_v3_kandidat.pt"); return
    meta = {r["botol"]: dict(tutup=r["tutup"], isi=r["isi"]) for r in r29}
    done = {(json.loads(l)["fold"], json.loads(l)["model"]) for l in open(RES)} if os.path.exists(RES) else set()

    def write(fold, model, rr, pt):
        with open(RES, "a") as fh:
            for b, (lab, ses, p) in score(pt, rr).items():
                fh.write(json.dumps(dict(fold=fold, model=model, botol=b, label=lab, p=round(p, 4), **meta.get(b, {}))) + "\n")
    for m, pt in (("m07", f"{M}/merek_coca_kandidat.pt"), ("m08_v2", f"{M}/merek_coca_v2_kandidat.pt")):
        for fold, rr in (("A29", r29), ("B27", r27)):
            if (fold, m) not in done:
                write(fold, m, rr, pt)
    if ("A29", "v3") not in done:
        write("A29", "v3", r29, f"{M}/merek_coca_v2_kandidat.pt")       # = dilatih tanpa 29 Okt
    if ("B27", "v3") not in done:
        pt = train(old + it(r29), f"{OUT}/tmp/b27", f"{OUT}/runs", "v3_b27")
        write("B27", "v3", r27, pt)
    report()


if __name__ == "__main__":
    main()
