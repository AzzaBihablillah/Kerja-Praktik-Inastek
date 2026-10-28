"""Model merek langkah 8: resep A (m07) + frame diam belt 27 Okt (Kiosk Capturer).
Label: sesi1-2 = coca_cola, sesi3-4 = non_coca_cola (label KELOMPOK dari user), sesi5 (air, merek campur) = audit per botol user 28 Okt
(https://claude.ai/artifact/54QAtZEtWudesJDd2FH9GE -> datasets/4_eksperimen/merek_air_audit/hasil_audit.csv).
Potongan = brand_crop dari kotak ver9c terbesar (sama dengan m02/app).
Uji: CV 2 fold. Botol fisik yang sama dipakai di sesi1&2 (Coca) dan sesi3&4 (non) -> dipasangkan; sesi5 dibagi dua per botol.
  fold A uji sesi1+2 + separuh sesi5, fold B uji sesi3+4 + separuh sesi5. 551 foto lama selalu ikut latih.
  Catatan kebocoran: botol sesi5 mungkin botol yang sama dengan sesi1-4 yang diisi air -> angka sesi5 bisa sedikit optimistis.
Keputusan per botol = rata-rata p_coca 3 frame (sama dengan rvm_kp.merek.Merek), ambang 0,5. Pembanding: merek_coca_kandidat (m07).
  python scripts/pipeline_merek/m08_belt27.py         -> datasets/4_eksperimen/merek/cv_belt27.jsonl + cv_belt27_report.txt
  python scripts/pipeline_merek/m08_belt27.py final   -> datasets/3_model_final/merek_coca_v2_kandidat.pt (m07 TIDAK ditimpa)"""
import csv, glob, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m02_cv_dasar import ROOT, OUT, CR, rows, train, brand_crop
import cv2

NAMA = {"mitra": "coca_cola", "bukan": "non_coca_cola"}
T27 = f"{ROOT}/1_foto_mentah/2026-10-27_scan_app"; RES = f"{OUT}/cv_belt27.jsonl"
SES_LAB = {"sesi1": "coca_cola", "sesi2": "coca_cola", "sesi3": "non_coca_cola", "sesi4": "non_coca_cola"}


def rows27():
    p = f"{OUT}/crops27.csv"
    if os.path.exists(p):
        return list(csv.DictReader(open(p)))
    from ultralytics import YOLO
    det = YOLO(f"{ROOT}/3_model_final/botol_ver9c.pt")
    aud = {r["id"]: r["merek_audit"] for r in csv.DictReader(open(f"{ROOT}/4_eksperimen/merek_air_audit/hasil_audit.csv", encoding="utf-8"))}
    out, n5 = [], 0
    for d in sorted(glob.glob(f"{T27}/sesi[1-5]_*")):
        ses = os.path.basename(d).split("_")[0]
        for r in csv.DictReader(open(f"{d}/SCAN.csv")):
            if not r.get("harapan") or r["alasan"] == "bukan_botol":
                continue
            bid = f"S{r['sesi']}B{r['botol']}"
            lab = SES_LAB.get(ses) or aud.get(bid)
            if lab not in ("coca_cola", "non_coca_cola"):
                continue
            if ses == "sesi5":
                fold = n5 % 2; n5 += 1
            else:
                fold = 0 if ses in ("sesi1", "sesi2") else 1
            for k in (1, 2, 3):
                f = f"{d}/s{r['sesi']}_b{r['botol']}_{r['waktu']}_{k}.png"
                if not os.path.exists(f):
                    continue
                im = cv2.imread(f)
                bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
                if not bb:
                    continue
                cid = f"B27_{bid}_{k}"
                cv2.imwrite(f"{CR}/{cid}.jpg", brand_crop(im, *max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))), [cv2.IMWRITE_JPEG_QUALITY, 95])
                out.append(dict(id=cid, botol=bid, sesi=ses, label=lab, fold=fold, produk=r.get("k_produk", "")))
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, list(out[0])); w.writeheader(); w.writerows(out)
    return out


def score(pt, rr):
    from ultralytics import YOLO
    m = YOLO(pt); ci = [i for i, v in m.names.items() if v == "coca_cola"][0]; P = {}
    for s in range(0, len(rr), 64):
        ch = rr[s:s + 64]
        for r, p in zip(ch, m.predict([f"{CR}/{r['id']}.jpg" for r in ch], imgsz=256, verbose=False)):
            P.setdefault(r["botol"], [r["label"], r["sesi"], []])[2].append(float(p.probs.data[ci]))
    return {b: (lab, ses, sum(ps) / len(ps)) for b, (lab, ses, ps) in P.items()}


def report():
    J = [json.loads(l) for l in open(RES)]
    L = ["Merek + belt 27 Okt — CV 2 fold (A: sesi1+2 + 1/2 sesi5, B: sesi3+4 + 1/2 sesi5), per botol = rata-rata 3 frame, ambang 0,5.", ""]
    for m in ("m07_lama", "m08_v2"):
        jj = [j for j in J if j["model"] == m]
        co = [j for j in jj if j["label"] == "coca_cola"]; no = [j for j in jj if j["label"] == "non_coca_cola"]
        L.append(f"{m:9s} benar {sum((j['p'] >= .5) == (j['label'] == 'coca_cola') for j in jj)}/{len(jj)} | Coca-Cola dikenali {sum(j['p'] >= .5 for j in co)}/{len(co)}"
                 f" | non-Coca salah dapat bonus {sum(j['p'] >= .5 for j in no)}/{len(no)}   per sesi: "
                 + ", ".join(f"{s} {sum((j['p'] >= .5) == (j['label'] == 'coca_cola') for j in jj if j['sesi'] == s)}/{sum(j['sesi'] == s for j in jj)}"
                             for s in sorted({j['sesi'] for j in jj})))
        L.append("          salah: " + ", ".join(f"{j['botol']}({j['label'][:4]} p={j['p']:.2f})" for j in jj if (j['p'] >= .5) != (j['label'] == 'coca_cola')))
    open(f"{OUT}/cv_belt27_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


def main():
    old = [(f"{CR}/{r['id']}.jpg", NAMA[r["kelas"]], r["id"]) for r in rows() if os.path.exists(f"{CR}/{r['id']}.jpg")]
    new = rows27()
    print(len(old), "foto lama +", len(new), "frame 27 Okt (", len({r['botol'] for r in new}), "botol )", flush=True)
    it = lambda rr: [(f"{CR}/{r['id']}.jpg", r["label"], r["id"]) for r in rr]
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        pt = train(old + it(new), f"{OUT}/tmp/final_v2", f"{OUT}/runs", "final_coca_v2")
        dst = f"{ROOT}/3_model_final/merek_coca_v2_kandidat.pt"; shutil.copy2(pt, dst); print("FINAL ->", dst); return
    done = {(json.loads(l)["fold"], json.loads(l)["model"]) for l in open(RES)} if os.path.exists(RES) else set()
    for k in (0, 1):
        te = [r for r in new if int(r["fold"]) == k]
        with open(RES, "a") as fh:
            if (k, "m07_lama") not in done:
                for b, (lab, ses, p) in score(f"{ROOT}/3_model_final/merek_coca_kandidat.pt", te).items():
                    fh.write(json.dumps(dict(fold=k, model="m07_lama", botol=b, sesi=ses, label=lab, p=round(p, 4))) + "\n")
        if (k, "m08_v2") in done:
            continue
        pt = train(old + it([r for r in new if int(r["fold"]) != k]), f"{OUT}/tmp/b27_f{k}", f"{OUT}/runs", f"b27_f{k}")
        with open(RES, "a") as fh:
            for b, (lab, ses, p) in score(pt, te).items():
                fh.write(json.dumps(dict(fold=k, model="m08_v2", botol=b, sesi=ses, label=lab, p=round(p, 4))) + "\n")
        print(f"fold {k}: uji {len({r['botol'] for r in te})} botol", flush=True)
    report()


if __name__ == "__main__":
    main()
