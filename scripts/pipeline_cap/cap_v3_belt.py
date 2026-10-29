"""Model tutup v3: data cap_belt_v2 (397 foto, Roboflow v2) + frame diam belt 27 & 29 Okt (Kiosk Capturer).
Kotak leher/tutup = usulan cap_belt_v2 (full frame, imgsz 960) per frame, DIAUDIT user 29 Okt (letak kotak):
  27 Okt https://claude.ai/artifact/HJwtBqRWifax2MUhzQCHZR (0 ditandai; 1 tanpa kotak tidak dipakai)
  29 Okt https://claude.ai/artifact/S9o16vTCGXZ5fim1Rpahz4 (3 foto blur ditandai BUANG, lalu user: kotaknya pas -> TETAP dipakai)
Kelas DARI LABEL SESI (bertutup -> cap, tanpa -> no_cap), bukan dari model. 29 Okt termasuk tutup yang sengaja ditukar warnanya.
Validasi PER HARI (botol & sesi yang tidak dilatih), dinilai seperti aplikasi: mode ROI profil nano (kotak botol ver9c, kanvas 320)
dan full-frame 640; per botol = suara mayoritas 3 frame (tanpa deteksi = salah):
  A29: latih lama + 27 Okt -> uji 29 Okt        B27: latih lama + 29 Okt -> uji 27 Okt
Pembanding: cap_belt_v2 (aktif) pada botol yang sama.
  python scripts/pipeline_cap/cap_v3_belt.py         -> datasets/4_eksperimen/cap/_cap_v3/ (cv_records.jsonl, cv_report.txt)
  python scripts/pipeline_cap/cap_v3_belt.py final   -> datasets/3_model_final/cap_belt_v3_kandidat.pt (cap_belt_v2 TIDAK ditimpa)"""
import csv, json, os, shutil, sys
from collections import Counter
from multiprocessing import freeze_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_cap_v2 as v2

REPO, DS = v2.REPO, v2.DS
OUT = f"{DS}/4_eksperimen/cap/_cap_v3"; v2.CV = OUT
AUD = {"27": f"{DS}/4_eksperimen/cap_audit_27okt/usulan.csv", "29": f"{DS}/4_eksperimen/cap_audit_29okt/usulan.csv"}
TOLAK = set()            # audit: tidak ada kotak salah; foto blur 29 Okt tetap dipakai (keputusan user)


def new_items():
    """Satu item per frame berkotak: gambar asli (png) + label YOLO dari kelas sesi."""
    import cv2
    os.makedirs(f"{OUT}/labels_baru", exist_ok=True); its = []
    for hari, p in AUD.items():
        for r in csv.DictReader(open(p, encoding="utf-8")):
            if r["sek"] == "nol" or r["id"] in TOLAK:
                continue
            cls = 0 if r["kelas"] == "cap" else 1
            for k, (f, box) in enumerate(json.loads(r["kotak"]), 1):
                if not box:
                    continue
                img = os.path.join(REPO, f); H, W = cv2.imread(img).shape[:2]
                x1, y1, x2, y2 = box; name = f"b{hari}_{r['id']}_{k}"
                lp = f"{OUT}/labels_baru/{name}.txt"
                open(lp, "w").write(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
                its.append(dict(name=name, img=img, lbl=lp, sesi=r["sesi"], merek="belt" + hari, cls=cls, hari=hari, botol=r["id"]))
    return its


def build(d, train_its):
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(f"{d}/images/train"); os.makedirs(f"{d}/labels/train")
    for it in train_its:
        ext = os.path.splitext(it["img"])[1]
        v2.link(it["img"], f"{d}/images/train/{it['name']}{ext}"); shutil.copy2(it["lbl"], f"{d}/labels/train/{it['name']}.txt")
    open(f"{d}/data.yaml", "w").write(f"path: {d}\ntrain: images/train\nval: images/train\nnames:\n  0: cap\n  1: no_cap\n")


def evaluate(pt, test_its, tag, fold):
    import cv2
    from ultralytics import YOLO
    from rvm_kp.analyzer import roi_canvas
    m = YOLO(pt); det = YOLO(f"{DS}/3_model_final/botol_ver9c.pt"); per = {}
    for it in test_its:
        im = cv2.imread(it["img"]); W = im.shape[1]
        r = m.predict(im, imgsz=640, conf=0.03, max_det=10, verbose=False)[0]
        full = int(max(r.boxes, key=lambda z: float(z.conf)).cls) if len(r.boxes) else -1
        bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
        roi = -1
        if bb:
            bx = max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            cv, _ = roi_canvas(im, *bx, 320, (1 / 3) * 1920 / W, 0.08)
            rr = m.predict(cv, imgsz=320, conf=0.25, verbose=False)[0]
            roi = int(max(rr.boxes, key=lambda z: float(z.conf)).cls) if len(rr.boxes) else -1
        p = per.setdefault(it["botol"], dict(fold=fold, model=tag, botol=it["botol"], hari=it["hari"], sesi=it["sesi"], cls=it["cls"], full=[], roi=[]))
        p["full"].append(full); p["roi"].append(roi)
    out = []
    for p in per.values():
        for mode in ("full", "roi"):
            c = Counter(p[mode]).most_common(1)[0][0]
            p[mode + "_ok"] = c == p["cls"]
        out.append(p)
    return out


def report():
    R = [json.loads(l) for l in open(f"{OUT}/cv_records.jsonl")]
    L = ["Model tutup v3 — validasi PER HARI, per botol suara mayoritas 3 frame.", ""]
    for fold, nm in (("A29", "uji 29 Okt (termasuk tutup ditukar)"), ("B27", "uji 27 Okt")):
        L.append(f"== {nm}")
        for mode in ("roi", "full"):
            for m in ("v2_aktif", "v3"):
                rr = [r for r in R if r["fold"] == fold and r["model"] == m]
                if not rr:
                    continue
                cap = [r for r in rr if r["cls"] == 0]; nc = [r for r in rr if r["cls"] == 1]
                L.append(f"  {mode:4s} {m:8s} bertutup benar {sum(r[mode + '_ok'] for r in cap)}/{len(cap)} | tanpa tutup benar "
                         f"{sum(r[mode + '_ok'] for r in nc)}/{len(nc)}")
        L.append("")
    open(f"{OUT}/cv_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L), flush=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    old = v2.items(); new = new_items()
    n27 = [i for i in new if i["hari"] == "27"]; n29 = [i for i in new if i["hari"] == "29"]
    print(len(old), "foto lama +", len(n27), "frame 27 Okt +", len(n29), "frame 29 Okt", flush=True)
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        d = f"{OUT}/data_final"; build(d, old + new); pt = v2.train(d, "final")
        shutil.copy2(pt, f"{DS}/3_model_final/cap_belt_v3_kandidat.pt"); shutil.rmtree(d, ignore_errors=True)
        print("FINAL ->", f"{DS}/3_model_final/cap_belt_v3_kandidat.pt"); return
    res = f"{OUT}/cv_records.jsonl"
    done = {(json.loads(l)["fold"], json.loads(l)["model"]) for l in open(res)} if os.path.exists(res) else set()
    for fold, test, tr in (("A29", n29, old + n27), ("B27", n27, old + n29)):
        recs = []
        if (fold, "v2_aktif") not in done:
            recs += evaluate(f"{DS}/3_model_final/cap_belt_v2.pt", test, "v2_aktif", fold)
        if (fold, "v3") not in done:
            d = f"{OUT}/data_{fold}"; build(d, tr)
            print(f"{fold}: latih {len(tr)}, uji {len(test)} frame", flush=True)
            recs += evaluate(v2.train(d, f"cv_{fold}"), test, "v3", fold); shutil.rmtree(d, ignore_errors=True)
        with open(res, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
    report()


if __name__ == "__main__":
    freeze_support()
    main()
