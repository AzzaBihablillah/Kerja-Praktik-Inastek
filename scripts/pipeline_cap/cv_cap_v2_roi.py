"""Setelan mode ROI (area botol) untuk model tutup v2, diuji ulang memakai model CV per fold yang sudah dilatih (tanpa latih ulang).
Pertanyaan: kenapa ROI (profil nano) kalah dari full-frame pada tutup hitam? Variabel: kanvas, skala, ambang conf, margin.
  python scripts/pipeline_cap/cv_cap_v2_roi.py   -> datasets/4_eksperimen/cap/_cv_cap_v2/roi_setelan.txt"""
import json, os, sys
import cv2
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cv_cap_v2 import CV, items, folds, HITAM
from rvm_kp.analyzer import roi_canvas
from ultralytics import YOLO

SETELAN = [("320 s1/3 conf.25 m.08 (sekarang)", 320, 1 / 3, 0.25, 0.08), ("320 s1/3 conf.03", 320, 1 / 3, 0.03, 0.08),
           ("320 s1/3 conf.03 m.15", 320, 1 / 3, 0.03, 0.15), ("416 s1/2 conf.25", 416, 1 / 2, 0.25, 0.08),
           ("416 s1/2 conf.03", 416, 1 / 2, 0.03, 0.08), ("416 s1/3 conf.03", 416, 1 / 3, 0.03, 0.08),
           ("480 s1/2 conf.03 m.15", 480, 1 / 2, 0.03, 0.15)]


def main():
    its = items(); F, _ = folds(its); boxes = json.load(open(f"{CV}/bottle_boxes.json"))
    res = {s[0]: [] for s in SETELAN}
    for k in range(5):
        m = YOLO(f"{CV}/runs/f{k}_gabung/weights/last.pt")
        for it in [i for i in its if F[i["merek"]] == k]:
            im = cv2.imread(it["img"]); W = im.shape[1]; bx = boxes.get(it["name"])
            for name, sz, sc, conf, mg in SETELAN:
                pred = -1
                if bx:
                    cv, _ = roi_canvas(im, *bx, sz, sc * 1920 / W, mg)
                    r = m.predict(cv, imgsz=sz, conf=conf, verbose=False)[0]
                    pred = int(max(r.boxes, key=lambda z: float(z.conf)).cls) if len(r.boxes) else -1
                res[name].append((it, pred))
    L = ["Mode ROI, model 'gabung' per fold (CV per merek), 397 foto. Kolom: semua | tutup hitam | Powerade | Coke Zero | 13 Okt | tanpa deteksi"]
    for name, rr in res.items():
        f = lambda cond: (sum(p == it["cls"] for it, p in rr if cond(it)), sum(1 for it, _ in rr if cond(it)))
        a, h, pw, cz, s13 = f(lambda i: True), f(lambda i: i["merek"] in HITAM), f(lambda i: i["merek"] == "powerade"), \
            f(lambda i: i["merek"] == "cokezero"), f(lambda i: i["sesi"] == "1013")
        L.append(f"  {name:34s} {a[0]}/{a[1]} | {h[0]}/{h[1]} | {pw[0]}/{pw[1]} | {cz[0]}/{cz[1]} | {s13[0]}/{s13[1]} | {sum(p == -1 for _, p in rr)}")
    open(f"{CV}/roi_setelan.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
