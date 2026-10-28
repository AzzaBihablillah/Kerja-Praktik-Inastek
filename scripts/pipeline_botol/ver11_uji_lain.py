"""Uji BIAS ver11 di data LAIN yang tidak dilatih (permintaan user 27 Okt: jangan sampai 'non-botol = plastik kresek').
  1. TrashNet DITAHAN (nomor % 5 == 0) ditempel ke foto belt 20 Okt (sama dengan uji_trashnet.py): per kategori
     (kardus, kaca, logam/kaleng, kertas, plastik lain, sampah, + PET kontrol) -> DITERIMA sebagai botol & DITANDAI non_bottle.
  2. TACO uji + sesi 20 Okt + old_test (ver9_nonbottle.py uji) dan cek jalan pintas botol tempelan (cek_jalan_pintas.py).
  python scripts/pipeline_botol/ver11_uji_lain.py [model.pt ...]   (bawaan: ver9c, ver10, ver11)
  -> datasets/4_eksperimen/botol/ver11/uji_lain.txt"""
import os, subprocess, sys
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from uji_trashnet import build, OUT as TN
from ver9_nonbottle import iou

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DS = os.path.join(REPO, "datasets"); M = f"{DS}/3_model_final"; OUT = f"{DS}/4_eksperimen/botol/ver11"
PTS = sys.argv[1:] or [f"{M}/botol_ver9c.pt", f"{M}/botol_ver10_kandidat.pt", f"{M}/botol_ver11_kandidat.pt"]


def trashnet(pts):
    from ultralytics import YOLO
    meta = build(); cats = sorted({m["cat"] for m in meta})
    L = [f"1. TrashNet ditahan, {len(meta)} tempelan. Sel = DITERIMA botol / DITANDAI non_bottle / jumlah (IoU>=0,3 dengan objek tempelan).",
         f"{'model':22s} " + " ".join(f"{c.replace('trashnet_', ''):>15s}" for c in cats) + "   non-PET: diterima | ditandai"]
    for pt in pts:
        m = YOLO(pt); acc = defaultdict(lambda: [0, 0, 0])
        for mm in meta:
            d = [(int(b.cls), b.xyxy[0].tolist()) for b in m.predict(f"{TN}/{mm['name']}.jpg", imgsz=640, conf=0.35, verbose=False)[0].boxes]
            a = acc[mm["cat"]]; a[0] += 1
            a[1] += any(c == 0 and iou(x, mm["pasted"]) >= 0.3 for c, x in d)
            a[2] += any(c == 1 and iou(x, mm["pasted"]) >= 0.3 for c, x in d)
        nn = [acc[c] for c in cats if not c.endswith("_PET")]; n = sum(x[0] for x in nn)
        L.append(f"{os.path.basename(pt)[:-3]:22s} " + " ".join(f"{acc[c][1]:>4d}/{acc[c][2]:>3d}/{acc[c][0]:<5d}" for c in cats) +
                 f"   {sum(x[1] for x in nn)}/{n} ({100 * sum(x[1] for x in nn) / n:.0f}%) | {sum(x[2] for x in nn)}/{n} ({100 * sum(x[2] for x in nn) / n:.0f}%)")
        print(L[-1], flush=True)
    return L


def main():
    L = ["Uji bias ver11 di data lain (tak dilatih). Ambang 0,35.", ""] + trashnet(PTS) + [""]
    py = sys.executable; env = dict(os.environ, YOLO_AUTOINSTALL="False", PYTHONUTF8="1")
    subprocess.run([py, os.path.join(REPO, "scripts", "pipeline_botol", "ver9_nonbottle.py"), "uji"] + PTS, env=env, cwd=REPO, check=False)
    for pt in PTS:
        tag = os.path.basename(pt)[:-3]; f = f"{DS}/4_eksperimen/botol/hasil_uji_{tag}.txt"
        if os.path.exists(f):
            L += ["2. " + tag + " — TACO / sesi 20 Okt / old_test:"] + ["   " + l for l in open(f, encoding="utf-8").read().splitlines()[1:]]
    r = subprocess.run([py, os.path.join(REPO, "scripts", "pipeline_botol", "cek_jalan_pintas.py")] + PTS, env=env, cwd=REPO,
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    L += ["", "3. Cek jalan pintas (botol ASLI ditempel; harus tetap diterima, tidak dicap non_bottle):"] + \
         ["   " + l for l in r.stdout.splitlines() if "botol tempelan" in l]
    open(f"{OUT}/uji_lain.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
