"""Audit label tutup di nama file (capon/capoff, cap/nocap) dengan model tutup cap_belt_v2 -> cari foto yang tertukar.
  python scripts/alat/audit_tutup.py --dir 1_foto_mentah/2026-10-22_water_v1
Per foto: kotak botol (botol_ver8b) -> tutup di area botol (setelan rvm_kp: kanvas 416, skala 1/2, conf 0,03) DAN full-frame 640.
CURIGA = kedua cara sepakat berlawanan dengan nama file (yakin), PERIKSA = hanya satu cara berlawanan / tidak terdeteksi.
Keluaran di folder itu: AUDIT-TUTUP.csv + AUDIT-TUTUP.jpg (potongan leher foto CURIGA/PERIKSA). Foto TIDAK diubah.
Catatan: untuk foto yang ikut melatih cap_belt_v2 (13/16/20 Okt) audit ini optimistis; pakai hasil CV (cv_cap_v2) untuk sesi itu."""
import argparse, csv, glob, os, re, sys
import cv2, numpy as np
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from rvm_kp import config
from rvm_kp.analyzer import roi_canvas
from ultralytics import YOLO

ap = argparse.ArgumentParser(); ap.add_argument("--dir", required=True); a = ap.parse_args()
D = a.dir if os.path.isabs(a.dir) else os.path.join(REPO, "datasets", a.dir)
det = YOLO(f"{config.M}/botol_ver8b.pt"); cap = YOLO(f"{config.M}/cap_belt_v2.pt")
NAMA = ["cap", "no_cap"]
rows, flag = [], []
for f in sorted(glob.glob(f"{D}/*.png")):
    n = os.path.basename(f)
    m = re.search(r"_(capon|capoff|cap|nocap)_\d+\.png$", n)
    if not m:
        continue
    lab = 0 if m[1] in ("capon", "cap") else 1
    im = cv2.imread(f); W = im.shape[1]
    r = cap.predict(im, imgsz=640, conf=0.03, verbose=False)[0]
    full = (int(max(r.boxes, key=lambda z: float(z.conf)).cls), float(max(r.boxes, key=lambda z: float(z.conf)).conf)) if len(r.boxes) else (-1, 0.0)
    bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.35, verbose=False)[0].boxes if int(b.cls) == 0]
    roi, cbox = (-1, 0.0), None
    if bb:
        bx = max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        cv, to_frame = roi_canvas(im, *bx, config.CAP_ROI_IMGSZ, config.CAP_ROI_SCALE * 1920 / W, config.CAP_ROI_MARGIN)
        rr = cap.predict(cv, imgsz=config.CAP_ROI_IMGSZ, conf=config.CONF_CAP_ROI, verbose=False)[0]
        if len(rr.boxes):
            b = max(rr.boxes, key=lambda z: float(z.conf)); roi = (int(b.cls), float(b.conf)); cbox = to_frame([float(v) for v in b.xyxy[0]])
    lawan = [p[0] == 1 - lab for p in (full, roi)]
    status = "CURIGA" if all(lawan) else ("PERIKSA" if any(lawan) or -1 in (full[0], roi[0]) else "ok")
    rows.append(dict(file=n, label=NAMA[lab], full=NAMA[full[0]] if full[0] >= 0 else "-", full_conf=round(full[1], 2),
                     roi=NAMA[roi[0]] if roi[0] >= 0 else "-", roi_conf=round(roi[1], 2), status=status))
    if status != "ok":
        if cbox is None and bb:
            cbox = bx
        cx, cy = (int((cbox[0] + cbox[2]) / 2), int((cbox[1] + cbox[3]) / 2)) if cbox else (W // 2, im.shape[0] // 2)
        c = im[max(0, cy - 160):cy + 160, max(0, cx - 160):cx + 160].copy()
        c = cv2.copyMakeBorder(c, 0, 320 - c.shape[0], 0, 320 - c.shape[1], cv2.BORDER_CONSTANT)
        cv2.rectangle(c, (0, 0), (320, 44), (0, 0, 0), -1)
        cv2.putText(c, n[:40], (4, 16), 0, 0.42, (0, 255, 255), 1)
        cv2.putText(c, f"{status}: label {NAMA[lab]} | model {rows[-1]['roi']} {roi[1]:.2f}", (4, 36), 0, 0.45,
                    (0, 0, 255) if status == "CURIGA" else (0, 200, 255), 1)
        flag.append(c)
if not rows:
    sys.exit("tidak ada foto capon/capoff di " + D)
with open(f"{D}/AUDIT-TUTUP.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
if flag:
    while len(flag) % 5:
        flag.append(np.zeros_like(flag[0]))
    cv2.imwrite(f"{D}/AUDIT-TUTUP.jpg", np.vstack([np.hstack(flag[i:i + 5]) for i in range(0, len(flag), 5)]))
k = lambda s: sum(r["status"] == s for r in rows)
print(f"{len(rows)} foto | ok {k('ok')} | CURIGA tertukar {k('CURIGA')} | PERIKSA {k('PERIKSA')}")
for r in rows:
    if r["status"] != "ok":
        print(f"  {r['status']:8s} {r['file']}  label={r['label']}  full={r['full']}({r['full_conf']})  roi={r['roi']}({r['roi_conf']})")
