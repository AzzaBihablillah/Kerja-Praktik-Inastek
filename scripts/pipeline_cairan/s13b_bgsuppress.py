"""Langkah 13b: P3 = supresi latar (domain-invariant). Piksel yang warnanya (Lab) dekat dengan warna belt di sekitar botol
-> diganti abu 114; tersisa hanya yang BERBEDA dari latar (cairan berwarna, label, tutup, pantulan).
Alasan: percobaan G/S menunjukkan model memakai 'domain' (belt vs web) sebagai jalan pintas; supresi latar menghilangkan cue itu
dan secara fisik sesuai target ('cairan berwarna beda dari belt', botol kosong bening ~ warna belt).
Referensi latar: belt = median Lab pita di atas/bawah kotak botol; (gambar web: median piksel tepi crop).
Keluaran: out/crops_fold_P3/<id>.jpg (nama sama dgn crops_fold)  [--thr 14]"""
import csv, os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fold_crop
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = os.path.join(ROOT, "4_eksperimen", "cairan"); DST = f"{OUT}/crops_fold_P3"; os.makedirs(DST, exist_ok=True)
THR = float(sys.argv[sys.argv.index("--thr") + 1]) if "--thr" in sys.argv else 14.0


def suppress(im, ref_lab, thr=THR):
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.linalg.norm(lab - ref_lab[None, None, :], axis=2)
    m = (d < thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    out = im.copy(); out[m > 0] = 114
    return out


rows = list(csv.DictReader(open(f"{OUT}/belt_index.csv")))
for i, r in enumerate(rows):
    im = cv2.imread(os.path.join(ROOT, r["path"])); x1, y1, x2, y2 = [int(float(r[k])) for k in ("x1", "y1", "x2", "y2")]; h = y2 - y1
    lab = cv2.cvtColor(im, cv2.COLOR_BGR2LAB).astype(np.float32)
    bg = np.concatenate([lab[max(0, y1 - int(h * .5)):max(1, y1 - 8), x1:x2].reshape(-1, 3), lab[min(im.shape[0] - 1, y2 + 8):y2 + int(h * .5), x1:x2].reshape(-1, 3)])
    ref = np.median(bg, axis=0) if len(bg) > 50 else np.array([70, 110, 125], np.float32)
    cv2.imwrite(f"{DST}/{r['src']}_{i:04d}.jpg", fold_crop(suppress(im, ref), x1, y1, x2, y2), [cv2.IMWRITE_JPEG_QUALITY, 95])
print(len(rows), "crop P3 (belt) thr", THR)
