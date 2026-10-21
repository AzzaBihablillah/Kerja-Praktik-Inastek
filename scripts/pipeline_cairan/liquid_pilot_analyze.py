"""Pilot deteksi isi cairan — langkah 2: fitur, kontrol kebocoran, uji sinyal.

PILOT: tiap kondisi = 1 botol fisik diputar -> TIDAK membuktikan generalisasi, hanya "ada sinyal atau tidak".
Kontrol:
  K0  sesi/pencahayaan  : bisakah KONDISI dibedakan hanya dari area belt statis (tanpa botol)?  (kalau ya -> confound)
  K1  label             : ROI hanya bagian botol BEBAS LABEL (dasar & bahu), bukan badan berlabel.
  K2  permutasi         : akurasi LOO dibandingkan dgn label diacak (p-value).
Uji:
  T1  air putih vs kosong (botol Coke sama)         : AUC per fitur + logistic L2 LOO
  T2  lintas-botol                                  : latih di Coke{air,kosong}, prediksi air di Fanta/LeMinerale/Prima
  T3  keruh vs bening, leave-one-KONDISI-out        : tiap kondisi ditahan utuh
Output: liquid_pilot/{features.csv,report.txt,compare.jpg}
"""
import csv, os
import cv2, numpy as np
from scipy.optimize import minimize
from ultralytics import YOLO

DS = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets"
P = f"{DS}/4_eksperimen/cairan_pilot"
BOTTLE = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/export_model/bottle_best_ver7.pt"
RNG = np.random.default_rng(0)
LOG = []


def log(s=""):
    print(s, flush=True); LOG.append(s)


def feats(roi_bgr, pre):
    g = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    H, S, V = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    rows = V.mean(1); cols = V.mean(0)
    ed = cv2.Canny(g, 60, 140)
    hue = np.angle(np.sum((S / 255.0) * np.exp(1j * np.deg2rad(H * 2))))
    return {f"{pre}_V": V.mean(), f"{pre}_Vsd": V.std(), f"{pre}_S": S.mean(),
            f"{pre}_huecos": np.cos(hue) * (S.mean() / 255.0), f"{pre}_huesin": np.sin(hue) * (S.mean() / 255.0),
            f"{pre}_edge": (ed > 0).mean(), f"{pre}_lap": float(np.log1p(cv2.Laplacian(g, cv2.CV_64F).var())),
            f"{pre}_dark": (V < 50).mean(), f"{pre}_bright": (V > 200).mean(),
            f"{pre}_rowrng": rows.max() - rows.min(), f"{pre}_colsd": cols.std()}


def load():
    rows = list(csv.DictReader(open(f"{P}/index.csv")))
    det = None
    out = []
    for r in rows:
        im = cv2.imread(f"{P}/raw/{r['file']}.png")
        x1, y1, x2, y2 = (int(r[k]) for k in ("x1", "y1", "x2", "y2"))
        if not int(r["det"]):
            det = det or YOLO(BOTTLE)
            res = det.predict(im, imgsz=640, conf=0.05, verbose=False)[0]
            bb = [b for b in res.boxes if int(b.cls) == 0]
            if not bb:
                log(f"  [dilewati: botol tak terdeteksi walau conf 0.05] {r['file']}"); continue
            b = max(bb, key=lambda z: float(z.conf)); x1, y1, x2, y2 = [int(v) for v in b.xyxy[0]]
            log(f"  [deteksi ulang conf 0.05] {r['file']} conf {float(b.conf):.2f}")
        x1, y1 = max(0, x1), max(0, y1); w, h = x2 - x1, y2 - y1
        base = im[y1 + int(.22 * h):y1 + int(.78 * h), x1 + int(.64 * w):x1 + int(.97 * w)]
        shoulder = im[y1 + int(.22 * h):y1 + int(.78 * h), x1 + int(.10 * w):x1 + int(.28 * w)]
        bg = im[70:96, 20:620]                                    # sabuk hijau statis (tanpa botol)
        f = {"file": r["file"], "cond": r["cond"], "brand": r["brand"], "fill": r["fill"]}
        f.update(feats(base, "base")); f.update(feats(shoulder, "sh"))
        bgf = feats(bg, "bg"); f.update({k: v for k, v in bgf.items() if k in ("bg_V", "bg_S", "bg_Vsd")})
        f["base_Vrel"] = f["base_V"] / (f["bg_V"] + 1e-6); f["sh_Vrel"] = f["sh_V"] / (f["bg_V"] + 1e-6)
        out.append(f)
    return out


def mat(rows, cols):
    return np.array([[r[c] for c in cols] for r in rows], dtype=np.float64)


def auc(a, b):                                                   # P(a > b)
    ra = np.concatenate([a, b]); order = ra.argsort().argsort() + 1.0
    # rata-rata rank utk ties
    _, inv, cnt = np.unique(ra, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=order); order = (sums / cnt)[inv]
    u = order[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return u / (len(a) * len(b))


def boot_auc(a, b, n=1000):
    v = [auc(RNG.choice(a, len(a)), RNG.choice(b, len(b))) for _ in range(n)]
    return np.percentile(v, [2.5, 97.5])


def fit_lr(X, y, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd
    def nll(w):
        z = Z @ w[:-1] + w[-1]; return np.sum(np.logaddexp(0, z) - y * z) + lam * np.sum(w[:-1] ** 2)
    def grad(w):
        z = Z @ w[:-1] + w[-1]; p = 1 / (1 + np.exp(-z)); g = np.zeros_like(w)
        g[:-1] = Z.T @ (p - y) + 2 * lam * w[:-1]; g[-1] = (p - y).sum(); return g
    w = minimize(nll, np.zeros(Z.shape[1] + 1), jac=grad, method="L-BFGS-B").x
    return lambda Xn: 1 / (1 + np.exp(-(((Xn - mu) / sd) @ w[:-1] + w[-1])))


def loo_acc(X, y, lam=1.0):
    ok = 0
    for i in range(len(y)):
        m = np.ones(len(y), bool); m[i] = False
        ok += int((fit_lr(X[m], y[m], lam)(X[i:i + 1])[0] > 0.5) == bool(y[i]))
    return ok / len(y)


def perm_p(X, y, acc, n=200):
    c = sum(loo_acc(X, RNG.permutation(y)) >= acc for _ in range(n))
    return (c + 1) / (n + 1)


def main():
    log("== PILOT isi cairan: persiapan =="); rows = load()
    cs = lambda c: [r for r in rows if r["cond"] == c]
    log(f"foto dianalisis: {len(rows)} | " + ", ".join(f"{c}={len(cs(c))}" for c in sorted({r['cond'] for r in rows})))
    with open(f"{P}/features.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    FB = [k for k in rows[0] if k.startswith("base_") and k != "base_V"] + ["base_V"]
    FS = [k for k in rows[0] if k.startswith("sh_")]
    FALL = [k for k in rows[0] if k.startswith(("base_", "sh_"))]
    BGF = ["bg_V", "bg_S", "bg_Vsd"]

    # ---- K0: sesi
    log("\n== K0  Kontrol sesi/pencahayaan (area belt statis, TANPA botol) ==")
    A, B = cs("C2"), cs("C3")
    for k in BGF:
        a, b = np.array([r[k] for r in A]), np.array([r[k] for r in B])
        d = (a.mean() - b.mean()) / (np.sqrt((a.var() + b.var()) / 2) + 1e-9)
        log(f"  {k:7s} air(C2) {a.mean():6.1f} vs kosong(C3) {b.mean():6.1f}  | Cohen d {d:+.2f}  AUC {auc(a, b):.2f}")
    X0 = mat(A + B, BGF); y0 = np.array([1] * len(A) + [0] * len(B))
    acc0 = loo_acc(X0, y0)
    log(f"  classifier HANYA dari belt statis, LOO: {100*acc0:.0f}%  (kalau tinggi -> sesi berbeda = confound; abaikan fitur absolut)")

    # ---- T1
    log("\n== T1  Air putih (C2) vs Kosong (C3) — botol Coke, ROI bebas-label ==")
    log("  fitur             AUC (air>kosong)  95% CI        arah")
    res = []
    for k in FALL:
        a, b = np.array([r[k] for r in A]), np.array([r[k] for r in B])
        ci = boot_auc(a, b, 400); res.append((abs(auc(a, b) - .5), k, auc(a, b), ci))
    for _, k, au, ci in sorted(res, reverse=True)[:8]:
        log(f"  {k:16s}  {au:5.2f}           [{ci[0]:.2f},{ci[1]:.2f}]   {'air lebih tinggi' if au > .5 else 'air lebih rendah'}")
    Xt = mat(A + B, FALL); yt = np.array([1] * len(A) + [0] * len(B))
    acc = loo_acc(Xt, yt); p = perm_p(Xt, yt, acc)
    log(f"  logistic L2 (semua {len(FALL)} fitur), leave-one-pose-out: {100*acc:.0f}% ({int(round(acc*len(yt)))}/{len(yt)}) | p-permutasi {p:.3f}")
    Xn = mat(A + B, [k for k in FALL if "rel" not in k and k not in ("base_V", "sh_V")]); acc_n = loo_acc(Xn, yt)
    log(f"  tanpa fitur kecerahan absolut (V): {100*acc_n:.0f}%")
    accb = loo_acc(mat(A + B, FB), yt); accs = loo_acc(mat(A + B, FS), yt)
    log(f"  hanya ROI dasar: {100*accb:.0f}% | hanya ROI bahu: {100*accs:.0f}%")

    # ---- T2
    log("\n== T2  Lintas-botol: latih di Coke{air,kosong}, prediksi botol LAIN berisi air ==")
    clf = fit_lr(Xt, yt)
    for c, nm in (("C5", "Fanta+air"), ("C7", "LeMinerale+air"), ("C8", "Prima+air")):
        Xc = mat(cs(c), FALL); pr = clf(Xc)
        log(f"  {nm:15s}: diprediksi 'air' {int((pr > .5).sum())}/{len(pr)}  (rata-rata p(air) {pr.mean():.2f})")
    log("  (tidak ada pasangan 'kosong' utk botol-botol ini -> hanya recall 'air'; bisa menipu kalau model cuma bilang 'air' utk semua yg bukan Coke-kosong)")

    # ---- T3
    log("\n== T3  Keruh (berwarna) vs bening, leave-one-KONDISI-out ==")
    colored = {"C1", "C4", "C6"}; clear = {"C2", "C3", "C5", "C7", "C8"}
    FC = [k for k in FALL if "rel" not in k]
    for hold in sorted(colored | clear):
        tr = [r for r in rows if r["cond"] not in (hold,)]; te = cs(hold)
        ytr = np.array([1 if r["cond"] in colored else 0 for r in tr]); yte = 1 if hold in colored else 0
        pr = fit_lr(mat(tr, FC), ytr)(mat(te, FC))
        log(f"  tahan {hold} ({'berwarna' if hold in colored else 'bening'}): benar {int(((pr > .5) == bool(yte)).sum())}/{len(te)}")

    # ---- gambar pembanding
    cmp_rows = []
    for c, nm in (("C3", "Coke KOSONG"), ("C2", "Coke AIR PUTIH"), ("C1", "Coke COLA")):
        ts = []
        for r in cs(c)[:5]:
            im = cv2.imread(f"{P}/crops/{r['file'].replace('/', '__')}.png")
            if im is None: continue
            ts.append(cv2.resize(im, (260, 110)))
        if ts:
            row = np.hstack(ts); cv2.putText(row, nm, (6, 18), 0, 0.6, (0, 255, 255), 2); cmp_rows.append(row)
    if cmp_rows:
        wmax = max(r.shape[1] for r in cmp_rows)
        cmp_rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in cmp_rows]
        cv2.imwrite(f"{P}/compare.jpg", np.vstack(cmp_rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    open(f"{P}/report.txt", "w", encoding="utf-8").write("\n".join(LOG))
    log("\nlaporan: liquid_pilot/report.txt")


if __name__ == "__main__":
    main()
