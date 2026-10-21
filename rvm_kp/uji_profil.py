"""Uji profil perangkat di foto belt nyata (bahan TA): apakah batasan Jetson Nano mengubah keputusan sistem?
  python -m rvm_kp.uji_profil            (laptop, PyTorch GPU; ±5 menit)
Faktor yang diuji (satu per satu terhadap acuan = profil laptop: 1080p, tutup full-frame, FP32):
  resolusi kerja 1920 / 1280 (720p) / 960 (540p);  mode tutup full-frame vs area botol (ROI) dengan beberapa kanvas & skala;  FP16.
Data: sesi 16 & 20 Okt (2026-10-16_water_v1, 2026-10-20_water_v1), label tutup dari nama file (capon/capoff).
  - Model TUTUP tidak pernah dilatih dengan sesi ini -> akurasi tutup = uji independen.
  - Model CAIRAN final (RH) DILATIH dengan sesi ini -> untuk cairan hanya dilaporkan KESEPAKATAN dengan acuan, bukan akurasi.
Hasil: datasets/4_eksperimen/benchmark/uji_profil_<tanggal>.txt dan .csv (per foto)."""
import csv, datetime, glob, os
import cv2, numpy as np
from .analyzer import RVMAnalyzer
from . import config

SESI = ["2026-10-16_water_v1", "2026-10-20_water_v1"]
# nama, kwargs RVMAnalyzer
CONFIGS = [
    ("acuan_1080_full_fp32", dict(frame_width=1920, cap_mode="full", half=False)),
    ("1080_roi320_s1/3", dict(frame_width=1920, cap_mode="roi", roi_imgsz=320, roi_scale=1 / 3, half=False)),
    ("1080_roi416_s1/3", dict(frame_width=1920, cap_mode="roi", roi_imgsz=416, roi_scale=1 / 3, half=False)),
    ("1080_roi416_s1/2", dict(frame_width=1920, cap_mode="roi", roi_imgsz=416, roi_scale=1 / 2, half=False)),
    ("720_full_fp32", dict(frame_width=1280, cap_mode="full", half=False)),
    ("720_roi320_s1/3", dict(frame_width=1280, cap_mode="roi", roi_imgsz=320, roi_scale=1 / 3, half=False)),
    ("720_roi416_s1/2", dict(frame_width=1280, cap_mode="roi", roi_imgsz=416, roi_scale=1 / 2, half=False)),
    ("540_roi320_s1/3", dict(frame_width=960, cap_mode="roi", roi_imgsz=320, roi_scale=1 / 3, half=False)),
    ("nano_720_roi320_fp16", dict(frame_width=1280, cap_mode="roi", roi_imgsz=320, roi_scale=1 / 3, half=True)),
    ("nano_720_roi416_s1/2_fp16", dict(frame_width=1280, cap_mode="roi", roi_imgsz=416, roi_scale=1 / 2, half=True)),
    ("1080_full_fp16", dict(frame_width=1920, cap_mode="full", half=True)),
]


def load():
    items = []
    for s in SESI:
        for f in sorted(glob.glob(os.path.join(config.REPO, "datasets", "1_foto_mentah", s, "b*.png"))):
            n = os.path.basename(f); tok = n[:-4].split("_")
            cap = "tutup" if "capon" in tok else ("tanpa_tutup" if "capoff" in tok else None)
            if cap is None:
                continue
            items.append(dict(file=n, sesi=s[:10], botol=s[5:10] + "_" + tok[0], cap=cap, kosong="kosong" in tok, path=f))
    return items


def gflops():
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    g = {}
    for k, pt, sz in [("botol", "botol_ver8b.pt", 640), ("tutup640", "cap_belt_final.pt", 640), ("tutup416", "cap_belt_final.pt", 416),
                      ("tutup320", "cap_belt_final.pt", 320), ("cairan", "cairan_RH.pt", 224)]:
        g[k] = get_flops(YOLO(os.path.join(config.M, pt)).model, sz)
    return g


def main():
    items = load(); frames = {it["file"]: cv2.imread(it["path"]) for it in items}
    rows, ref = [], {}
    g = gflops()
    od = os.path.join(config.REPO, "datasets", "4_eksperimen", "benchmark"); os.makedirs(od, exist_ok=True)
    L = [f"Uji profil {datetime.datetime.now():%Y-%m-%d %H:%M} | {len(items)} foto, "
         f"{len({i['botol'] for i in items})} botol, sesi {', '.join(SESI)} | PyTorch GPU laptop",
         f"Label tutup: {sum(i['cap'] == 'tutup' for i in items)} bertutup, {sum(i['cap'] == 'tanpa_tutup' for i in items)} tanpa tutup",
         "GFLOPs per model: " + ", ".join(f"{k} {v:.1f}" for k, v in g.items()), ""]
    hdr = (f"{'konfigurasi':28s} {'botol':>7s} {'tutup benar':>12s} {'bertutup':>9s} {'tanpa':>7s} {'?':>3s} "
           f"{'cairan=acuan':>12s} {'|dp|':>6s} {'ukuran=acuan':>12s} {'ms/frame':>9s} {'tutup ms':>8s} {'GFLOPs':>7s}")
    L.append(hdr)
    for name, kw in CONFIGS:
        an = RVMAnalyzer("pytorch", device=0, **kw)
        for f in list(frames.values())[:3]:
            an.analyze(f)                                           # pemanasan
        det = ok = okc = okn = unk = agree_l = agree_s = 0; dp = []; tt = []; tc = []
        for it in items:
            res = an.analyze(frames[it["file"]]); tt.append(an.last_timings["total"]); tc.append(an.last_timings["tutup"])
            r = max(res, key=lambda z: z.conf) if res else None
            row = dict(konfigurasi=name, file=it["file"], botol=it["botol"], cap_label=it["cap"], kosong=it["kosong"],
                       terdeteksi=r is not None, cap=r.cap if r else "", cap_conf=round(r.cap_conf, 3) if r else "",
                       p_liquid=round(r.p_liquid, 4) if r else "", size=r.size if r else "", area_1920=round(r.area_1920) if r else "")
            rows.append(row)
            if r is None:
                continue
            det += 1; good = r.cap == it["cap"]; ok += good; unk += r.cap == "?"
            okc += good and it["cap"] == "tutup"; okn += good and it["cap"] == "tanpa_tutup"
            if name == CONFIGS[0][0]:
                ref[it["file"]] = r
            elif it["file"] in ref:
                agree_l += r.has_liquid == ref[it["file"]].has_liquid; agree_s += r.size == ref[it["file"]].size
                dp.append(abs(r.p_liquid - ref[it["file"]].p_liquid))
        nc = sum(i["cap"] == "tutup" for i in items); nn = len(items) - nc
        cmp_n = sum(1 for it in items if it["file"] in ref)
        capg = g["tutup640"] if kw["cap_mode"] == "full" else g[f"tutup{kw['roi_imgsz']}"]
        fl = g["botol"] + capg + g["cairan"]
        first = name == CONFIGS[0][0]
        L.append(f"{name:28s} {det:>3d}/{len(items):<3d} {ok:>4d}/{det:<3d}{100 * ok / max(det, 1):4.0f}% {okc:>4d}/{nc:<4d} {okn:>3d}/{nn:<3d} {unk:>3d} "
                 f"{'-' if first else f'{agree_l}/{cmp_n}':>12s} {'-' if first else f'{np.mean(dp):.3f}':>6s} "
                 f"{'-' if first else f'{agree_s}/{cmp_n}':>12s} {np.mean(tt):9.1f} {np.mean(tc):8.1f} {fl:7.1f}")
        print(L[-1], flush=True)
        del an
    L += ["", "Catatan: 'tutup benar' = kelas tutup terbaik di botol utama sesuai nama file ('?' dihitung salah).",
          "GFLOPs = botol 640 + tutup (640 full-frame, atau kanvas ROI per botol) + cairan 224, untuk 1 botol per frame.",
          "ms/frame = laptop RTX 4050, BUKAN angka Jetson Nano."]
    p = os.path.join(od, f"uji_profil_{datetime.date.today()}")
    open(p + ".txt", "w", encoding="utf-8").write("\n".join(L) + "\n")
    with open(p + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("\n".join(L[-3:])); print("->", p + ".txt")


if __name__ == "__main__":
    main()
