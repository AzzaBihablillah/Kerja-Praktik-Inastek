"""Benchmark kecepatan & kesesuaian hasil antar-backend pada foto belt nyata (bahan laporan).
  python -m rvm_kp.benchmark --backend pytorch openvino [--profil nano|laptop] [--device cpu] [--n 40]
  python -m rvm_kp.benchmark --backend tensorrt --profil nano            (di Jetson Nano)
Mengukur per frame: pra-proses, botol, tutup, cairan, total (rata-rata & persentil 95), memori (RAM proses, GPU torch),
dan kesepakatan keputusan cairan/tutup dengan backend pertama. Hasil: datasets/4_eksperimen/benchmark/benchmark_<tanggal>_<perangkat>.txt
Catatan jujur: angka berlaku untuk PERANGKAT INI saja."""
import argparse, datetime, glob, os, platform, random, time
import cv2, numpy as np
from .analyzer import RVMAnalyzer
from .config import REPO, DEFAULT_PROFILE


def mem():
    out = {}
    try:
        import psutil
        out["ram_mb"] = psutil.Process().memory_info().rss / 1e6
    except ImportError:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            out["gpu_mb"] = torch.cuda.max_memory_allocated() / 1e6
    except Exception:
        pass
    return out


def device_name():
    if os.path.exists("/proc/device-tree/model"):
        return open("/proc/device-tree/model").read().strip("\0 \n")
    return platform.processor() or platform.machine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", nargs="+", default=["pytorch", "openvino"])
    ap.add_argument("--profil", default=DEFAULT_PROFILE)
    ap.add_argument("--device", default=None, help="mis. cpu atau 0 (GPU); default: otomatis")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--dir", default=os.path.join(REPO, "datasets", "1_foto_mentah", "2026-10-20_water_v1"))
    ap.add_argument("--tag", default="", help="label tambahan nama file hasil")
    a = ap.parse_args()
    fs = sorted(glob.glob(os.path.join(a.dir, "b*.png")) + glob.glob(os.path.join(a.dir, "b*.jpg")))
    random.Random(0).shuffle(fs); frames = [cv2.imread(f) for f in fs[:a.n]]
    if not frames:
        raise SystemExit(f"tidak ada foto di {a.dir}")
    lines = [f"Benchmark {datetime.datetime.now():%Y-%m-%d %H:%M} | {device_name()} | profil={a.profil} | device={a.device or 'otomatis'} | "
             f"{len(frames)} frame {frames[0].shape[1]}x{frames[0].shape[0]} ({os.path.basename(a.dir)})"]
    try:
        import torch
        lines.append(f"Python {platform.python_version()}, torch {torch.__version__}, CUDA {torch.cuda.is_available()} "
                     f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    except Exception:
        pass
    ref = None
    for be in a.backend:
        try:
            an = RVMAnalyzer(be, device=a.device, profile=a.profil)
        except Exception as e:
            lines.append(f"{be:9s} GAGAL memuat: {str(e)[:160]}"); continue
        T = {k: [] for k in ("pra", "botol", "tutup", "cairan", "total")}; dec = []
        try:
            for f in frames[:3]:
                an.analyze(f)                                                # pemanasan
            for f in frames:
                t0 = time.perf_counter(); res = an.analyze(f); wall = (time.perf_counter() - t0) * 1e3
                for k in T:
                    T[k].append(wall if k == "total" else an.last_timings[k])
                r = max(res, key=lambda z: z.conf) if res else None
                dec.append((r.has_liquid, r.cap) if r else None)
        except Exception as e:
            lines.append(f"{be:9s} GAGAL saat inferensi: {str(e)[:160]}"); continue
        agree = ""
        if ref is not None:
            agree = (f" | sama dgn {a.backend[0]}: cairan {sum(x is not None and y is not None and x[0] == y[0] for x, y in zip(dec, ref))}/{len(dec)},"
                     f" tutup {sum(x is not None and y is not None and x[1] == y[1] for x, y in zip(dec, ref))}/{len(dec)}")
        ref = ref or dec
        m = mem()
        lines.append(f"{be:9s} [{an.cap_mode}, {an.frame_width}px, fp16={an.half or be == 'tensorrt'}] total {np.mean(T['total']):6.1f} ms "
                     f"(p95 {np.percentile(T['total'], 95):6.1f}) ≈ {1000 / np.mean(T['total']):5.1f} FPS | pra {np.mean(T['pra']):5.1f} | "
                     f"botol {np.mean(T['botol']):5.1f} | tutup {np.mean(T['tutup']):5.1f} | cairan {np.mean(T['cairan']):5.1f} ms | "
                     + " ".join(f"{k} {v:.0f}" for k, v in m.items()) + agree)
        print(lines[-1], flush=True)
    od = os.path.join(REPO, "datasets", "4_eksperimen", "benchmark"); os.makedirs(od, exist_ok=True)
    dev = "jetson" if os.path.exists("/etc/nv_tegra_release") else "laptop"
    p = os.path.join(od, f"benchmark_{datetime.date.today()}_{dev}_{a.profil}_{a.device or 'auto'}{a.tag}.txt")
    open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n"); print("\n".join(lines)); print("->", p)


if __name__ == "__main__":
    main()
