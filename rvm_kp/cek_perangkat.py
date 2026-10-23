"""Cek kesiapan perangkat untuk sistem RVM KP (jalan di Jetson Nano maupun laptop).
  python -m rvm_kp.cek_perangkat
Memeriksa: OS/JetPack, Python, mode daya & clock (Jetson), RAM & swap, torch+CUDA, TensorRT, Ultralytics, OpenCV, kamera,
file model, lalu memberi daftar langkah yang masih kurang. Tidak mengubah apa pun."""
import glob, os, platform, shutil, subprocess, sys

from . import config

OK, WARN, BAD = "[OK]  ", "[??]  ", "[X]   "
todo = []


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


def line(tag, msg, fix=None):
    print(tag + msg)
    if fix and tag != OK:
        todo.append(fix)


def main():
    print(f"== Sistem: {platform.system()} {platform.machine()} | Python {platform.python_version()}")
    dt = "/proc/device-tree/model"                    # di dalam Docker /etc/nv_tegra_release tidak ada, device-tree tetap terbaca
    jetson = os.path.exists("/etc/nv_tegra_release") or (os.path.exists(dt) and "jetson" in open(dt).read().lower())
    if jetson:
        rel = open("/etc/nv_tegra_release").readline().strip() if os.path.exists("/etc/nv_tegra_release") else "(di dalam Docker: versi L4T lihat di host)"
        line(OK, f"Jetson terdeteksi: {rel}")
        model = sh("tr -d '\\0' < /proc/device-tree/model 2>/dev/null || tr -d '\\0' < /etc/rvm_board_model")   # kedua: di Docker
        line(OK, f"Board: {model or '?'}")
        if "R32" in rel:
            line(OK, "L4T R32 = JetPack 4.x (Jetson Nano: versi terakhir 4.6.x)")
        q = sh("nvpmodel -q 2>/dev/null")
        line(OK if "MAXN" in q else WARN, f"Mode daya: {q.splitlines()[0] if q else '? (jalankan dengan sudo)'}",
             "sudo nvpmodel -m 0 && sudo jetson_clocks   (mode MAXN 10 W; WAJIB adaptor barrel 5V 4A + jumper J48)")
        gui = sh("systemctl get-default")
        line(OK if "multi-user" in gui else WARN, f"Target boot: {gui or '?'}",
             "Matikan desktop GUI agar RAM lega: sudo systemctl set-default multi-user.target && sudo reboot")
    else:
        line(OK, "Bukan Jetson (laptop/PC). Profil 'nano' di sini hanya MENIRU batasan Nano, bukan kecepatannya.")
    if sys.version_info < (3, 8):
        line(BAD, "Python < 3.8: Ultralytics modern tidak bisa dipasang",
             "Jalankan di dalam Docker ultralytics/ultralytics:latest-jetson-jetpack4 (Python 3.8) - lihat rvm_kp/JETSON-NANO.md")

    try:
        import psutil
        vm, sw = psutil.virtual_memory(), psutil.swap_memory()
        line(OK if vm.total > 3.5e9 else WARN, f"RAM {vm.total / 1e9:.1f} GB (tersedia {vm.available / 1e9:.1f} GB), swap {sw.total / 1e9:.1f} GB")
        if jetson and sw.total < 3.5e9:
            line(WARN, "Swap < 4 GB", "Tambah swapfile 4 GB (rvm_kp/JETSON-NANO.md langkah 3)")
    except ImportError:
        print(sh("free -h") or "(psutil tidak ada; lewati cek RAM)")

    try:
        import torch
        cuda = torch.cuda.is_available()
        line(OK if cuda else BAD, f"torch {torch.__version__}, CUDA {cuda} {torch.cuda.get_device_name(0) if cuda else ''}",
             "torch tanpa CUDA: di Jetson pakai torch dari image Docker Ultralytics/NVIDIA, bukan pip biasa")
    except ImportError:
        line(BAD, "torch tidak terpasang", "Pasang torch (laptop: index CUDA pytorch.org; Jetson: dari image Docker)")
    try:
        import tensorrt as trt
        line(OK, f"TensorRT {trt.__version__}")
    except ImportError:
        line(OK if not jetson else BAD, "TensorRT Python tidak ada" + (" (wajar di laptop)" if not jetson else ""),
             "Di Jetson: gunakan image Docker JetPack 4 (TensorRT 8.2 sudah ada)")
    try:
        import ultralytics
        line(OK, f"Ultralytics {ultralytics.__version__} (diuji: 8.4.54)")
    except ImportError:
        line(BAD, "Ultralytics tidak terpasang", "pip install ultralytics (di Jetson: sudah ada di image Docker)")
    import cv2
    info = cv2.getBuildInformation()
    gst = "GStreamer:                   YES" in info or "GStreamer: YES" in info.replace("  ", " ")
    line(OK, f"OpenCV {cv2.__version__} (GStreamer {'ya' if gst else 'tidak'})")
    for mod in ("fastapi", "uvicorn", "snap7"):
        try:
            __import__(mod); line(OK, f"{mod} ada")
        except Exception:
            line(WARN, f"{mod} tidak ada (perlu untuk server/PLC saja)", f"pip install {'python-snap7' if mod == 'snap7' else mod}")

    if platform.system() == "Linux":
        cams = sorted(glob.glob("/dev/video*"))
        line(OK if cams else WARN, f"Kamera: {', '.join(cams) or 'tidak ada /dev/video*'}",
             "Colok kamera C922; di Docker tambahkan --device /dev/video0")
        if cams and shutil.which("v4l2-ctl"):
            fm = sh(f"v4l2-ctl -d {cams[0]} --list-formats-ext | grep -A1 MJPG | head -4")
            print("      " + fm.replace("\n", "\n      "))

    print("== Model")
    for be, ms in config.MODEL_SETS.items():
        have = {k: os.path.exists(v) for k, v in ms.items()}
        need = ["bottle", "cap_roi", "liquid"]
        st = OK if all(have[k] for k in need) else (WARN if be != ("tensorrt" if jetson else "pytorch") else BAD)
        line(st, f"{be:9s} " + "  ".join(f"{k}={'ada' if v else '-'}" for k, v in have.items()),
             "Buat engine: python -m rvm_kp.export_models --format engine --half --workspace 1" if be == "tensorrt" and jetson else None)
    print("== Langkah berikutnya" if todo else "== Semua siap")
    for i, t in enumerate(dict.fromkeys(todo), 1):
        print(f"  {i}. {t}")


if __name__ == "__main__":
    main()
