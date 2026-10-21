"""Ekspor model .pt ke format perangkat lain (hasil ke datasets/3_model_final/ekspor/).
  python -m rvm_kp.export_models --format onnx openvino                (laptop)
  python -m rvm_kp.export_models --format engine --half --workspace 1  (DI Jetson Nano, di dalam Docker JetPack 4; ±10-20 menit/model)
Model tutup diekspor dua kali: ukuran full-frame (cap_belt_final.*) dan ukuran kanvas area botol (cap_belt_final_roi.*).
Opset ONNX default 12: TensorRT 8.2 (JetPack 4.6, Jetson Nano) tidak menjamin opset baru; opset 12 cukup untuk semua operasi YOLOv8.
Catatan: .engine terikat versi TensorRT & GPU tempat ia dibuat; JANGAN disalin dari laptop ke Jetson."""
import argparse, os, shutil
from ultralytics import YOLO
from .config import M, E, IMGSZ_BOTTLE, IMGSZ_CAP_FULL, IMGSZ_LIQUID, CAP_ROI_IMGSZ

MODELS = {
    "botol_ver8b": (f"{M}/botol_ver8b.pt", IMGSZ_BOTTLE),
    "cap_belt_final": (f"{M}/cap_belt_final.pt", IMGSZ_CAP_FULL),
    "cap_belt_final_roi": (f"{M}/cap_belt_final.pt", CAP_ROI_IMGSZ),
    "cairan_RH": (f"{M}/cairan_RH.pt", IMGSZ_LIQUID),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", nargs="+", default=["onnx", "openvino"])
    ap.add_argument("--half", action="store_true", help="FP16 (wajib untuk engine Jetson Nano)")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--workspace", type=float, default=None, help="GB memori kerja TensorRT (Nano 4 GB: 1)")
    ap.add_argument("--only", nargs="+", default=None, help="nama model tertentu, mis. cap_belt_final_roi")
    a = ap.parse_args(); os.makedirs(E, exist_ok=True)
    for name, (pt, imgsz) in MODELS.items():
        if a.only and name not in a.only:
            continue
        for fmt in a.format:
            tmp = os.path.join(E, "_tmp_" + name); os.makedirs(tmp, exist_ok=True)
            src = os.path.join(tmp, name + ".pt"); shutil.copy2(pt, src)          # ekspor ditulis di samping .pt -> pakai salinan
            kw = dict(format=fmt, imgsz=imgsz, half=a.half, dynamic=False, batch=1)
            if fmt in ("onnx", "engine"):
                kw.update(opset=a.opset, simplify=True)
            if fmt == "engine" and a.workspace:
                kw["workspace"] = a.workspace
            outp = YOLO(src).export(**kw)
            ext = {"onnx": ".onnx", "engine": ".engine"}.get(fmt)
            dst = os.path.join(E, name + (ext if ext else f"_{fmt}_model"))
            if os.path.exists(dst):
                shutil.rmtree(dst) if os.path.isdir(dst) else os.remove(dst)
            shutil.move(str(outp), dst)
            shutil.rmtree(tmp, ignore_errors=True)
            print("OK", fmt, f"imgsz {imgsz}", "->", os.path.relpath(dst), flush=True)


if __name__ == "__main__":
    main()
