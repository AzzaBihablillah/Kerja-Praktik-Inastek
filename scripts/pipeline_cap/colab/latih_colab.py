"""Latih model tutup v4w di Colab - resep IDENTIK scripts/pipeline_cap/cv_cap_v2.train (30 epoch, imgsz 960, batch 8, seed 0,
tanpa val, close_mosaic 10, degrees 0, scale 0.3, translate 0.1, fliplr 0.5); beda hanya workers (8, kecepatan muat data).
Hasil disimpan LANGSUNG di Drive (runs/) -> aman bila sesi putus; dijalankan ulang = lanjut dari epoch terakhir."""
import os, sys, torch
from ultralytics import YOLO
D = "/content/drive/MyDrive/rvm_cap_v4w"; POOL = "/content/pool"; RUNS = f"{D}/runs"
for name in sys.argv[1:] or ["cv_A29", "cv_B27", "final"]:
    last = f"{RUNS}/{name}/weights/last.pt"
    if os.path.exists(last):
        ep = torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1)
        if ep == -1:
            print(name, "SUDAH SELESAI"); continue
        print(name, "LANJUT dari epoch", ep + 1, flush=True); YOLO(last).train(resume=True); continue
    y = f"{POOL}/{name}.yaml"
    open(y, "w").write(f"path: {POOL}\ntrain: {POOL}/lists/{name}.txt\nval: {POOL}/lists/{name}.txt\nnames:\n  0: cap\n  1: no_cap\n")
    YOLO(f"{POOL}/cap_v4_640_best.pt").train(data=y, imgsz=960, epochs=30, batch=8, patience=0, val=False, seed=0, device=0, workers=8,
        cache=False, close_mosaic=10, degrees=0.0, scale=0.3, translate=0.1, fliplr=0.5, flipud=0.0, project=RUNS, name=name,
        plots=False, exist_ok=True, verbose=False)
    print(name, "SELESAI", flush=True)
