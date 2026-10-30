"""Lanjutkan model tutup v4w FINAL dari checkpoint Colab (epoch 28/30, runtime terputus) DI LAPTOP:
data latih lokal dibangun dengan isi identik (data lama + belt 27/29 + salinan warna teraudit), lalu alamat data/proyek di
checkpoint diarahkan ke folder lokal (workers=0 seperti latih lokal). Hasil -> datasets/3_model_final/cap_belt_v4w_kandidat.pt
(kandidat bayangan; TIDAK aktif - v4w tidak lulus kriteria per hari, CATATAN L-33)."""
import os, shutil, sys, torch
from multiprocessing import freeze_support
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_cap_v2 as v2, cap_v3_belt as c3, cap_v4_warna as c4


def main():
    c3.OUT = c4.OUT; v2.CV = c4.OUT
    src = f"{c4.OUT}/runs/_lokal_terhenti/final_colab_ep28/last.pt"
    tr = v2.items() + c3.new_items(); aug = c4.augment(tr); d = f"{c4.OUT}/data_final"
    print(len(tr), "+", len(aug), "=", len(tr) + len(aug), "(Colab: 3044)", flush=True)
    assert len(tr) + len(aug) == 3044
    c3.build(d, tr + aug)
    run = f"{c4.OUT}/runs/final"; os.makedirs(f"{run}/weights", exist_ok=True)
    ck = torch.load(src, map_location="cpu", weights_only=False); a = ck["train_args"]
    print("lanjut dari epoch", ck["epoch"] + 1, "dari", a["epochs"], flush=True)
    a.update(data=f"{d}/data.yaml", project=f"{c4.OUT}/runs", name="final", save_dir=run, workers=0, device=0)
    torch.save(ck, f"{run}/weights/last.pt")
    from ultralytics import YOLO
    YOLO(f"{run}/weights/last.pt").train(resume=True)
    ck2 = torch.load(f"{run}/weights/last.pt", map_location="cpu", weights_only=False)
    assert ck2.get("epoch", 0) == -1, "latih belum selesai"
    shutil.copy2(f"{run}/weights/last.pt", f"{v2.DS}/3_model_final/cap_belt_v4w_kandidat.pt"); shutil.rmtree(d, ignore_errors=True)
    print("FINAL ->", f"{v2.DS}/3_model_final/cap_belt_v4w_kandidat.pt", flush=True)


if __name__ == "__main__":
    freeze_support(); main()
