"""Model merek langkah 7: model FINAL kandidat = resep A (foto belt saja, 30 epoch, seed 0) pada SEMUA 551 foto belt,
nama kelas coca_cola / non_coca_cola. Hasil CV resep ini: CATATAN-PERCOBAAN-TA §5 (F: 222/261 coca, salah terima 11/290).
  venv/Scripts/python.exe scripts/pipeline_merek/m07_final.py -> datasets/3_model_final/merek_coca_kandidat.pt"""
import os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from m02_cv_dasar import ROOT, OUT, CR, rows, train

NAMA = {"mitra": "coca_cola", "bukan": "non_coca_cola"}
R = [r for r in rows() if os.path.exists(f"{CR}/{r['id']}.jpg")]
print(len(R), "foto;", sum(r["kelas"] == "mitra" for r in R), "coca_cola", flush=True)
pt = train([(f"{CR}/{r['id']}.jpg", NAMA[r["kelas"]], r["id"]) for r in R], f"{OUT}/tmp/final", f"{OUT}/runs", "final_coca")
dst = f"{ROOT}/3_model_final/merek_coca_kandidat.pt"; shutil.copy2(pt, dst)
from ultralytics import YOLO
print("tersimpan", dst, YOLO(dst).names)
