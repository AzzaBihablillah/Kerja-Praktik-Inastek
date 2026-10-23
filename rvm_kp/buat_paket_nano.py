"""Siapkan folder kecil untuk disalin ke Jetson Nano (flashdisk / scp), tanpa membawa seluruh datasets/ (±10 GB).
  python -m rvm_kp.buat_paket_nano          -> datasets/_paket_nano/  (±150 MB)
Isi: rvm_kp/ (kode), app/ (aplikasi kiosk, tanpa database lokal app/data/), 3 model .pt final, ONNX opset 12 (cadangan untuk trtexec),
40 foto belt 20 Okt (untuk benchmark & uji).
Struktur folder sama dengan proyek, jadi config.py tidak perlu diubah. Di Nano: salin ke ~/rvm lalu ikuti rvm_kp/JETSON-NANO.md."""
import glob, os, random, shutil
from .config import REPO, M, E

OUT = os.path.join(REPO, "datasets", "_paket_nano")
PT = ["botol_ver9c.pt", "cap_belt_v2.pt", "cairan_RH.pt", "merek_coca_kandidat.pt"]
ONNX = ["botol_ver9c.onnx", "cap_belt_v2.onnx", "cap_belt_v2_roi.onnx", "cairan_RH.onnx", "merek_coca_kandidat.onnx"]
SESI = "2026-10-20_water_v1"


def main():
    if os.path.exists(OUT):
        shutil.rmtree(OUT)                                           # folder hasil buatan skrip ini sendiri
    shutil.copytree(os.path.join(REPO, "rvm_kp"), os.path.join(OUT, "rvm_kp"), ignore=shutil.ignore_patterns("__pycache__"))
    app = os.path.join(REPO, "app")
    if os.path.isdir(app):                                           # data/ = database poin & akun uji laptop: tidak ikut
        shutil.copytree(app, os.path.join(OUT, "app"), ignore=shutil.ignore_patterns("__pycache__", "data"))
    md = os.path.join(OUT, "datasets", "3_model_final"); os.makedirs(os.path.join(md, "ekspor"))
    for f in PT:
        shutil.copy2(os.path.join(M, f), md)
    for f in ONNX:
        if os.path.exists(os.path.join(E, f)):
            shutil.copy2(os.path.join(E, f), os.path.join(md, "ekspor"))
    fs = sorted(glob.glob(os.path.join(REPO, "datasets", "1_foto_mentah", SESI, "b*.png")))
    random.Random(0).shuffle(fs)
    fd = os.path.join(OUT, "datasets", "1_foto_mentah", SESI); os.makedirs(fd)
    for f in fs[:40]:
        shutil.copy2(f, fd)
    size = sum(os.path.getsize(os.path.join(d, f)) for d, _, fl in os.walk(OUT) for f in fl)
    print(f"OK -> {OUT}  ({size / 1e6:.0f} MB). Salin folder ini ke Jetson (mis. ~/rvm), lalu baca rvm_kp/JETSON-NANO.md")


if __name__ == "__main__":
    main()
