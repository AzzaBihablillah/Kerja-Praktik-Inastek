"""Sistem analisis RVM untuk pekerjaan KP PT Inastek (tutup, cairan berwarna, ukuran). Lihat README.md."""
import numpy as _np

# TensorRT 8.2 (JetPack 4.6, Jetson Nano) memanggil np.bool, alias yang dihapus di numpy 1.24 (bawaan image Docker Ultralytics).
# Alias ini identik dengan bool bawaan Python; tanpa ini engine TensorRT gagal dimuat (uji 24 Okt).
if not hasattr(_np, "bool") or "bool" not in _np.__dict__:
    _np.bool = bool

from .analyzer import RVMAnalyzer, BottleResult, fold_crop  # noqa: F401,E402
