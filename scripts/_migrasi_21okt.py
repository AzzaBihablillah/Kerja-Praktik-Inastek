"""Migrasi struktur direktori 21 Okt 2026 (disetujui user). TIDAK menghapus data (kecuali tmp_* & __pycache__, disetujui).
Semua pemindahan dicatat di datasets/PINDAH-LOG.csv (dari -> ke) sehingga dapat dibalik.
  python scripts/_migrasi_21okt.py --dry   (hanya tampilkan rencana)
  python scripts/_migrasi_21okt.py         (eksekusi)"""
import csv, os, shutil, subprocess, sys, glob

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(REPO, "datasets")
DRY = "--dry" in sys.argv
LOG = os.path.join(D, "PINDAH-LOG.csv")
tracked = set(subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True).stdout.split("\n"))

DATA = {  # sumber relatif datasets/ -> tujuan relatif datasets/
    "primary": "1_foto_mentah",
    **{k: f"2_dataset_cap/{k}" for k in ["cap_belt_v1", "cap_belt_ds", "cap_belt_all", "cap_v4", "_rf_export", "shoot_v1", "shoot_v2",
                                          "cap_shoot_v1", "cap_shoot_v2", "bottle_cap_base", "cap_conveyor_raw"]},
    "cap_runs": "4_eksperimen/cap/cap_runs", "_cv_belt": "4_eksperimen/cap/_cv_belt",
    "liquid_exp/out": "4_eksperimen/cairan", "liquid_pilot": "4_eksperimen/cairan_pilot",
    **{k: f"5_data_luar/{k}" for k in ["_dl_water_detection", "_dl_soda_bottles", "_dl_ext", "_dl_taco", "_warisan", "water_internet_fanta"]},
    **{k: f"_arsip/eksperimen_cap_lama/{k}" for k in ["synth_cap_v1", "synth_cap_v2", "synth_cap_v3", "cap_exp_all_good", "cap_exp_combined",
                                                       "cap_exp_full_heavy", "cap_exp_full_only", "nocap_labeling", "cap_shoot_v1_cutout"]},
    **{k: f"3_model_final/dasar/{k}" for k in ["yolov8n.pt", "yolov8s.pt", "yolov8n-seg.pt", "yolov8n-cls.pt", "yolo26n.pt"]},
    "liquid_baseline_feats.json": "4_eksperimen/cairan_pilot/liquid_baseline_feats.json",
}
SCRIPTS = {  # nama di datasets/ (atau liquid_exp/) -> tujuan relatif repo
    **{f: f"scripts/alat/{f}" for f in ["capture_water.py", "capture_free.py", "capture_conveyor.py", "water_qc.py", "qc_water_set.py",
                                        "live_liquid_demo.py", "keep_awake.py"]},
    **{f: f"scripts/pipeline_cap/{f}" for f in ["prelabel_belt.py", "build_belt_ds.py", "train_belt.py", "train_belt_final.py", "cv_belt.py",
                                                "cap_decider.py", "cap_decision_test.py", "live_cap_test.py", "live_ab_test.py"]},
    **{f: f"scripts/pipeline_cairan/{f}" for f in ["liquid_baseline.py", "liquid_learning_curve.py", "liquid_pilot_prep.py", "liquid_pilot_analyze.py"]},
    **{f: f"scripts/_arsip/{f}" for f in ["add_conveyor.py", "add_nocap.py", "cap_box_shoot.py", "compose_v4.py", "cutout_shoot.py", "label_shoot.py",
                                          "make_synth.py", "rename_shoot.py", "rename_shoot_v2.py", "salvage_held.py", "train_synth.py", "verify_nohand.py"]},
}
moves = []
for s, t in DATA.items():
    moves.append((os.path.join(D, s), os.path.join(D, t)))
for f, t in SCRIPTS.items():
    moves.append((os.path.join(D, f), os.path.join(REPO, t)))
for f in sorted(glob.glob(os.path.join(D, "liquid_exp", "*.py"))):
    moves.append((f, os.path.join(REPO, "scripts", "pipeline_cairan", os.path.basename(f))))
for f in sorted(os.listdir(D)):                                   # gambar debug lepas di datasets/
    if f.lower().endswith((".jpg", ".png")):
        moves.append((os.path.join(D, f), os.path.join(D, "_arsip", "gambar_debug", f)))
moves.append((os.path.join(REPO, "camera_check"), os.path.join(D, "_arsip", "camera_check_root")))
moves.append((os.path.join(REPO, "testing", "camera_check"), os.path.join(D, "_arsip", "camera_check_testing")))
DOCS = {"_build_nb.py": "laporan_kp/_build_nb.py", "laporan-deteksi-cap-nocap.ipynb": "laporan_kp/laporan-deteksi-cap-nocap.ipynb",
        "laporan-deteksi-cap-nocap.md": "laporan_kp/laporan-deteksi-cap-nocap.md", "DAFTAR-KERJA-19-OKT.md": "_arsip/DAFTAR-KERJA-19-OKT.md",
        "RENCANA-15-27-OKT.md": "_arsip/RENCANA-15-27-OKT.md", "HASIL-EKSPERIMEN-CAIRAN-18-OKT.md": "_arsip/HASIL-EKSPERIMEN-CAIRAN-18-OKT.md"}
for s, t in DOCS.items():
    moves.append((os.path.join(REPO, "docs", s), os.path.join(REPO, "docs", t)))

rows = []
for src, dst in moves:
    if not os.path.exists(src):
        print("LEWATI (tidak ada):", os.path.relpath(src, REPO)); continue
    if os.path.exists(dst):
        print("LEWATI (tujuan sudah ada):", os.path.relpath(dst, REPO)); continue
    rel = os.path.relpath(src, REPO).replace("\\", "/")
    print(("RENCANA " if DRY else "PINDAH  ") + rel + "  ->  " + os.path.relpath(dst, REPO).replace("\\", "/"))
    if DRY:
        continue
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if rel in tracked:
        subprocess.run(["git", "mv", src, dst], cwd=REPO, check=True)
    else:
        shutil.move(src, dst)
    rows.append([rel, os.path.relpath(dst, REPO).replace("\\", "/"), "git mv" if rel in tracked else "move"])

if not DRY:
    # hapus sementara (disetujui user): tmp_* di eksperimen cairan, semua __pycache__ di datasets/ dan scripts/
    gone = []
    for p in glob.glob(os.path.join(D, "4_eksperimen", "cairan", "tmp_*")):
        shutil.rmtree(p, ignore_errors=True); gone.append(p)
    for base in (D, os.path.join(REPO, "scripts")):
        for root, dirs, _ in os.walk(base):
            for dd in list(dirs):
                if dd == "__pycache__":
                    shutil.rmtree(os.path.join(root, dd), ignore_errors=True); gone.append(os.path.join(root, dd)); dirs.remove(dd)
    # log & flag eksperimen ke subfolder log/
    ex = os.path.join(D, "4_eksperimen", "cairan"); os.makedirs(os.path.join(ex, "log"), exist_ok=True)
    for f in os.listdir(ex):
        if (f.startswith("log_") and f.endswith(".txt")) or f.endswith(".flag") or f == "det_progress.txt":
            shutil.move(os.path.join(ex, f), os.path.join(ex, "log", f)); rows.append([f"datasets/liquid_exp/out/{f}", f"datasets/4_eksperimen/cairan/log/{f}", "move"])
    if os.path.isdir(os.path.join(D, "liquid_exp")) and not os.listdir(os.path.join(D, "liquid_exp")):
        os.rmdir(os.path.join(D, "liquid_exp"))
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["dari", "ke", "cara"])
        w.writerows(rows)
    print(f"\n{len(rows)} pemindahan dicatat di {os.path.relpath(LOG, REPO)}; dihapus (sementara): {len(gone)} folder")
