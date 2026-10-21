"""Perbarui path di skrip AKTIF (alat/, pipeline_cap/, pipeline_cairan/) setelah migrasi 21 Okt.
Skrip di scripts/_arsip/ sengaja TIDAK diubah (path lama). Menulis log perubahan ke scripts/_migrasi_path_log.txt."""
import os, re, glob

S = os.path.dirname(os.path.abspath(__file__))
DSX = 'os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")'   # repo/datasets
TOK = {"primary": "1_foto_mentah", "cap_runs": "4_eksperimen/cap/cap_runs", "_cv_belt": "4_eksperimen/cap/_cv_belt",
       "liquid_pilot": "4_eksperimen/cairan_pilot",
       **{k: f"5_data_luar/{k}" for k in ["_warisan", "_dl_water_detection", "_dl_soda_bottles", "_dl_ext", "_dl_taco", "water_internet_fanta"]},
       **{k: f"2_dataset_cap/{k}" for k in ["cap_belt_v1", "cap_belt_ds", "cap_belt_all", "cap_v4", "_rf_export", "shoot_v1", "shoot_v2",
                                            "cap_shoot_v1", "cap_shoot_v2", "bottle_cap_base", "cap_conveyor_raw"]},
       **{k: f"3_model_final/dasar/{k}" for k in ["yolov8n-cls.pt", "yolov8n.pt", "yolov8s.pt", "yolov8n-seg.pt", "yolo26n.pt"]}}
log = []
for f in sorted(glob.glob(f"{S}/alat/*.py") + glob.glob(f"{S}/pipeline_cap/*.py") + glob.glob(f"{S}/pipeline_cairan/*.py")):
    s0 = s = open(f, encoding="utf-8").read()
    s = s.replace('"liquid_exp", "out"', '"4_eksperimen", "cairan"').replace("liquid_exp/out", "4_eksperimen/cairan")
    for k, v in TOK.items():
        s = re.sub(r'(?<=[/"\'\\])' + re.escape(k) + r'(?=[/"\'\\])', v, s)
    # akar datasets: skrip pindah dari datasets/ (atau datasets/liquid_exp/) ke scripts/<sub>/
    s = s.replace("ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))", f"ROOT = {DSX}")
    s = s.replace('OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")', f'OUT = os.path.join({DSX}, "4_eksperimen", "cairan")')
    s = s.replace("os.chdir(os.path.dirname(os.path.abspath(__file__)))", f"os.chdir({DSX})")
    if s != s0:
        open(f, "w", encoding="utf-8").write(s)
        log.append(f"{os.path.relpath(f, S)}: {sum(1 for a, b in zip(s0.splitlines(), s.splitlines()) if a != b)} baris diubah")
open(f"{S}/_migrasi_path_log.txt", "w", encoding="utf-8").write("\n".join(log) + "\n")
print("\n".join(log))
