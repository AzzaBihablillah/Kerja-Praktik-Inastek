"""Langkah 20: registri dataset -- hitung gambar dan kotak per kelas per split untuk semua dataset berformat YOLO di proyek
dan unduhan (tanpa memindah/mengubah apa pun). Keluaran: out/dataset_registry.csv + ringkasan di layar.
Dataset non-YOLO (foto mentah primary, crop eksperimen) dihitung terpisah di akhir."""
import csv, glob, json, os, collections
import yaml

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")                 # datasets/
OUT = os.path.join(ROOT, "4_eksperimen", "cairan")
DL = "C:/Users/ASUS/Downloads"
ENTRIES = [  # (kode, path, domain, catatan)
    ("bottle_detection_warisan", f"{ROOT}/5_data_luar/_warisan/bottle_detection_1280", "campuran web + rig belt lama", "dataset model bottle ver7 (diperkecil 1280)"),
    ("brand2_warisan", f"{DL}/brand2", "campuran web + rig lama", "dataset model merek brand_best_ver5"),
    ("2_dataset_cap/cap_v4", f"{ROOT}/2_dataset_cap/cap_v4", "sintetik + real studio + belt", "model cap v4 (3204 gambar)"),
    ("2_dataset_cap/cap_belt_all", f"{ROOT}/2_dataset_cap/cap_belt_all", "belt asli 1080p", "180 foto, 30 botol; model cap_belt_final"),
    ("2_dataset_cap/cap_shoot_v1", f"{ROOT}/2_dataset_cap/cap_shoot_v1", "studio (alas gelap)", "13 botol x 20 foto"),
    ("2_dataset_cap/cap_shoot_v2", f"{ROOT}/2_dataset_cap/cap_shoot_v2", "studio (alas putih)", "15 botol x 6 foto"),
    ("synth_cap_v1", f"{ROOT}/synth_cap_v1", "sintetik cut-paste", ""),
    ("synth_cap_v2", f"{ROOT}/synth_cap_v2", "sintetik cut-paste", "gagal (degradasi asimetris)"),
    ("synth_cap_v3", f"{ROOT}/synth_cap_v3", "sintetik cut-paste", ""),
    ("cap_exp_all_good", f"{ROOT}/cap_exp_all_good", "eksperimen komposit", ""),
    ("cap_exp_combined", f"{ROOT}/cap_exp_combined", "eksperimen komposit", ""),
    ("cap_exp_full_heavy", f"{ROOT}/cap_exp_full_heavy", "eksperimen komposit", ""),
    ("cap_exp_full_only", f"{ROOT}/cap_exp_full_only", "eksperimen komposit", ""),
    ("rf_bottle_cap_detection_2", f"{DL}/bottle cap detection 2.v1i.yolov8", "Roboflow (CC BY 4.0)", "unipolar-automation"),
    ("rf_cap_defect_v2", f"{DL}/cap_defect.v2i.yolov8", "Roboflow (CC BY 4.0)", "cacat tutup, bukan lingkup"),
    ("5_data_luar/water_internet_fanta", f"{ROOT}/5_data_luar/water_internet_fanta", "Roboflow, hasil augmentasi", "subset Fanta; split per foto sumber"),
    ("water_detection_roboflow", f"{ROOT}/5_data_luar/_dl_water_detection", "Roboflow, hasil augmentasi", "900 gambar 5 kelas warna"),
]


def names_of(d):
    for f in ("data.yaml",):
        p = os.path.join(d, f)
        if os.path.exists(p):
            y = yaml.safe_load(open(p, encoding="utf-8"))
            n = y.get("names")
            if isinstance(n, dict):
                return [n[k] for k in sorted(n)]
            if n:
                return list(n)
    p = os.path.join(d, "classes.txt")
    if os.path.exists(p):
        return open(p).read().split()
    return None


def scan(d):
    names = names_of(d)
    per = collections.defaultdict(lambda: [0, collections.Counter(), 0])           # split -> [gambar, kotak/kelas, gambar kosong]
    for f in glob.glob(os.path.join(d, "**", "*.txt"), recursive=True):
        parts = f.replace("\\", "/").split("/")
        if "labels" not in parts or os.path.basename(f) in ("classes.txt",):
            continue
        li = parts.index("labels")
        split = parts[li + 1] if li + 2 < len(parts) and parts[li + 1] in ("train", "val", "valid", "test") else "all"
        ids = [l.split()[0] for l in open(f).read().splitlines() if l.strip()]
        per[split][0] += 1
        per[split][2] += not ids
        for i in ids:
            per[split][1][int(float(i))] += 1
    return names, per


rows, lines = [], []
for code, path, dom, note in ENTRIES:
    if not os.path.isdir(path):
        lines.append(f"{code}: TIDAK ADA {path}"); continue
    names, per = scan(path)
    tot = collections.Counter(); n_img = 0; n_empty = 0
    for s, (ni, c, ne) in per.items():
        tot.update(c); n_img += ni; n_empty += ne
    desc = {(names[k] if names and k < len(names) else str(k)): v for k, v in sorted(tot.items())}
    big, small = (max(tot.values()), min(tot.values())) if tot else (0, 0)
    ratio = round(big / small, 1) if small else None
    rows.append(dict(kode=code, gambar=n_img, gambar_tanpa_kotak=n_empty, split=";".join(f"{s}:{v[0]}" for s, v in sorted(per.items())),
                     kelas=json.dumps(desc, ensure_ascii=False), rasio_terbesar_terkecil=ratio, domain=dom, catatan=note, path=os.path.relpath(path, ROOT) if path.startswith(ROOT) else path))
    lines.append(f"{code:28s} gambar {n_img:6d} (tanpa kotak {n_empty:4d}) | {desc} | rasio {ratio}")
# RF100 soda-bottles (COCO)
P = f"{ROOT}/5_data_luar/_dl_soda_bottles/raw/home/zuppif/Documents/Work/RoboFlow/ODinW-RF100-challenge/rf100/soda-bottles"
if os.path.isdir(P):
    cnt, ni = collections.Counter(), 0
    for sp in ("train", "valid", "test"):
        d = json.load(open(f"{P}/{sp}/_annotations.coco.json")); ni += len(d["images"]); names = {c["id"]: c["name"] for c in d["categories"]}
        cnt.update(names[a["category_id"]] for a in d["annotations"])
    lines.append(f"{'rf100_soda_bottles':28s} gambar {ni:6d} | {dict(cnt)} | foto lemari pendingin, semua botol penuh")
    rows.append(dict(kode="rf100_soda_bottles", gambar=ni, gambar_tanpa_kotak=0, split="train/valid/test", kelas=json.dumps(dict(cnt)), rasio_terbesar_terkecil=round(max(cnt.values()) / min(cnt.values()), 1),
                     domain="foto lemari pendingin (fisheye)", catatan="CC BY 4.0; semua positif", path="5_data_luar/_dl_soda_bottles"))
# foto mentah primary
for d in sorted(glob.glob(f"{ROOT}/1_foto_mentah/*/")):
    n = len([f for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))])
    lines.append(f"1_foto_mentah/{os.path.basename(os.path.dirname(d)):34s} {n:5d} foto mentah")
with open(f"{OUT}/dataset_registry.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print("\n".join(lines))
