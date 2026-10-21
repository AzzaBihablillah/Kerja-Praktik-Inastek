"""Append real conveyor cap/no_cap data to synth_cap_v3, colour-balanced.

cap    : sampled ~PER_COLOR per cap-colour bucket (from _by_cap_color manifest), tight labels
no_cap : all conveyor no_cap (mp + rg)
grouped split by burst key (80/10/10), copied as cv_<stem>.jpg into the synth splits.
Then writes data.yaml and prints the combined dataset composition.
"""
import csv, os, re, glob, random, shutil, collections
import cv2

BASE = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/bottle_cap_base"
CONV = f"{BASE}/conveyor"
OUT = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/synth_cap_v3"
PER_COLOR = 130
SPLIT = (0.80, 0.10, 0.10)
SEED = 0
rng = random.Random(SEED)


def gkey(stem):
    s = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", stem)
    for pat in (r"^(bc_KakaoTalk_\d{8}_\d+)_\d+$", r"^(mp_IMG_\d{8}_\d{6})",
                r"^(rg_.*?_\d{8}_\d{6})", r"^(bc_P\d+)_"):
        m = re.match(pat, s)
        if m:
            return m.group(1)
    return s


def split_groups(stems):
    groups = collections.defaultdict(list)
    for st in stems:
        groups[gkey(st)].append(st)
    gk = sorted(groups)
    rng.shuffle(gk)
    n = len(gk); n_tr = int(n * SPLIT[0]); n_va = int(n * SPLIT[1])
    out = {}
    for i, k in enumerate(gk):
        s = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
        for st in groups[k]:
            out[st] = s
    return out


# ---- cap: colour -> list of (img_path, label_path) ----
man = list(csv.DictReader(open(f"{BASE}/_by_cap_color/cap_color_manifest.csv")))
by_color = collections.defaultdict(list)
for r in man:
    src = r["source"]
    orig = os.path.basename(r["file"])[len(src) + 1:]           # strip "<src>_"
    ip = f"{CONV}/cap/{src}__cap/images/{orig}"
    lp = f"{CONV}/cap/{src}__cap/labels/{os.path.splitext(orig)[0]}.txt"
    if os.path.exists(ip) and os.path.exists(lp):
        by_color[r["cap_color"]].append((ip, lp))

cap_pick = []
for col, lst in sorted(by_color.items()):
    rng.shuffle(lst)
    take = lst[:PER_COLOR]
    cap_pick += take
    print(f"  cap {col:14s} avail {len(lst):4d}  take {len(take)}")

# ---- no_cap: all ----
nocap_pick = []
for d in glob.glob(f"{CONV}/no_cap/*/images/*.jpg"):
    lp = d.replace("/images/", "/labels/")[:-4] + ".txt"
    if os.path.exists(lp):
        nocap_pick.append((d, lp))
print(f"  no_cap avail {len(nocap_pick)}  take all")

# ---- grouped split per class, copy in ----
added = {"train": 0, "val": 0, "test": 0}
for cls_name, picks in (("cap", cap_pick), ("no_cap", nocap_pick)):
    stems = [os.path.splitext(os.path.basename(ip))[0] for ip, _ in picks]
    sof = split_groups(stems)
    for (ip, lp), st in zip(picks, stems):
        split = sof[st]
        im = cv2.imread(ip)
        if im is None:
            continue
        if im.shape[:2] != (640, 640):
            im = cv2.resize(im, (640, 640))
        name = f"cv_{st}"
        cv2.imwrite(f"{OUT}/images/{split}/{name}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        shutil.copy2(lp, f"{OUT}/labels/{split}/{name}.txt")
        added[split] += 1
print(f"conveyor added: {added}")

# ---- data.yaml ----
open(f"{OUT}/data.yaml", "w").write(
    f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\n"
    "names:\n  0: cap\n  1: no_cap\n")

# ---- final composition ----
print("\n=== synth_cap_v3 FINAL ===")
grand = collections.Counter()
for split in ("train", "val", "test"):
    n_img = len(glob.glob(f"{OUT}/images/{split}/*.jpg"))
    cc = collections.Counter()
    for lf in glob.glob(f"{OUT}/labels/{split}/*.txt"):
        for ln in open(lf):
            if ln.strip():
                cc[int(ln.split()[0])] += 1
    syn = len(glob.glob(f"{OUT}/images/{split}/syn_*.jpg"))
    cv = len(glob.glob(f"{OUT}/images/{split}/cv_*.jpg"))
    print(f"  {split:5s}: {n_img:5d} img (syn {syn}, conveyor {cv}) | boxes cap {cc[0]}  no_cap {cc[1]}")
    grand["img"] += n_img; grand["cap"] += cc[0]; grand["nocap"] += cc[1]
print(f"  TOTAL: {grand['img']} img | boxes cap {grand['cap']}  no_cap {grand['nocap']}")
