"""Compose the REAL portion of cap_v4.

val / test = real photos only (held-out whole bottles from shoot_v1 + shoot_v2,
held-out bursts from conveyor). NO synthetic, NO augmentation.
Synthetic is added 100% to train afterwards by make_synth.py (SYNTH_OUT=cap_v4, SYNTH_ALL_TRAIN=1).

Real train also goes in; synthetic just multiplies background/lighting variety on top.
Grouped split: shoot_v1/v2 by bottle id, conveyor by burst key.
"""
import glob, os, re, csv, random, collections
import cv2

D = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets"
OUT = f"{D}/cap_v4"
BASE = f"{D}/bottle_cap_base"
CONV = f"{BASE}/conveyor"
LONG = 960
SEED = 0
PER_COLOR = 130
rng = random.Random(SEED)

for s in ("train", "val", "test"):
    os.makedirs(f"{OUT}/images/{s}", exist_ok=True)
    os.makedirs(f"{OUT}/labels/{s}", exist_ok=True)


def save(img_path, lbl_path, split, name):
    im = cv2.imread(img_path)
    if im is None:
        return False
    h, w = im.shape[:2]
    if max(h, w) > LONG:
        sc = LONG / max(h, w)
        im = cv2.resize(im, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(f"{OUT}/images/{split}/{name}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if lbl_path and os.path.exists(lbl_path):
        open(f"{OUT}/labels/{split}/{name}.txt", "w").write(open(lbl_path).read())
    else:
        open(f"{OUT}/labels/{split}/{name}.txt", "w").close()
    return True


def split_groups(groups, n_val, n_test):
    keys = sorted(groups)
    rng.shuffle(keys)
    val = set(keys[:n_val])
    test = set(keys[n_val:n_val + n_test])
    return {k: ("val" if k in val else "test" if k in test else "train") for k in keys}


counts = collections.Counter()
report = {}

# ---------- shoot_v1: 13 bottles -> 10 / 2 / 1 ----------
g1 = collections.defaultdict(list)
for p in glob.glob(f"{D}/cap_shoot_v1/images/*.jpg"):
    stem = os.path.splitext(os.path.basename(p))[0]
    g1[re.match(r"(b\d+)", stem).group(1)].append((p, f"{D}/cap_shoot_v1/labels/{stem}.txt", stem))
s1_split = split_groups(g1, n_val=2, n_test=1)
report["shoot_v1"] = {k: s1_split[k] for k in sorted(g1)}
for bk, items in g1.items():
    for p, lp, stem in items:
        if save(p, lp, s1_split[bk], f"s1_{stem}"):
            counts[f"shoot_v1/{s1_split[bk]}"] += 1

# ---------- shoot_v2: 15 bottles -> 11 / 2 / 2 ----------
g2 = collections.defaultdict(list)
for p in glob.glob(f"{D}/cap_shoot_v2/images/*.jpg"):
    stem = os.path.splitext(os.path.basename(p))[0]
    g2[re.match(r"(b\d+)", stem).group(1)].append((p, f"{D}/cap_shoot_v2/labels/{stem}.txt", stem))
s2_split = split_groups(g2, n_val=2, n_test=2)
report["shoot_v2"] = {k: s2_split[k] for k in sorted(g2)}
for bk, items in g2.items():
    for p, lp, stem in items:
        if save(p, lp, s2_split[bk], f"s2_{stem}"):
            counts[f"shoot_v2/{s2_split[bk]}"] += 1


# ---------- conveyor ----------
def gkey(stem):
    s = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", stem)
    for pat in (r"^(bc_KakaoTalk_\d{8}_\d+)", r"^(mp_IMG_\d{8}_\d{6})",
                r"^(rg_L\d+)", r"^(rg_.*?_\d{8}_\d{6})"):
        m = re.match(pat, s)
        if m:
            return m.group(1)
    return s


# conveyor cap, colour-balanced
color_of = {}
for r in csv.DictReader(open(f"{BASE}/_by_cap_color/cap_color_manifest.csv")):
    fn = r["file"].split("/")[-1]                       # <src>_<origname>.jpg
    fn = re.sub(r"^(bc_|mp_|rg_)", "", fn, count=1)     # -> <origname>.jpg
    color_of[re.sub(r"_jpg\.rf\.[0-9a-f]+\.jpg$", "", fn)] = r["cap_color"]

cap_by_color = collections.defaultdict(list)
for src in ("bc__cap", "mp__cap", "rg__cap"):
    for p in glob.glob(f"{CONV}/cap/{src}/images/*.jpg"):
        stem = os.path.splitext(os.path.basename(p))[0]
        key = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", stem)
        col = color_of.get(key, "unknown")
        cap_by_color[col].append((p, f"{CONV}/cap/{src}/labels/{stem}.txt", stem))
cap_pick = []
for col, items in cap_by_color.items():
    rng.shuffle(items)
    cap_pick += items[:min(PER_COLOR, len(items))]

nocap_pick = []
for src in ("mp__no_cap", "rg__no_cap"):
    for p in glob.glob(f"{CONV}/no_cap/{src}/images/*.jpg"):
        stem = os.path.splitext(os.path.basename(p))[0]
        nocap_pick.append((p, f"{CONV}/no_cap/{src}/labels/{stem}.txt", stem))

# burst-group the whole conveyor pick, split 86 / 7 / 7
conv_all = cap_pick + nocap_pick
cg = collections.defaultdict(list)
for p, lp, stem in conv_all:
    cg[gkey(stem)].append((p, lp, stem))
n = len(cg)
conv_split = split_groups(cg, n_val=max(1, n // 14), n_test=max(1, n // 14))
for gk, items in cg.items():
    for p, lp, stem in items:
        if save(p, lp, conv_split[gk], f"cv_{stem}"):
            counts[f"conveyor/{conv_split[gk]}"] += 1

# ---------- data.yaml ----------
open(f"{OUT}/data.yaml", "w").write(
    f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\n"
    "names:\n  0: cap\n  1: no_cap\n")

# ---------- report ----------
print("=== bottle -> split ===")
for shoot, m in report.items():
    print(f"  {shoot}: " + "  ".join(f"{b}={s}" for b, s in m.items()))
print("\n=== real image counts ===")
for k in sorted(counts):
    print(f"  {k}: {counts[k]}")
tot = collections.Counter()
for k, v in counts.items():
    tot[k.split("/")[1]] += v
print(f"  --> REAL train {tot['train']} / val {tot['val']} / test {tot['test']}")


def box_classes(split):
    c = collections.Counter()
    for lf in glob.glob(f"{OUT}/labels/{split}/*.txt"):
        for ln in open(lf):
            if ln.strip():
                c[int(ln.split()[0])] += 1
    return c


for s in ("train", "val", "test"):
    c = box_classes(s)
    print(f"  {s} boxes: cap {c[0]}  no_cap {c[1]}")
print("\nNEXT: SYNTH_OUT=cap_v4 SYNTH_ALL_TRAIN=1 python make_synth.py   (adds synthetic to train only)")
