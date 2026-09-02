"""Fix-up: add the 109 real conveyor no_cap images (glob/label bug in add_conveyor.py)."""
import glob, os, re, random, shutil, collections
import cv2

CONV = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/bottle_cap_base/conveyor"
OUT = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/synth_cap_v2"
rng = random.Random(0)


def gkey(stem):
    s = re.sub(r"_jpg\.rf\.[0-9a-f]+$", "", stem)
    for pat in (r"^(mp_IMG_\d{8}_\d{6})", r"^(rg_L\d+)", r"^(rg_.*?_\d{8}_\d{6})"):
        m = re.match(pat, s)
        if m:
            return m.group(1)
    return s


picks = []
for sub in ("mp__no_cap", "rg__no_cap"):
    idir = os.path.join(CONV, "no_cap", sub, "images")
    ldir = os.path.join(CONV, "no_cap", sub, "labels")
    for ip in glob.glob(os.path.join(idir, "*.jpg")):
        stem = os.path.splitext(os.path.basename(ip))[0]
        lp = os.path.join(ldir, stem + ".txt")
        if os.path.exists(lp):
            picks.append((ip, lp, stem))
print(f"no_cap found: {len(picks)}")

groups = collections.defaultdict(list)
for _, _, st in picks:
    groups[gkey(st)].append(st)
gk = sorted(groups)
rng.shuffle(gk)
n = len(gk); n_tr = int(n * 0.8); n_va = int(n * 0.1)
sof = {}
for i, k in enumerate(gk):
    s = "train" if i < n_tr else "val" if i < n_tr + n_va else "test"
    for st in groups[k]:
        sof[st] = s

# skip any already present
added = collections.Counter()
for ip, lp, st in picks:
    name = f"cv_{st}"
    dst_split = sof[st]
    if any(os.path.exists(f"{OUT}/images/{s}/{name}.jpg") for s in ("train", "val", "test")):
        continue
    im = cv2.imread(ip)
    if im is None:
        continue
    if im.shape[:2] != (640, 640):
        im = cv2.resize(im, (640, 640))
    cv2.imwrite(f"{OUT}/images/{dst_split}/{name}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
    shutil.copy2(lp, f"{OUT}/labels/{dst_split}/{name}.txt")
    added[dst_split] += 1
print("no_cap added:", dict(added))

# recount
print("\n=== synth_cap_v2 FINAL (with real no_cap) ===")
tot = collections.Counter()
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
    tot["i"] += n_img; tot["c"] += cc[0]; tot["n"] += cc[1]
print(f"  TOTAL: {tot['i']} img | cap {tot['c']}  no_cap {tot['n']}")
