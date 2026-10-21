"""Organize shoot_v2: 90 photos = 15 bottles x 6, name-sorted = shoot order.

per-bottle 6-photo sequence:
  1  cap    topdown  side A
  2  cap    topdown  side B
  3  no_cap topdown  side A
  4  no_cap topdown  side B
  5  no_cap lift (diangkat dikit)
  6  cap    lift (diangkat dikit)

Downscales 108 MP -> ~2 MP (PIL draft, chunked) into shoot_v2/bNN_<slug>/ + MANIFEST.csv
"""
import glob, os, csv, time, shutil
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

RAW = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/primary/2026-09-06_shoot_v2_raw"
OUT = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/shoot_v2"
LONG = 1632                                   # ~2 MP for 4:3

BOTTLES = [
    "vica-kecil", "teh-pucuk-less-sugar", "air-alam-tanggung", "pristine-tanggung",
    "ichitan-thai-milk-green-tea", "pocari-800ml", "kopiko-coffee-latte", "cheers-tanggung",
    "ichitan-korean-banana-milk", "frestea-melati", "golda", "air-mineral-fitri-kecil",
    "vit-kecil", "air-mineral-superindo-tanggung", "clairine-tanggung",
]
# slot -> (cap_state, slot_name)
SLOTS = {
    1: ("cap", "topdown-A"), 2: ("cap", "topdown-B"),
    3: ("nocap", "topdown-A"), 4: ("nocap", "topdown-B"),
    5: ("nocap", "lift"), 6: ("cap", "lift"),
}

imgs = sorted(glob.glob(os.path.join(RAW, "*.jpg")), key=lambda p: os.path.basename(p))
assert len(imgs) == 90, f"expected 90, got {len(imgs)}"

if os.path.exists(OUT):
    shutil.rmtree(OUT)
rows = []
for bi, brand in enumerate(BOTTLES):
    chunk = imgs[bi * 6:(bi + 1) * 6]
    tag = f"b{bi+1:02d}_{brand}"
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    for k, src in enumerate(chunk):
        slot = k + 1
        state, sname = SLOTS[slot]
        name = f"{tag}_{state}_{slot:02d}_{sname}.jpg"
        dst = os.path.join(OUT, tag, name)
        im = Image.open(src)
        im.draft("RGB", (LONG * 2, LONG * 2))
        im = im.convert("RGB")
        sc = LONG / max(im.size)
        im = im.resize((round(im.size[0] * sc), round(im.size[1] * sc)), Image.LANCZOS)
        im.save(dst, "JPEG", quality=92)
        im.close()
        rows.append([f"{tag}/{name}", bi + 1, brand, state, slot, sname, os.path.basename(src)])
    print(f"  b{bi+1:02d} {brand:32s}  cap {sum(1 for r in rows[-6:] if r[3]=='cap')}  nocap {sum(1 for r in rows[-6:] if r[3]=='nocap')}")
    time.sleep(0.4)

with open(os.path.join(OUT, "MANIFEST.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "bottle_no", "brand", "cap_state", "slot", "slot_name", "orig_name"])
    w.writerows(rows)
open(os.path.join(OUT, "classes.txt"), "w").write("cap\nno_cap\n")
nc = sum(1 for r in rows if r[3] == "cap"); nn = sum(1 for r in rows if r[3] == "nocap")
print(f"\nwrote {len(rows)} -> {OUT}  |  cap {nc} / no_cap {nn}")
