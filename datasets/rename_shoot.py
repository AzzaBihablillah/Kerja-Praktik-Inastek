"""Rename the 260 shoot photos into brand folders.
name-sorted order = shoot order = brand order, chunked 20 per bottle.
  bottle 1     : slots 1-10 = no_cap, 11-20 = cap   (generic slot numbers)
  bottles 2-13 : slots 1-10 = cap,    11-20 = no_cap (full slot template)
"""
import glob, os, csv, shutil

RAW = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/shoot_v1_raw"
OUT = "C:/Users/ASUS/D/project/Kerja-Praktik-Inastek/datasets/shoot_v1"

BOTTLES = [
    ("le-mineral", "besar"), ("aqua", "tanggung"), ("nestle-purelife", "besar"),
    ("crystalin", "tanggung"), ("natsbee-honey-orange", "tanggung"),
    ("air-mineral-alfamart", "besar"), ("cleo", "besar"),
    ("big-cola-lemon-lime", "1l"), ("big-cola", "1l"), ("big-cola-nipis-madu", "1l"),
    ("minyak-tropical", "1l"), ("isoplus", "tanggung"), ("le-mineral", "tanggung"),
]
SLOTS = ["topdown-side1", "topdown-side2", "topdown-side3",
         "lift45-N", "lift45-NE", "lift45-E", "lift45-SE", "lift45-S",
         "side45-up", "side45-down"]

imgs = sorted(glob.glob(os.path.join(RAW, "*.jpg")), key=lambda p: os.path.basename(p))
print(f"{len(imgs)} raw images (expect 260)")
assert len(imgs) == 260, "count != 260 - check for retakes/missing"

if os.path.exists(OUT):
    shutil.rmtree(OUT)
rows = []
for bi, (brand, size) in enumerate(BOTTLES):
    chunk = imgs[bi * 20:(bi + 1) * 20]
    tag = f"b{bi+1:02d}_{brand}-{size}"
    os.makedirs(os.path.join(OUT, tag), exist_ok=True)
    for k, src in enumerate(chunk):
        first_half = k < 10
        slot_i = k % 10
        if bi == 0:                                  # bottle 1: no_cap first, generic slots
            state = "nocap" if first_half else "cap"
            slot = f"{slot_i+1:02d}"
            slot_name = ""
        else:                                        # bottles 2-13: cap first, templated
            state = "cap" if first_half else "nocap"
            slot = f"{slot_i+1:02d}_{SLOTS[slot_i]}"
            slot_name = SLOTS[slot_i]
        name = f"{tag}_{state}_{slot}.jpg"
        shutil.copy2(src, os.path.join(OUT, tag, name))
        rows.append([f"{tag}/{name}", bi + 1, brand, size, state, slot_i + 1,
                     slot_name, "yes" if slot_i < 3 else "", os.path.basename(src)])

with open(os.path.join(OUT, "MANIFEST.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["file", "bottle_no", "brand", "size", "cap_state", "slot",
                "slot_name", "brand_ref", "orig_name"])
    w.writerows(rows)

print(f"wrote {len(rows)} -> {OUT}\n")
for bi, (brand, size) in enumerate(BOTTLES):
    d = os.path.join(OUT, f"b{bi+1:02d}_{brand}-{size}")
    nc = len(glob.glob(os.path.join(d, "*_cap_*.jpg")))
    nn = len(glob.glob(os.path.join(d, "*_nocap_*.jpg")))
    print(f"  b{bi+1:02d} {brand}-{size:9s}  cap {nc}  nocap {nn}")
