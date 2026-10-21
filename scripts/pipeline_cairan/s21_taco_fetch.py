"""Langkah 21: ambil negatif 'non_bottle' dari TACO (Trash Annotations in Context, CC BY 4.0; Proenca & Simoes 2020).
Pilihan: kantong/film plastik, pembungkus, kantong, gelas plastik, kaleng, botol kaca, karton, wadah makanan, styrofoam, foil, tutup plastik.
Aturan: gambar yang memuat 'Clear plastic bottle' / 'Other plastic bottle' DIBUANG (botol PET tak berlabel akan mencemari negatif);
kotak < 6% sisi pendek gambar dilewati. Gambar versi 640 px (flickr_640_url), unduh pelan (jeda 0.2 s, boleh dilanjutkan).
Keluaran: _dl_taco/nonbottle_ds/{images,labels} (kelas 0 = non_bottle saja) + out/taco_registry.csv."""
import csv, json, os, time, urllib.request
import cv2

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
T = f"{ROOT}/5_data_luar/_dl_taco"; OUT = f"{ROOT}/4_eksperimen/cairan"
os.makedirs(f"{T}/raw640", exist_ok=True); os.makedirs(f"{T}/nonbottle_ds/images", exist_ok=True); os.makedirs(f"{T}/nonbottle_ds/labels", exist_ok=True)
d = json.load(open(f"{T}/annotations.json")); cat = {c["id"]: c["name"] for c in d["categories"]}
NEG = {"Plastic film", "Other plastic wrapper", "Single-use carrier bag", "Garbage bag", "Crisp packet", "Disposable plastic cup", "Drink can", "Food can",
       "Glass bottle", "Drink carton", "Other carton", "Disposable food container", "Foam food container", "Foam cup", "Styrofoam piece", "Aluminium foil",
       "Paper bag", "Plastic lid", "Paper cup", "Meal carton", "Tupperware", "Spread tub", "Squeezable tube", "Aerosol", "Glass jar", "Glass cup"}
BOT = {"Clear plastic bottle", "Other plastic bottle"}
by = {}
for a in d["annotations"]:
    by.setdefault(a["image_id"], []).append(a)
UA = {"User-Agent": "KP-RVM-UGM/1.0 (academic internship dataset research)"}
rows, n_ok, n_fail = [], 0, 0
for im in d["images"]:
    A = by.get(im["id"], [])
    if {cat[a["category_id"]] for a in A} & BOT:
        continue
    sel = [a for a in A if cat[a["category_id"]] in NEG and min(a["bbox"][2:]) >= 0.06 * min(im["width"], im["height"])]
    if not sel:
        continue
    name = f"taco_{im['id']:04d}"
    p = f"{T}/raw640/{name}.jpg"
    if not os.path.exists(p):
        data = None
        for url in (im.get("flickr_640_url"), im.get("flickr_url")):          # 640 px dulu, lalu ukuran asli (di-resize ke 640)
            if not url:
                continue
            for wait in (0, 5, 20):
                time.sleep(wait)
                try:
                    data = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read(); break
                except Exception as e:
                    if "429" not in str(e):
                        break
            if data:
                break
        if not data:
            n_fail += 1; continue
        arr = cv2.imdecode(__import__("numpy").frombuffer(data, "uint8"), cv2.IMREAD_COLOR)
        if arr is None:
            n_fail += 1; continue
        s_ = 640 / max(arr.shape[:2])
        if s_ < 1:
            arr = cv2.resize(arr, (round(arr.shape[1] * s_), round(arr.shape[0] * s_)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(p, arr, [cv2.IMWRITE_JPEG_QUALITY, 92]); time.sleep(0.4)
    img = cv2.imread(p)
    if img is None:
        n_fail += 1; continue
    H, W = img.shape[:2]; sx, sy = W / im["width"], H / im["height"]
    lines = []
    for a in sel:
        x, y, w, h = a["bbox"]; x, y, w, h = x * sx, y * sy, w * sx, h * sy
        lines.append(f"0 {(x + w / 2) / W:.6f} {(y + h / 2) / H:.6f} {w / W:.6f} {h / H:.6f}")
        rows.append(dict(image=name, categoria=cat[a["category_id"]], w=round(w), h=round(h)))
    os.replace(p, p) if False else None
    cv2.imwrite(f"{T}/nonbottle_ds/images/{name}.jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    open(f"{T}/nonbottle_ds/labels/{name}.txt", "w").write("\n".join(lines) + "\n")
    n_ok += 1
    if n_ok % 50 == 0:
        print("gambar", n_ok, flush=True)
open(f"{T}/nonbottle_ds/classes.txt", "w").write("non_bottle\n")
open(f"{T}/LICENSE-NOTE.txt", "w").write("TACO: Proenca & Simoes, 'TACO: Trash Annotations in Context for Litter Detection', arXiv:2003.06975. Gambar dari Flickr, anotasi CC BY 4.0. Wajib atribusi.\n")
with open(f"{OUT}/taco_registry.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["image", "categoria", "w", "h"]); w.writeheader(); w.writerows(rows)
print("selesai", n_ok, "gambar,", len(rows), "kotak, gagal", n_fail)
