"""Detektor botol vs non-botol: uji negatif TACO-tempel, dan (bila perlu) latih ver9 = ver8b + negatif TACO-tempel.
  python scripts/pipeline_botol/ver9_nonbottle.py uji  [model.pt ...]   bangun set uji sintetis (sekali) + evaluasi model
  python scripts/pipeline_botol/ver9_nonbottle.py latih                 bangun data latih sintetis + latih ver9 (resep ver8b)
  python scripts/pipeline_botol/ver9_nonbottle.py latih_b / latih_c     ver9b (+botol ber-alpha) / ver9c (+ potongan TrashNet)
Keluaran: datasets/4_eksperimen/botol/ (uji_taco/, latih_taco/, ver9/, hasil_uji_<model>.txt, ringkasan.txt)

Evaluasi (ambang sistem: kelas bottle conf >= 0.35):
  A. objek TACO-tempel (objek & foto uji tak pernah dilatih) DITERIMA sebagai botol  -> salah terima (makin kecil makin baik)
  B. botol asli di foto yang sama tetap terdeteksi
  C. sesi 20 Okt asli (117 foto, tak pernah dilatih detektor): botol terdeteksi
  D. old_test data tim (284 foto, foto nyata; banyak non-botol nyata di belt rig lama): mAP50 per kelas, recall
  E. foto nyata non-botol (intake 6 Sep, 3 foto alas putih): diterima sebagai botol?"""
import csv, glob, json, os, random, shutil, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from taco_paste import DS, DD, load_objects, paste_object, paste_bottle, yolo_line

OUT = f"{DS}/4_eksperimen/botol"; UJI = f"{OUT}/uji_taco"; LAT = f"{OUT}/latih_taco"
V7 = os.path.join(DS, "..", "export_model", "bottle_best_ver7.pt")
V8B = f"{DS}/3_model_final/botol_ver8b.pt"
CONF = 0.35


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0, min(a[3], b[3]) - max(a[1], b[1])); i = ix * iy
    return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i + 1e-9)


def build_uji():
    if os.path.exists(f"{UJI}/meta.json"):
        return json.load(open(f"{UJI}/meta.json"))
    from ultralytics import YOLO
    os.makedirs(f"{UJI}/images", exist_ok=True)
    det = YOLO(V8B); objs = load_objects()["uji"]; rng = random.Random(7)
    print("objek uji:", len(objs), "dari", len({o["img"] for o in objs}), "gambar TACO", flush=True)
    meta = []
    fs = sorted(glob.glob(f"{DS}/1_foto_mentah/2026-10-20_water_v1/b*.png")); k = 0
    for rep in range(2):
        for f in fs:
            im = cv2.resize(cv2.imread(f), (960, 540), interpolation=cv2.INTER_AREA)
            r = det.predict(im, imgsz=640, conf=CONF, verbose=False)[0]
            bb = [b.xyxy[0].tolist() for b in r.boxes if int(b.cls) == 0]
            if not bb:
                continue
            bot = max(bb, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            taken = [(bot[0] - 15, bot[1] - 15, bot[2] + 15, bot[3] + 15)]; pasted = []
            for _ in range(2):
                o = objs[k % len(objs)]; k += 1
                b = paste_object(im, o, taken, rng)
                if b:
                    taken.append((b[0] - 10, b[1] - 10, b[2] + 10, b[3] + 10)); pasted.append(dict(box=[int(v) for v in b], cat=o["cat"]))
            name = f"u{rep}_{os.path.basename(f)[:-4]}"
            cv2.imwrite(f"{UJI}/images/{name}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            meta.append(dict(name=name, bottle=[int(v) for v in bot], pasted=pasted))
    json.dump(meta, open(f"{UJI}/meta.json", "w"))
    sheet = [cv2.resize(cv2.imread(f"{UJI}/images/{m['name']}.jpg"), (480, 270)) for m in meta[:24]]
    cv2.imwrite(f"{UJI}/contoh.jpg", np.vstack([np.hstack(sheet[i:i + 4]) for i in range(0, 24, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("set uji:", len(meta), "foto,", sum(len(m["pasted"]) for m in meta), "objek tempel", flush=True)
    return meta


def evaluate(pt, meta):
    from ultralytics import YOLO
    m = YOLO(pt); tag = os.path.basename(os.path.dirname(os.path.dirname(pt))) if "weights" in pt else os.path.basename(pt)[:-3]
    L = [f"== {tag} ({pt})"]
    # A, B
    acc, nb, n, bot_ok = 0, 0, 0, 0; percat = {}
    for mm in meta:
        r = m.predict(f"{UJI}/images/{mm['name']}.jpg", imgsz=640, conf=CONF, verbose=False)[0]
        dets = [(int(b.cls), b.xyxy[0].tolist()) for b in r.boxes]
        bot_ok += any(c == 0 and iou(d, mm["bottle"]) >= 0.5 for c, d in dets)
        for p in mm["pasted"]:
            n += 1; a = any(c == 0 and iou(d, p["box"]) >= 0.3 for c, d in dets); acc += a
            nb += any(c == 1 and iou(d, p["box"]) >= 0.3 for c, d in dets)
            pc = percat.setdefault(p["cat"], [0, 0]); pc[0] += 1; pc[1] += a
    L.append(f"A. objek TACO-tempel DITERIMA sebagai botol: {acc}/{n} = {100 * acc / n:.1f}%  (dikenali non_bottle: {nb}/{n})")
    L.append("   per kategori (diterima/total): " + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(percat.items(), key=lambda kv: -kv[1][0])))
    L.append(f"B. botol asli di foto bertempel tetap terdeteksi: {bot_ok}/{len(meta)}")
    # C
    fs = sorted(glob.glob(f"{DS}/1_foto_mentah/2026-10-20_water_v1/b*.png")); ok = 0
    for f in fs:
        r = m.predict(f, imgsz=640, conf=CONF, verbose=False)[0]; ok += any(int(b.cls) == 0 for b in r.boxes)
    L.append(f"C. sesi 20 Okt asli: botol terdeteksi {ok}/{len(fs)}")
    # D
    val = f"{OUT}/old_test.yaml"
    if not os.path.exists(val):
        lst = [f"{DD}/B2/images/{r['name']}.jpg" for r in csv.DictReader(open(f"{DD}/det_index.csv")) if r["tag"] == "old_test"]
        open(f"{OUT}/old_test.txt", "w").write("\n".join(lst) + "\n")
        open(val, "w").write(f"train: {OUT}/old_test.txt\nval: {OUT}/old_test.txt\nnames:\n  0: bottle\n  1: non_bottle\n")
    v = m.val(data=val, imgsz=640, batch=16, conf=0.001, plots=False, verbose=False, workers=0, project=f"{OUT}/_val", name=tag, exist_ok=True)
    ap50 = v.box.ap50; rec = v.box.r
    L.append(f"D. old_test (284 foto nyata): mAP50 bottle {ap50[0]:.3f}, non_bottle {ap50[1]:.3f} | recall bottle {rec[0]:.3f}, non_bottle {rec[1]:.3f}")
    # E
    fs = sorted(glob.glob(f"{DS}/1_foto_mentah/2026-09-06_intake/nonbottle/*.jpg")); acc_e = 0
    for f in fs:
        r = m.predict(f, imgsz=640, conf=CONF, verbose=False)[0]; acc_e += any(int(b.cls) == 0 for b in r.boxes)
    L.append(f"E. foto nyata non-botol (intake, alas putih): diterima sebagai botol {acc_e}/{len(fs)}")
    open(f"{OUT}/hasil_uji_{tag}.txt", "w", encoding="utf-8").write("\n".join(L) + "\n")
    print("\n".join(L), flush=True)


def bottle_cutouts(belts):
    """ver9b: botol asli sebagai potongan ber-alpha (cara tempel SAMA dengan negatif TACO) agar tepi tempelan tidak informatif.
    Sumber: botol belt latih (C13/W16/P15, mask elips berhalus) + botol studio bernomor GENAP (ganjil disimpan untuk uji)."""
    pool = []
    for n, im, boxes in belts:
        for c, (x1, y1, x2, y2) in boxes:
            if c != 0:
                continue
            x1, y1, x2, y2 = int(max(0, x1)), int(max(0, y1)), int(x2), int(y2); cc = im[y1:y2, x1:x2].copy(); h, w = cc.shape[:2]
            if h < 20 or w < 20:
                continue
            a = np.zeros((h, w), np.uint8); cv2.ellipse(a, (w // 2, h // 2), (int(w * 0.52), int(h * 0.56)), 0, 0, 360, 255, -1)
            pool.append(dict(rgba=np.dstack([cc, cv2.GaussianBlur(a, (9, 9), 0)])))
    for f in sorted(glob.glob(f"{DS}/_arsip/eksperimen_cap_lama/cap_shoot_v1_cutout/nohand/b*.png")):
        if int(os.path.basename(f)[1:3]) % 2 == 0:
            s = cv2.imread(f, cv2.IMREAD_UNCHANGED)
            if s is not None and s.shape[2] == 4:
                pool.append(dict(rgba=s))
    return pool


def trashnet_objects(split):
    """Potongan TrashNet utuh (rembg). split 'latih' = nomor % 5 != 0, 'uji' = nomor % 5 == 0.
    Kembalikan (negatif, botol_pet): negatif = semua kelas kecuali plastic yang dipilah PET (trashnet_pilah.csv, pilah visual Claude)."""
    T = f"{DS}/5_data_luar/_dl_trashnet"
    pil = {int(r["nomor"]): r["pet"] for r in csv.DictReader(open(f"{T}/trashnet_pilah.csv"))}
    pet = {n for n, v in pil.items() if v == "1"}; ragu = {n for n, v in pil.items() if v == "x"}   # ragu (HDPE, botol isi ulang, dll.) dibuang
    neg, pos = [], []
    for f in sorted(glob.glob(f"{T}/cutout/*/*.png")):
        c = os.path.basename(os.path.dirname(f)); n = os.path.basename(f)[:-4]; num = int(n.replace(c, ""))
        if (num % 5 == 0) != (split == "uji") or (c == "plastic" and num in ragu):
            continue
        rgba = cv2.imread(f, cv2.IMREAD_UNCHANGED)
        if rgba is None or rgba.ndim != 3 or rgba.shape[2] != 4:
            continue
        o = dict(rgba=rgba, cat=f"trashnet_{c}" + ("_PET" if c == "plastic" and num in pet else ""), img=n)
        (pos if c == "plastic" and num in pet else neg).append(o)
    return neg, pos


def build_latih(variant="a"):
    from ultralytics import YOLO  # noqa: F401
    global LAT
    LAT = f"{OUT}/latih_taco" + ("" if variant == "a" else f"_{variant}")
    shutil.rmtree(LAT, ignore_errors=True); os.makedirs(f"{LAT}/images"); os.makedirs(f"{LAT}/labels")
    objs = load_objects()["latih"]; rng = random.Random(3)
    print("objek latih:", len(objs), flush=True)
    idx = [r["name"] for r in csv.DictReader(open(f"{DD}/det_index.csv")) if r["tag"].split("_")[0] in ("C13", "P15") or
           (r["tag"].startswith("W16_f") and not r["tag"].endswith("_q"))]
    belts = []
    for n in idx:
        im = cv2.imread(f"{DD}/B2/images/{n}.jpg"); H, W = im.shape[:2]
        lab = [l.split() for l in open(f"{DD}/B2/labels/{n}.txt").read().strip().splitlines()]
        boxes = [(int(c), ((float(x) - float(w) / 2) * W, (float(y) - float(h) / 2) * H, (float(x) + float(w) / 2) * W, (float(y) + float(h) / 2) * H))
                 for c, x, y, w, h in lab]
        belts.append((n, im, boxes))
    bpool = bottle_cutouts(belts) if variant in ("b", "c") else None
    tneg, tpos = trashnet_objects("latih") if variant == "c" else ([], [])
    if bpool:
        print("potongan botol ber-alpha:", len(bpool), "| TrashNet negatif", len(tneg), "PET", len(tpos), flush=True)

    def pick_neg():
        return tneg[rng.randrange(len(tneg))] if tneg and rng.random() < 0.5 else objs[rng.randrange(len(objs))]

    def pick_pos():
        return tpos[rng.randrange(len(tpos))] if tpos and rng.random() < 0.3 else bpool[rng.randrange(len(bpool))]
    k = 0
    for rep in range(2):
        for n, im0, boxes in belts:
            im = im0.copy(); H, W = im.shape[:2]
            taken = [(b[0] - 10, b[1] - 10, b[2] + 10, b[3] + 10) for _, b in boxes]; lines = [yolo_line(c, b, W, H) for c, b in boxes]
            for _ in range(rng.choice([1, 2, 2])):
                b = paste_object(im, pick_neg(), taken, rng); k += 1
                if b:
                    taken.append((b[0] - 10, b[1] - 10, b[2] + 10, b[3] + 10)); lines.append(yolo_line(1, b, W, H))
            if bpool:                                                  # ver9b: jumlah tempelan botol ~ jumlah tempelan negatif
                for _ in range(rng.choice([1, 2, 2])):
                    b = paste_object(im, pick_pos(), taken, rng)
                    if b:
                        taken.append((b[0] - 10, b[1] - 10, b[2] + 10, b[3] + 10)); lines.append(yolo_line(0, b, W, H))
            elif rng.random() < 0.5:
                sn, sim, sb = belts[rng.randrange(len(belts))]; bb = [b for c, b in sb if c == 0]
                if bb:
                    res = paste_bottle(im, sim, bb[0], taken, rng)
                    if res:
                        box, area = res; taken.append(area); lines.append(yolo_line(0, box, W, H))
            cv2.imwrite(f"{LAT}/images/t{rep}_{n}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
            open(f"{LAT}/labels/t{rep}_{n}.txt", "w").write("\n".join(lines) + "\n")
    fs = sorted(glob.glob(f"{LAT}/images/*.jpg"))
    sheet = []
    for f in fs[:: max(1, len(fs) // 12)][:12]:
        im = cv2.imread(f); H, W = im.shape[:2]
        for l in open(f.replace("images", "labels")[:-4] + ".txt").read().split("\n"):
            if l.strip():
                c, x, y, w, h = map(float, l.split())
                cv2.rectangle(im, (int((x - w / 2) * W), int((y - h / 2) * H)), (int((x + w / 2) * W), int((y + h / 2) * H)), (0, 0, 255) if c else (0, 255, 0), 2)
        sheet.append(cv2.resize(im, (480, 270)))
    cv2.imwrite(f"{LAT}/contoh.jpg", np.vstack([np.hstack(sheet[i:i + 4]) for i in range(0, 12, 4)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print("data latih sintetis:", len(fs), "foto", flush=True)
    return fs


def train(fs, name="ver9"):
    from ultralytics import YOLO
    old = [l.strip().replace("\\", "/").split("/det_data/")[-1] for l in open(f"{DS}/4_eksperimen/cairan/final2/ver8b_train.txt") if l.strip()]
    tr = [f"{DD}/{p}" for p in old]
    assert all(os.path.exists(p) for p in tr[:50]), "path data ver8b tidak ditemukan"
    tr += [f.replace("\\", "/") for f in fs]
    open(f"{OUT}/{name}_train.txt", "w").write("\n".join(tr) + "\n")
    open(f"{OUT}/{name}.yaml", "w").write(f"train: {OUT}/{name}_train.txt\nval: {OUT}/old_test.txt\nnames:\n  0: bottle\n  1: non_bottle\n")
    print(f"latih {name}:", len(tr), "foto (ver8b", len(old), "+ sintetis", len(fs), ")", flush=True)
    YOLO(V7).train(data=f"{OUT}/{name}.yaml", epochs=10, imgsz=640, batch=16, workers=0, seed=0, deterministic=True, patience=0, plots=False,
                   verbose=False, project=OUT, name=name, exist_ok=True, hsv_h=0.0, hsv_s=0.3, hsv_v=0.3, degrees=0, fliplr=0.5,
                   flipud=0.0, mosaic=1.0, close_mosaic=3)
    return f"{OUT}/{name}/weights/last.pt"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "uji"
    meta = build_uji()
    if cmd == "uji":
        for pt in (sys.argv[2:] or [V7, V8B]):
            evaluate(pt, meta)
    elif cmd == "latih":
        evaluate(train(build_latih("a"), "ver9"), meta)
    elif cmd == "latih_c":                                             # ver9c: ver9b + potongan TrashNet (PET = bottle; kaca/kaleng/kertas/plastik lain = non_bottle)
        evaluate(train(build_latih("c"), "ver9c"), meta)
    elif cmd == "latih_b":                                             # ver9b: + botol ber-alpha (mitigasi jalan pintas tempelan)
        evaluate(train(build_latih("b"), "ver9b"), meta)
