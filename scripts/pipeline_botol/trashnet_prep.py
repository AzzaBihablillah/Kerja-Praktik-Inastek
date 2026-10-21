"""TrashNet (MIT; Thung & Yang 2016) -> data deteksi botol/non-botol. Satu objek per foto di alas putih (512x384).
Kotak objek: pengurangan latar (latar = median tepi foto; piksel berbeda jauh = objek), komponen terbesar.
Pemilahan kelas: 'plastic' berisi botol PET DAN plastik lain -> lembar kontak bernomor (dengan tebakan botol_ver8b) untuk
dipilah visual; hasil pilah ditulis tangan ke trashnet_pilah.csv. Kelas lain (glass, paper, cardboard, metal, trash) = non_bottle
(botol KACA = non_bottle: bukan PET).
  python -I scripts/pipeline_botol/trashnet_prep.py lembar    -> boxes.json + lembar_plastic_XX.jpg
  python -I scripts/pipeline_botol/trashnet_prep.py bangun    -> dataset YOLO (images/labels) + split.csv"""
import csv, glob, json, os, sys
import cv2, numpy as np

DS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
T = f"{DS}/5_data_luar/_dl_trashnet"; RAW = f"{T}/raw/dataset-resized"; OUT = f"{T}/yolo"
KELAS = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]


def box_of(im):
    h, w = im.shape[:2]; lab = cv2.cvtColor(cv2.GaussianBlur(im, (5, 5), 0), cv2.COLOR_BGR2LAB).astype(np.float32)
    border = np.concatenate([lab[:6].reshape(-1, 3), lab[-6:].reshape(-1, 3), lab[:, :6].reshape(-1, 3), lab[:, -6:].reshape(-1, 3)])
    bg = np.median(border, 0); d = np.linalg.norm(lab - bg, axis=2)
    thr = max(18.0, np.percentile(np.linalg.norm(border - bg, axis=1), 99) * 1.5)
    m = (d > thr).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)); m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lb, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); x, y, bw, bh, area = st[k]
    if area < 0.01 * w * h:
        return None
    return [int(x), int(y), int(x + bw), int(y + bh)]


def lembar():
    from ultralytics import YOLO
    det = YOLO(f"{DS}/3_model_final/botol_ver8b.pt"); boxes = {}
    for c in KELAS:
        for f in sorted(glob.glob(f"{RAW}/{c}/*.jpg")):
            im = cv2.imread(f); b = box_of(im)
            r = det.predict(im, imgsz=640, conf=0.25, verbose=False)[0]
            bot = max([float(z.conf) for z in r.boxes if int(z.cls) == 0], default=0.0)
            boxes[os.path.basename(f)] = dict(kelas=c, box=b, ver8b_bottle=round(bot, 2))
    json.dump(boxes, open(f"{T}/boxes.json", "w"), indent=0)
    pl = sorted(n for n, v in boxes.items() if v["kelas"] == "plastic")
    for s in range(0, len(pl), 48):
        tiles = []
        for i, n in enumerate(pl[s:s + 48]):
            im = cv2.imread(f"{RAW}/plastic/{n}"); v = boxes[n]
            if v["box"]:
                cv2.rectangle(im, tuple(v["box"][:2]), tuple(v["box"][2:]), (0, 0, 255), 3)
            t = cv2.resize(im, (256, 192)); num = n.replace("plastic", "").replace(".jpg", "")
            cv2.rectangle(t, (0, 0), (256, 22), (0, 0, 0), -1)
            cv2.putText(t, f"{num}  v8b {v['ver8b_bottle']:.2f}", (4, 16), 0, 0.5, (0, 255, 255), 1); tiles.append(t)
        while len(tiles) % 8:
            tiles.append(np.zeros_like(tiles[0]))
        cv2.imwrite(f"{T}/lembar_plastic_{s // 48 + 1:02d}.jpg", np.vstack([np.hstack(tiles[j:j + 8]) for j in range(0, len(tiles), 8)]),
                    [cv2.IMWRITE_JPEG_QUALITY, 85])
    nb = sum(v["box"] is None for v in boxes.values())
    print(f"{len(boxes)} foto, tanpa kotak {nb}; lembar plastic: {(len(pl) + 47) // 48}")


def bangun():
    boxes = json.load(open(f"{T}/boxes.json"))
    pet = {int(r["nomor"]) for r in csv.DictReader(open(f"{T}/trashnet_pilah.csv")) if r["pet"] == "1"}
    for s in ("images", "labels"):
        os.makedirs(f"{OUT}/{s}", exist_ok=True)
    rows = []
    for n, v in sorted(boxes.items()):
        if v["box"] is None:
            continue
        num = int(n.replace(v["kelas"], "").replace(".jpg", ""))
        cls = 0 if (v["kelas"] == "plastic" and num in pet) else 1
        im = cv2.imread(f"{RAW}/{v['kelas']}/{n}"); H, W = im.shape[:2]; x1, y1, x2, y2 = v["box"]
        name = f"tn_{n[:-4]}"
        cv2.imwrite(f"{OUT}/images/{name}.jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 92])
        open(f"{OUT}/labels/{name}.txt", "w").write(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}\n")
        rows.append(dict(name=name, kelas=v["kelas"], label=["bottle", "non_bottle"][cls], uji=int(num % 5 == 0)))
    with open(f"{OUT}/split.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(len(rows), "foto; bottle", sum(r["label"] == "bottle" for r in rows), "; uji", sum(r["uji"] for r in rows))


def potong():
    """Potong objek dengan rembg (u2net, model lokal ~/.rembg/models/u2net). Simpan hanya objek UTUH (mask tidak menyentuh tepi foto),
    luas 3-60% foto. Keluaran: cutout/<kelas>/<nama>.png (RGBA) + lembar_pilah_XX.jpg (plastic & glass bernomor, untuk pilah PET)."""
    os.environ.setdefault("U2NET_HOME", os.path.expanduser("~/.rembg/models/u2net"))
    from rembg import new_session, remove
    ses = new_session("u2net"); keep = {}
    for c in ["plastic", "metal", "trash", "glass", "paper", "cardboard"]:               # prioritas: paling mirip botol dulu
        od = f"{T}/cutout/{c}"; os.makedirs(od, exist_ok=True)
        fs = sorted(glob.glob(f"{RAW}/{c}/*.jpg"))
        for f in (fs[:150] if c in ("paper", "cardboard") else fs):                   # kertas/kardus: cukup 150 foto
            n = os.path.basename(f)[:-4]; dst = f"{od}/{n}.png"
            if not os.path.exists(dst):
                im = cv2.imread(f); a = remove(im, session=ses, only_mask=True)
                a = np.array(a); a = cv2.resize(a, (im.shape[1], im.shape[0])) if a.shape[:2] != im.shape[:2] else a
                m = (a > 128).astype(np.uint8)
                nl, lb, st, _ = cv2.connectedComponentsWithStats(m)
                if nl < 2:
                    continue
                k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); x, y, w, h, area = st[k]; H, W = m.shape
                if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2 or not 0.03 * W * H <= area <= 0.6 * W * H:
                    continue
                alpha = np.where(lb == k, a, 0).astype(np.uint8)
                cv2.imwrite(dst, np.dstack([im, alpha])[y:y + h, x:x + w])
            keep.setdefault(c, []).append(n)
        print(c, len(keep.get(c, [])), "utuh", flush=True)
    for c in ("plastic", "glass"):
        ns = keep.get(c, [])
        for s in range(0, len(ns), 48):
            tiles = []
            for n in ns[s:s + 48]:
                rgba = cv2.imread(f"{T}/cutout/{c}/{n}.png", cv2.IMREAD_UNCHANGED); h, w = rgba.shape[:2]; k = min(250 / w, 180 / h)
                rgba = cv2.resize(rgba, (max(1, int(w * k)), max(1, int(h * k)))); a = rgba[:, :, 3:4] / 255.0
                t = np.full((200, 256, 3), 90, np.uint8); oy, ox = 20 + (180 - rgba.shape[0]) // 2, (256 - rgba.shape[1]) // 2
                t[oy:oy + rgba.shape[0], ox:ox + rgba.shape[1]] = (rgba[:, :, :3] * a + t[oy:oy + rgba.shape[0], ox:ox + rgba.shape[1]] * (1 - a)).astype(np.uint8)
                cv2.putText(t, n.replace(c, ""), (4, 15), 0, 0.5, (0, 255, 255), 1); tiles.append(t)
            while len(tiles) % 8:
                tiles.append(np.zeros_like(tiles[0]))
            cv2.imwrite(f"{T}/lembar_pilah_{c}_{s // 48 + 1:02d}.jpg", np.vstack([np.hstack(tiles[j:j + 8]) for j in range(0, len(tiles), 8)]),
                        [cv2.IMWRITE_JPEG_QUALITY, 85])


if __name__ == "__main__":
    {"lembar": lembar, "bangun": bangun, "potong": potong}[sys.argv[1]]()
