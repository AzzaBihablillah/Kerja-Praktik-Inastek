"""Model tutup v4: resep IDENTIK cap_v3_belt (data lama + belt 27/29 Okt, 30 epoch, imgsz 960, validasi PER HARI), ditambah
SATU salinan berwarna buatan per foto latih (scripts/pipeline_cap/warna_tutup.py):
  kelas cap    : 70% tutup diwarnai ulang (warna_tutup.mask_tutup3: mengikuti KROMA tutup asli, kecerahan/bayangan asli dipertahankan;
                 cincin TETAP) -> tutup beda warna dgn cincin; tutup hitam/belang/arah ragu dilewati (tanpa salinan); 30% semua digeser
  (versi 1 = garis potong lurus, ditolak audit user 30 Okt: tutup separuh berwarna, salah ujung, belang -> disimpan di aug/)
  kelas no_cap : cincin/leher digeser warnanya
Hanya piksel benda (bukan latar belt) yang diwarnai. Tujuan: memutus jalan pintas 'tutup sewarna cincin' (kegagalan tutup biru &
tutup tukaran, CATATAN L-32). Foto UJI tidak pernah diwarnai.
Kriteria lulus DITETAPKAN SEBELUM hasil (30 Okt, disetujui user): di KEDUA hari uji, dibanding v3 pada simulasi keputusan,
salah terima lebih sedikit DAN salah tolak tidak bertambah; tanpa-tutup-terbaca-tutup (per botol, ROI) tidak bertambah.
  Tambahan (user 30 Okt, sebelum hasil): pada botol bertutup ASLI (tutup sewarna cincin, tidak ditukar) deteksi tutup v4w TIDAK BOLEH
  lebih buruk dari v3 - kasus umum tidak boleh dikorbankan demi kasus khusus. Sesi tutup asli: 29 Okt 106, 113, 115, 116, 118, 119;
  27 Okt semua sesi bertutup kecuali 2 tutup pengganti biru (S51B96, S53B195). Sesi tutup campuran (110, 111, 112, 114) dinilai terpisah.
  python scripts/pipeline_cap/cap_v4_warna.py        -> datasets/4_eksperimen/cap/_cap_v4w/ (cv_records.jsonl, cv_report.txt)
  python scripts/pipeline_cap/cap_v4_warna.py final  -> datasets/3_model_final/cap_belt_v4w_kandidat.pt (hanya bila lulus)"""
import json, os, shutil, sys
from multiprocessing import freeze_support

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_cap_v2 as v2
import cap_v3_belt as c3
import warna_tutup as wt

DS = v2.DS
OUT = f"{DS}/4_eksperimen/cap/_cap_v4w"; AUG = f"{OUT}/aug2"   # aug2 = pewarnaan v3 (mengikuti warna tutup); aug = versi 1 (ditolak audit user)


def augment(its, seed=0):
    """Satu salinan berwarna per item; disimpan (cache) di AUG. Item tanpa kotak botol -> tanpa salinan."""
    import cv2, numpy as np
    from ultralytics import YOLO
    os.makedirs(f"{AUG}/images", exist_ok=True); os.makedirs(f"{AUG}/labels", exist_ok=True)
    det = None; out = []
    MODE = json.load(open(f"{AUG}/mode.json")) if os.path.exists(f"{AUG}/mode.json") else {}
    BUANG = set(json.load(open(f"{AUG}/audit_buang.json"))["buang"]) if os.path.exists(f"{AUG}/audit_buang.json") else set()
    for i, it in enumerate(its):
        name = it["name"] + "_w"; ip, lp = f"{AUG}/images/{name}.jpg", f"{AUG}/labels/{name}.txt"
        if MODE.get(it["name"], "").startswith("lewat"):   # sudah diputuskan dilewati: JANGAN dibuat ulang (k-means tidak deterministik)
            continue
        if not os.path.exists(ip):
            if os.path.exists(f"{AUG}/audit_buang.json"):
                continue                              # setelah audit: tidak ada salinan baru yang belum diperiksa user
            det = det or YOLO(f"{DS}/3_model_final/botol_ver9c.pt")
            im = cv2.imread(it["img"]); H, W = im.shape[:2]
            c, cx, cy, w, h = map(float, open(it["lbl"]).read().split()[:5])
            neck = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
            ov = lambda b: max(0, min(b[2], neck[2]) - max(b[0], neck[0])) * max(0, min(b[3], neck[3]) - max(b[1], neck[1]))
            bb = [b.xyxy[0].tolist() for b in det.predict(im, imgsz=640, conf=0.25, verbose=False)[0].boxes if int(b.cls) == 0]
            bb = [b for b in bb if ov(b) > 0]
            if not bb:
                continue
            rng = np.random.default_rng(seed * 100003 + i)
            mb = wt.mask_benda(im, neck)
            if it["cls"] == 0 and rng.random() < 0.7:
                m, why = wt.mask_tutup3(im, neck, max(bb, key=ov), mb)
                if m is None:                       # tidak layak diwarnai (gelap/belang/arah) -> TANPA salinan
                    MODE[it["name"]] = "lewat_" + why; continue
                im2 = wt.warnai2(im, m, rng); MODE[it["name"]] = "warna"
            else:
                im2 = wt.geser_benda(im, mb, rng); MODE[it["name"]] = "geser" if it["cls"] == 0 else "nocap"
            cv2.imwrite(ip, im2, [cv2.IMWRITE_JPEG_QUALITY, 95]); shutil.copy2(it["lbl"], lp)
        if it["name"] in BUANG:                     # dibuang user di audit (halaman audit warna tutup, versi 2)
            continue
        out.append(dict(it, name=name, img=ip, lbl=lp))
    json.dump(MODE, open(f"{AUG}/mode.json", "w"), indent=0)
    return out


def latih_atau_lanjut(d, name):
    """Laptop bisa dimatikan di tengah latih: bila last.pt ada dan latih BELUM selesai (epoch != -1), lanjutkan (resume)
    dari epoch terakhir yang tersimpan; resep & data identik (folder data dibangun ulang dengan isi sama)."""
    last = f"{OUT}/runs/{name}/weights/last.pt"
    if os.path.exists(last):
        import torch
        ep = torch.load(last, map_location="cpu", weights_only=False).get("epoch", -1)
        if ep != -1:
            from ultralytics import YOLO
            print(f"{name}: LANJUT dari epoch {ep + 1}", flush=True)
            YOLO(last).train(resume=True)
            return last
    return v2.train(d, name)


def main():
    c3.OUT = OUT; v2.CV = OUT; os.makedirs(OUT, exist_ok=True)
    old = v2.items(); new = c3.new_items()
    n27 = [i for i in new if i["hari"] == "27"]; n29 = [i for i in new if i["hari"] == "29"]
    if len(sys.argv) > 1 and sys.argv[1] == "final":
        tr = old + new; d = f"{OUT}/data_final"; c3.build(d, tr + augment(tr)); pt = v2.train(d, "final")
        shutil.copy2(pt, f"{DS}/3_model_final/cap_belt_v4w_kandidat.pt"); shutil.rmtree(d, ignore_errors=True)
        print("FINAL ->", f"{DS}/3_model_final/cap_belt_v4w_kandidat.pt"); return
    res = f"{OUT}/cv_records.jsonl"
    done = {(json.loads(l)["fold"], json.loads(l)["model"]) for l in open(res)} if os.path.exists(res) else set()
    only = os.environ.get("FOLD")
    for fold, test, tr in (("A29", n29, old + n27), ("B27", n27, old + n29)):
        if (only and fold != only) or (fold, "v4w") in done:
            continue
        d = f"{OUT}/data_{fold}"; aug = augment(tr); c3.build(d, tr + aug)
        print(f"{fold}: latih {len(tr)} + {len(aug)} berwarna, uji {len(test)} frame", flush=True)
        recs = c3.evaluate(latih_atau_lanjut(d, f"cv_{fold}"), test, "v4w", fold); shutil.rmtree(d, ignore_errors=True)
        with open(res, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    freeze_support()
    main()
