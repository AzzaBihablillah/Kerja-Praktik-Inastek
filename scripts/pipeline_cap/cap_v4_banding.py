"""Nilai model tutup v4w (dilatih di Colab, diunduh ke _cap_v4w/runs/cv_*/weights/last.pt) pada hari yang TIDAK dilatih, dengan
cara & foto yang sama dengan cap_v3_belt.evaluate, lalu bandingkan dengan v3 (aktif) per KELOMPOK TUTUP:
  asli     = bertutup, tutup sewarna cincin (tidak ditukar)   -> kriteria 3: v4w tidak boleh lebih buruk
  campuran = sesi tutup dicampur/ditukar (29 Okt 110,111,112,114) + 2 tutup pengganti biru 27 Okt (S51B96, S53B195)
  tanpa    = tanpa tutup                                       -> kriteria 2: terbaca bertutup tidak boleh bertambah
  python scripts/pipeline_cap/cap_v4_banding.py   -> _cap_v4w/cv_records.jsonl + banding_report.txt"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cv_cap_v2 as v2, cap_v3_belt as c3, cap_v4_warna as c4

CAMPUR29 = {"110", "111", "112", "114"}; GANTI27 = {"D51B96", "D53B195"}


def kelompok(r):
    if r["cls"] == 1:
        return "tanpa"
    if (r["hari"] == "29" and r["sesi"] in CAMPUR29) or (r["hari"] == "27" and r["botol"] in GANTI27):
        return "campuran"
    return "asli"


def main():
    c3.OUT = c4.OUT; v2.CV = c4.OUT
    new = c3.new_items(); test = {"A29": [i for i in new if i["hari"] == "29"], "B27": [i for i in new if i["hari"] == "27"]}
    res = f"{c4.OUT}/cv_records.jsonl"
    done = {(json.loads(l)["fold"], json.loads(l)["model"]) for l in open(res)} if os.path.exists(res) else set()
    for fold in ("A29", "B27"):
        pt = f"{c4.OUT}/runs/cv_{fold}/weights/last.pt"
        if (fold, "v4w") in done or not os.path.exists(pt):
            continue
        recs = c3.evaluate(pt, test[fold], "v4w", fold)
        with open(res, "a") as fh:
            for r in recs:
                fh.write(json.dumps(r) + "\n")
        print("dinilai", fold, len(recs), "botol", flush=True)
    R = [json.loads(l) for l in open(f"{v2.DS}/4_eksperimen/cap/_cap_v3/cv_records.jsonl")] + \
        ([json.loads(l) for l in open(res)] if os.path.exists(res) else [])
    L = ["Tutup v4w (warna buatan) vs v3 (aktif) - per botol, mayoritas 3 frame, hari uji TIDAK dilatih.",
         "Angka = benar/total. asli & campuran: tutup terdeteksi; tanpa: terbaca tanpa tutup.", ""]
    for fold, nm in (("A29", "uji 29 Okt"), ("B27", "uji 27 Okt")):
        L.append(f"== {nm}")
        for mode in ("roi", "full"):
            for m in ("v2_aktif", "v3", "v4w"):
                rr = [r for r in R if r["fold"] == fold and r["model"] == m]
                if not rr:
                    continue
                parts = []
                for k in ("asli", "campuran", "tanpa"):
                    g = [r for r in rr if kelompok(r) == k]
                    if g:
                        parts.append(f"{k} {sum(r[mode + '_ok'] for r in g)}/{len(g)}")
                L.append(f"  {mode:4s} {m:8s} " + " | ".join(parts))
        L.append("")
    open(f"{c4.OUT}/banding_report.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))


if __name__ == "__main__":
    main()
