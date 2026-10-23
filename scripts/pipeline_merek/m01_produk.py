"""Model merek (bonus poin produk mitra = SEMUA produk Coca-Cola Company, keputusan user 23 Okt), langkah 1: tabel produk.
Sumber: foto belt kita sendiri (13, 16, 20, 22 Okt + frame diam aplikasi). Merek dari NAMA FILE / kelompok lembar cek.
Botol isi ulang dinilai dari BOTOLNYA (yang dilihat kamera = label botol): botol Aqua isi cola = bukan mitra.
Label diusulkan Claude per PRODUK lalu divalidasi user (halaman validasi). Keluaran: datasets/4_eksperimen/merek/produk.csv"""
import csv, glob, os, re, sys
ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
OUT = f"{ROOT}/4_eksperimen/merek"; os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "pipeline_cairan"))
# token nama file -> (kunci produk, usulan)   M = mitra (Coca-Cola Company), B = bukan, ? = ragu
P = [(r"lemin", "le_minerale", "B"), (r"crystalin", "crystalline", "B"), (r"superindomerah", "superindo_merah", "B"), (r"superindo", "superindo", "B"),
     (r"aqua(?!viva)", "aqua", "B"), (r"pristine", "pristine", "B"), (r"airalam", "air_alam", "B"), (r"prima", "prima", "B"),
     (r"purelife", "pure_life", "B"), (r"bigstrawberry", "big_strawberry", "B"), (r"biglemonlime", "big_lemonlime", "B"), (r"bigcola", "big_cola", "B"),
     (r"bignipis", "big_nipis_madu", "B"), (r"cheers", "cheers", "B"), (r"frestea", "frestea", "M"), (r"cleo", "cleo", "B"), (r"pocari", "pocari", "B"),
     (r"clairine", "clairine", "B"), (r"isoplus", "isoplus", "B"), (r"vica", "vica", "B"), (r"vit\d", "vit", "B"), (r"natsbee", "natsbee", "B"),
     (r"ichitan", "ichitan", "B"), (r"teh-?pucuk", "teh_pucuk", "B"), (r"tropical", "tropical", "B"), (r"aquviva", "aquviva", "B"),
     (r"sprite", "sprite", "M"), (r"cola-zero|cocacola-zero", "coca_cola_zero", "M"), (r"cola-kecil", "coca_cola", "M"), (r"fanta", "fanta", "M"),
     (r"powerade", "powerade", "M"), (r"minute-?maid", "minute_maid", "M"), (r"nutriboost", "minute_maid_nutriboost", "M"), (r"sunbay", "sunbay", "B"),
     (r"mizone", "mizone", "B")]
SCAN = {"S22_fanta": ("fanta", "M"), "S22_pocari": ("pocari", "B"), "S22_aqua_cola": ("aqua", "B"), "S22_crystalin": ("crystalline", "B"),
        "S22_leminerale": ("le_minerale", "B"), "S22_leminerale2": ("le_minerale", "B"), "S22_cokezero": ("coca_cola_zero", "M"),
        "S22_cheers": ("cheers", "B"), "S22_cheers2": ("cheers", "B"), "S22_botol17": ("botol_17_tak_dikenal", "?"), "S22_sprite": ("sprite", "M"),
        "S22_sprite_kecil": ("sprite", "M"), "S22_fitri": ("fitri", "B"), "S22_coke": ("coca_cola", "M"), "S22_frestea": ("frestea", "M"), "S22_aqua": ("aqua", "B")}


def produk_of(name):
    tok = name.split("_")[1]
    for pat, key, lab in P:
        if re.search(pat, tok):
            return key, lab
    return tok, "?"


def main():
    from s31_data_22okt import GROUP_SCAN
    rows = []
    for s in ("2026-10-13_conveyor_v1", "2026-10-16_water_v1", "2026-10-20_water_v1", "2026-10-22_water_v1"):
        for f in sorted(glob.glob(f"{ROOT}/1_foto_mentah/{s}/b*.png")):
            n = os.path.basename(f); key, lab = produk_of(n)
            rows.append(dict(path=os.path.relpath(f, ROOT), sesi=s[5:10], botol=n.split("_")[0], produk=key, usul=lab))
    for i, r in enumerate(csv.DictReader(open(f"{ROOT}/1_foto_mentah/2026-10-22_scan_app/SCAN.csv")), 1):
        key, lab = SCAN[GROUP_SCAN[i]]
        for k in (1, 2, 3):
            f = f"{ROOT}/1_foto_mentah/2026-10-22_scan_app/s{r['sesi']}_b{r['botol']}_{r['waktu']}_{k}.png"
            if os.path.exists(f):
                rows.append(dict(path=os.path.relpath(f, ROOT), sesi="scan", botol=f"#{i}", produk=key, usul=lab))
    with open(f"{OUT}/foto.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    from collections import Counter, defaultdict
    per = defaultdict(list)
    for r in rows:
        per[r["produk"]].append(r)
    prod = [dict(produk=k, usul=v[0]["usul"], n_foto=len(v), n_botol=len({(x["sesi"], x["botol"]) for x in v}), contoh=v[len(v) // 2]["path"]) for k, v in sorted(per.items())]
    with open(f"{OUT}/produk.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prod[0])); w.writeheader(); w.writerows(prod)
    print(len(rows), "foto,", len(prod), "produk;", Counter(p["usul"] for p in prod), "| foto mitra:", sum(r["usul"] == "M" for r in rows))
    for p in prod:
        print(f"  {p['usul']} {p['produk']:24s} {p['n_foto']:4d} foto {p['n_botol']:3d} botol")


if __name__ == "__main__":
    main()
