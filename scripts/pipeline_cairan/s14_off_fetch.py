"""Langkah 14: kumpulkan foto depan produk minuman dari Open Food Facts (CC BY-SA; wajib atribusi) untuk data luar.
Label dari JENIS minuman (bukan dari gambar; harus divalidasi manusia):
  berwarna = cola/Fanta/Mirinda/Pepsi/teh/jus/isotonik berwarna | tidak = air putih/air soda bening/Sprite/7up/tonic (cairan bening)
Kutipan sopan: header UA mengenalkan diri, jeda 0.3 s. Hanya gambar 400 px 'image_front_url'.
Keluaran: _dl_ext/off/images/<code>.jpg + _dl_ext/off/candidates.csv"""
import csv, json, os, time, urllib.parse, urllib.request

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
D = os.path.join(ROOT, "5_data_luar/_dl_ext", "off"); os.makedirs(f"{D}/images", exist_ok=True)
UA = {"User-Agent": "KP-RVM-UGM/1.0 (academic internship dataset research)"}
POS = ["coca cola", "coca cola zero", "coca cola vanilla", "fanta orange", "fanta strawberry", "fanta grape", "fanta", "pepsi", "pepsi max", "mirinda",
       "orange soda", "cola", "dr pepper", "ice tea peach", "ice tea lemon", "green tea", "nestea", "lipton ice tea", "fuze tea", "teh botol", "orange juice",
       "apple juice", "cranberry juice", "tropicana", "pineapple juice", "grape juice", "isotonic drink", "gatorade", "powerade", "pocari sweat", "fruit punch",
       "cream soda", "ginger ale", "cherry cola", "strawberry drink", "mango drink", "lychee drink"]
NEG = ["sprite", "7up", "mineral water", "spring water", "sparkling water", "still water", "tonic water", "lemon lime soda", "aqua", "evian", "volvic",
       "vittel", "cristaline", "pure life water", "water 1.5l", "water 500ml", "le minerale", "club soda", "soda water", "bottled water"]
seen = {}
out = f"{D}/candidates.csv"
if os.path.exists(out):
    for r in csv.DictReader(open(out)):
        seen[r["code"]] = r
def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30).read()
rows = list(seen.values())
for label, qs in (("berwarna", POS), ("tidak", NEG)):
    for q in qs:
        for page in (1, 2):
            try:
                js = json.loads(get("https://search.openfoodfacts.org/search?" + urllib.parse.urlencode({"q": q, "page": page, "page_size": 100, "fields": "code,product_name,brands,quantity,image_front_url"})))
            except Exception as e:
                print("gagal", q, page, e); time.sleep(5); continue
            new = 0
            for h in js.get("hits", []):
                c, u = h.get("code"), h.get("image_front_url")
                if not c or not u or c in seen:
                    continue
                try:
                    data = get(u)
                except Exception:
                    continue
                open(f"{D}/images/{c}.jpg", "wb").write(data)
                r = dict(code=c, label=label, query=q, name=h.get("product_name") or "", brands=",".join(h.get("brands") or []) if isinstance(h.get("brands"), list) else (h.get("brands") or ""),
                         quantity=h.get("quantity") or "", url=u)
                seen[c] = r; rows.append(r); new += 1; time.sleep(0.3)
            print(f"{label} '{q}' p{page}: +{new} (total {len(rows)})", flush=True)
            with open(out, "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=["code", "label", "query", "name", "brands", "quantity", "url"]); w.writeheader(); w.writerows(rows)
print("selesai", len(rows))
