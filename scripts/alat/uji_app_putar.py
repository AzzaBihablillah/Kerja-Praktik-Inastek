"""Uji APLIKASI UTUH (app.engine: deteksi -> berhenti-lalu-pindai -> keputusan -> bonus merek -> catatan) dengan KAMERA TIRUAN yang
memutar ulang frame diam tersimpan sesi belt 27 & 29 Okt, botol demi botol, dengan jeda belt kosong di antaranya.
AMAN dijalankan di lab: talang ESP32 (sortir) & PLC dimatikan, DB & folder simpan diarahkan ke folder sementara.
Hasil per botol dibandingkan dengan simulasi keputusan (replay_keputusan, varian set aktif = semua_baru).
  python scripts/alat/uji_app_putar.py [batas_botol]   -> datasets/4_eksperimen/uji_app/<waktu>/ (hasil.csv, ringkas.txt)"""
import os, sys, time, json, csv, datetime
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts", "alat"))
os.environ["RVM_SORTIR_SERIAL"] = ""
from app import config
OUT = os.path.join(REPO, "datasets", "4_eksperimen", "uji_app", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(OUT, exist_ok=True)
config.SORTIR_ENABLE = False; config.SORTIR_SERIAL = ""            # talang TIDAK disentuh
# frame diam tersimpan = posisi AKHIR botol (setelah meluncur saat belt berhenti), bisa sedikit di luar zona pemicu 0,40-0,60;
# di mesin, berhenti dipicu saat botol MASIH bergerak masuk zona. Untuk kamera tiruan zona pemicu = zona periksa (hanya di uji ini).
config.STOP_ZONE_X = config.ZONE_X
config.DB_PATH = os.path.join(OUT, "uji.db"); config.SCAN_SAVE_DIR = os.path.join(OUT, "{tanggal}_scan")
import cv2, numpy as np
import replay_keputusan as rk
import rvm_kp.kamera as kam

ROWS = rk.sesi_rows()
if len(sys.argv) > 1:
    ROWS = ROWS[:int(sys.argv[1])]
URUT = []                                                        # (hari, sesi, botol, t_mulai) urutan botol yang disajikan


class KameraTiruan:
    """read(): per botol -> frame1 (memicu jejak & berhenti) lalu frame1, frame2, frame3 (frame diam saat pindai);
    di antara botol -> belt kosong (hitam) selama LOST_SEC + 0,5 dtk agar jejak lama selesai."""
    def __init__(self, *a, **k):
        self.i = -1; self.q = []; self.t_kosong = time.time(); self.blank = np.zeros((720, 1280, 3), np.uint8); self.habis = False; self.mulai = False

    def read(self, timeout=2.0):
        time.sleep(0.03)                                         # ~30 fps kamera
        if not self.mulai:                                       # belt kosong sampai sesi dimulai
            return self.blank
        if self.q:
            return self.q.pop(0)
        if time.time() < self.t_kosong:
            return self.blank
        if self.i >= 0 and not getattr(self, "_jeda_set", False):
            self.t_kosong = time.time() + config.LOST_SEC + 0.5; self._jeda_set = True
            return self.blank
        self.i += 1; self._jeda_set = False
        if self.i >= len(ROWS):
            self.habis = True; return self.blank
        r = ROWS[self.i]; fr = [cv2.imread(f) for f in r["frames"]]
        while len(fr) < 3:
            fr.append(fr[-1])
        URUT.append((r["hari"], r["sesi"], r["botol"], time.time()))
        self.q = [fr[0], fr[1], fr[2]]
        return fr[0]

    def release(self):
        pass

    def close(self):
        pass


kam.Kamera = KameraTiruan
from app.db import DB
from app.plc import Conveyor
from app.engine import Machine as Engine

pesan = []
eng = Engine(DB(path=config.DB_PATH), Conveyor(enabled=False), lambda m: pesan.append(m), sim=False, backend="pytorch", profile="nano")
eng.start()
t0 = time.time()
while eng.status != "siap":
    if eng.status.startswith("galat"):
        print("GALAT:", eng.status, eng.status_msg); sys.exit(1)
    time.sleep(0.5)
print("model dimuat dalam %.1f dtk" % (time.time() - t0), flush=True)
s, err = eng.start_session(); assert s, err
cam = eng.src; time.sleep(1); cam.mulai = True; t1 = time.time()
while not cam.habis:
    time.sleep(1)
    if eng.session is None:                                      # sesi berakhir sendiri (mis. idle) -> mulai lagi
        eng.start_session()
time.sleep(config.LOST_SEC + 2); eng.end_session("uji_selesai"); dur = time.time() - t1
time.sleep(1); eng.stop()

# keputusan aplikasi (DB, urut waktu) dipasangkan dengan botol yang disajikan (urut waktu)
import sqlite3
con = sqlite3.connect(config.DB_PATH)
cols = [c[1] for c in con.execute("PRAGMA table_info(bottles)")]
B = [dict(zip(cols, r)) for r in con.execute("SELECT * FROM bottles ORDER BY ts")]
REP = {(r["hari"], r["sesi"], r["botol"]): r for r in map(json.loads, open(f"{rk.OUT}/hasil.jsonl")) if r["varian"] == "semua_baru"}
out = []; j = 0
for k, (hari, sesi, botol, ts) in enumerate(URUT):
    t_next = URUT[k + 1][3] if k + 1 < len(URUT) else 1e18
    cands = [b for b in B if ts - 0.2 <= b["ts"] < t_next - 0.2]
    a = cands[0] if cands else None
    rp = REP.get((hari, sesi, botol)); row = next(r for r in ROWS if (r["hari"], r["sesi"], r["botol"]) == (hari, sesi, botol))
    out.append(dict(hari=hari, sesi=sesi, botol=botol, harapan=row["harapan"], app=a["decision"] if a else "TIDAK_TERCATAT",
                    alasan_app=(a or {}).get("reason", ""), poin=(a or {}).get("points", ""), replay=rp["keputusan"] if rp else "",
                    n_catat=len(cands)))
with open(f"{OUT}/hasil.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, list(out[0])); w.writeheader(); w.writerows(out)
n = len(out); benar = sum(o["app"] == o["harapan"] for o in out); sama = sum(o["app"] == o["replay"] for o in out)
L = [f"Uji aplikasi utuh (kamera tiruan, model aktif rvm_kp MODEL_SETS), {n} botol, {dur/60:.1f} menit, {n/dur*60:.0f} botol/menit",
     f"benar vs label kelompok: {benar}/{n}", f"sama dengan simulasi keputusan (semua_baru): {sama}/{n}",
     f"tidak tercatat: {sum(o['app']=='TIDAK_TERCATAT' for o in out)} | tercatat >1x: {sum(o['n_catat']>1 for o in out)}",
     "beda dengan simulasi: " + str([(o['hari'], o['sesi'], o['botol'], o['app'], o['alasan_app'], o['replay']) for o in out if o['app'] != o['replay']][:30])]
for hari in ("27", "29"):
    oo = [o for o in out if o["hari"] == hari]
    if oo:
        L.append(f"  {hari} Okt: benar {sum(o['app']==o['harapan'] for o in oo)}/{len(oo)}, salah terima {sum(o['app']=='terima' and o['harapan']=='tolak' for o in oo)}, salah tolak {sum(o['app']!='terima' and o['harapan']=='terima' for o in oo)}")
open(f"{OUT}/ringkas.txt", "w", encoding="utf-8").write("\n".join(L) + "\n"); print("\n".join(L))
