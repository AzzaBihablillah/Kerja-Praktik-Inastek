"""UJI KETAHANAN aplikasi utuh (server web + kiosk + model) dengan KAMERA TIRUAN yang memutar frame diam sesi belt
BERULANG-ULANG tanpa henti. PLC & talang ESP32 dimatikan; DB ke folder uji; frame tidak disimpan (hemat kartu SD).
Sesi dimulai ulang otomatis. Tiap 60 dtk menulis status.json: botol disajikan vs tercatat, keputusan sama dengan label,
RAM proses, FPS, umur proses.  Di Jetson (dalam container):
  python3 scripts/alat/app_ketahanan.py --backend tensorrt      -> /rvm/expo/ketahanan/<waktu>/"""
import os, sys, time, json, threading, datetime, sqlite3, urllib.request
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts", "alat"))
os.environ["RVM_SORTIR_SERIAL"] = ""
from app import config
OUT = os.path.join(REPO, "expo", "ketahanan", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
os.makedirs(OUT, exist_ok=True)
config.SORTIR_ENABLE = False; config.SORTIR_SERIAL = ""
config.STOP_ZONE_X = config.ZONE_X                  # frame diam = posisi akhir botol (sama dengan uji_app_putar)
config.DB_PATH = os.path.join(OUT, "uji.db"); config.SCAN_SAVE_DIR = ""
import cv2, numpy as np
import replay_keputusan as rk
import rvm_kp.kamera as kam

ROWS = rk.sesi_rows()
SAJI = []                                           # (t, harapan, kunci) tiap botol yang disajikan
T0 = time.time()


class KameraUlang:
    def __init__(self, *a, **k):
        self.i = -1; self.q = []; self.t_kosong = time.time() + 5; self.blank = np.zeros((720, 1280, 3), np.uint8); self.jeda = False
        self.cache = {}

    def _img(self, f):
        im = self.cache.get(f)
        if im is None:
            im = self.cache[f] = cv2.imread(f)
        return im

    def read(self, timeout=2.0):
        time.sleep(0.03)
        if self.q:
            return self.q.pop(0)
        if time.time() < self.t_kosong:
            return self.blank
        if self.i >= 0 and not self.jeda:
            self.t_kosong = time.time() + config.LOST_SEC + 0.5; self.jeda = True
            return self.blank
        self.jeda = False; self.i += 1
        r = ROWS[self.i % len(ROWS)]; fr = [self._img(f) for f in r["frames"]]
        while len(fr) < 3:
            fr.append(fr[-1])
        SAJI.append((time.time(), r["harapan"], "%s/%s/%s" % (r["hari"], r["sesi"], r["botol"])))
        self.q = [fr[0], fr[1], fr[2]]
        return fr[0]

    def release(self):
        pass

    def close(self):
        pass


kam.Kamera = KameraUlang                             # app.engine mengimpor Kamera saat membuka kamera -> dapat versi tiruan


def rss_mb():
    try:
        for l in open("/proc/self/status"):
            if l.startswith("VmRSS"):
                return int(l.split()[1]) // 1024
    except Exception:
        return -1


def penjaga(port):
    """mulai sesi bila tidak ada; tulis status.json tiap 60 dtk."""
    base = "http://127.0.0.1:%d" % port
    while True:
        time.sleep(10)
        try:
            h = json.loads(urllib.request.urlopen(base + "/health", timeout=5).read().decode())
            m = h.get("mesin", {})
            if m.get("status") == "siap" and not m.get("sesi_aktif"):
                urllib.request.urlopen(urllib.request.Request(base + "/api/sesi/mulai", data=b"{}", headers={"Content-Type": "application/json"}), timeout=5)
            if int(time.time() - T0) % 60 < 10:
                n_saji = len(SAJI); B = []
                if os.path.exists(config.DB_PATH):
                    con = sqlite3.connect(config.DB_PATH)
                    B = [(r[0], r[1]) for r in con.execute("SELECT ts, decision FROM bottles ORDER BY ts")]; con.close()
                # pasangkan: botol tercatat setelah disajikan & sebelum botol berikutnya
                benar = salah = hilang = 0; j = 0; salah_list = []
                for k, (t, harap, kunci) in enumerate(SAJI[:-1]):
                    t_next = SAJI[k + 1][0]
                    while j < len(B) and B[j][0] < t - 0.2:
                        j += 1
                    if j < len(B) and B[j][0] < t_next - 0.2:
                        if B[j][1] == harap:
                            benar += 1
                        else:
                            salah += 1; salah_list.append([kunci, B[j][1], harap])
                    else:
                        hilang += 1
                st = dict(waktu=datetime.datetime.now().isoformat(timespec="seconds"), menit=round((time.time() - T0) / 60, 1),
                          status=m.get("status"), fps=m.get("fps"), sesi_aktif=m.get("sesi_aktif"), disajikan=n_saji, tercatat_db=len(B),
                          benar=benar, salah=salah, hilang=hilang, salah_contoh=salah_list[-5:], rss_mb=rss_mb(), ws=h.get("ws"))
                json.dump(st, open(os.path.join(OUT, "status.json"), "w"))
                open(os.path.join(OUT, "riwayat.jsonl"), "a").write(json.dumps(st) + "\n")
        except Exception as e:
            open(os.path.join(OUT, "riwayat.jsonl"), "a").write(json.dumps(dict(waktu=time.time(), galat=str(e))) + "\n")


if __name__ == "__main__":
    port = 8000
    threading.Thread(target=penjaga, args=(port,), daemon=True).start()
    sys.argv = ["app", "--no-simpan"] + sys.argv[1:]
    from app.__main__ import main
    main()
