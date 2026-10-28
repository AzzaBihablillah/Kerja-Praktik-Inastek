"""Penjadwal PENYAPU V di ujung belt (servo buatan tim): melempar benda ke wadah TERIMA atau BUANG.

Alat: V terbalik menggantung di atas ujung belt, benda masuk ke dalam V, servo mengayunkan V ke satu sisi sehingga benda
terlempar ke wadah. Tanpa sensor tambahan -> kamera satu-satunya sensor, jadi penjadwal dibuat GAGAL-AMAN:
  * JAM BELT: yang dihitung adalah detik belt BERJALAN (Conveyor.running). Saat belt berhenti (pindai / menunggu benda asing
    diambil) jam ikut berhenti, karena benda di belt juga diam.
  * TIKET TERIMA: botol yang DITERIMA tiba di V setelah SORTIR_TRAVEL_SEC detik belt jalan sejak keputusannya -> V menyapu ke
    TERIMA tepat saat itu.
  * SAPUAN BUANG BERKALA: setiap SORTIR_BUANG_EVERY detik belt jalan, V menyapu ke BUANG (kecuali dekat jadwal tiket terima).
    Jadi benda apa pun di bawah V -- termasuk yang TIDAK terdeteksi kamera -- terbuang paling lambat pada sapuan berikutnya.
    Kegagalan deteksi = dibuang, bukan diterima.
  * TIKET BUANG: benda yang ditolak (bertutup, berisi, bukan botol, ...) juga dijadwalkan disapu ke BUANG saat tiba.
Batasan: dua benda yang tiba berdekatan (< SORTIR_GUARD_SEC) bisa ikut tersapu ke sisi yang sama -> jarak antar benda di belt
harus cukup (berhenti-lalu-pindai sudah memisahkan benda satu per satu). Semua angka = KALIBRASI di mesin (app/config.py).

Perintah ke servo lewat `_kirim(posisi)`: "TERIMA" | "BUANG" | "TENGAH". Kini dicatat (konsol + SORTIR.csv di folder scan).
Bila SORTIR_SERIAL diisi (mis. "COM5" / "/dev/ttyUSB0"), juga dikirim lewat serial dengan protokol USULAN satu huruf + newline:
  T = sapu ke TERIMA, B = sapu ke BUANG, N = kembali ke TENGAH  (sesuaikan dengan firmware tim)."""
import os
import threading
import time

from . import config

# Firmware tim `sortir_servo.ino` (ESP32, 28 Okt): "VALID" -> 15 deg (wadah TERIMA), "INVALID" -> 165 deg (BUANG), tahan 0,4 s lalu
# KEMBALI SENDIRI ke 90 deg; balasan "BUSY x" / "DONE x" (terukur 1,2 s per siklus). Perintah saat sibuk diantre firmware.
# TENGAH tidak dikirim (firmware kembali sendiri).
POS_KODE = {"TERIMA": b"VALID\n", "BUANG": b"INVALID\n"}


class Sortir(object):
    def __init__(self, conveyor=None, sim=False):
        self.conveyor, self.sim = conveyor, sim
        self.lock = threading.Lock()
        self.odo = 0.0                     # detik belt berjalan (jam belt)
        self.tiket = []                    # [dict(odo_tiba, jalur, id, alasan)]
        self.posisi = "TENGAH"
        self.balik_wall = None
        self.next_buang = config.SORTIR_BUANG_EVERY
        self.last_terima = None            # jam belt sapuan TERIMA terakhir (jendela aman sesudahnya)
        self.buang_tiba = []               # jam tiba benda yang ditolak (aturan BUANG menang, 28 Okt)
        self.n = dict(terima=0, buang_tiket=0, buang_berkala=0, dilewati=0)
        self.terakhir = None
        self.ser = None
        self._stop = threading.Event(); self._t = None; self._last = time.time()

    # ------------------------------------------------------------------ siklus hidup
    def start(self):
        if not config.SORTIR_ENABLE or self._t is not None:
            return
        if config.SORTIR_SERIAL:
            try:
                import serial
                self.ser = serial.Serial(config.SORTIR_SERIAL, config.SORTIR_BAUD, timeout=0.2)
                t0 = time.time(); buf = b""                 # membuka port me-reset ESP32: tunggu 'READY' (boot ~2,3 s)
                while time.time() - t0 < 4 and b"READY" not in buf:
                    buf += self.ser.read(256)
                print("[sortir] serial", config.SORTIR_SERIAL, "terbuka", "(READY)" if b"READY" in buf else "(tanpa READY!)", flush=True)
            except Exception as e:                   # tanpa serial: tetap jalan, perintah hanya dicatat
                self.ser = None; print("[sortir] serial gagal:", e, flush=True)
        self._last = time.time()
        self._t = threading.Thread(target=self._run, name="rvm-sortir", daemon=True); self._t.start()
        print("[sortir] penyapu V aktif: tiba %.1f s belt, sapuan buang tiap %.1f s belt%s" % (
            config.SORTIR_TRAVEL_SEC, config.SORTIR_BUANG_EVERY, "" if config.SORTIR_BUANG_PERIODIK else " (MATI)"), flush=True)

    def stop(self):
        self._stop.set()
        if self._t is not None:
            self._t.join(timeout=2)
        try:
            if self.ser:
                self.ser.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ dari mesin
    def keputusan(self, b):
        """Dipanggil SEKALI per keputusan akhir (Machine._record). Menjadwalkan sapuan saat benda tiba di V."""
        jalur = "TERIMA" if b.get("decision") == "terima" else "BUANG"
        with self.lock:
            tiba = self.odo + config.SORTIR_TRAVEL_SEC
            if jalur == "BUANG":
                self.buang_tiba = [x for x in self.buang_tiba if x > self.odo - 10] + [tiba]
            if jalur == "TERIMA" and config.SORTIR_KOSONGKAN:
                # 28 Okt (talang tilting: benda DIAM di talang sampai dimiringkan): kosongkan talang ke BUANG sesaat sebelum botol
                # sah mendarat, agar benda tak terdeteksi yang tertinggal di talang tidak ikut ke wadah TERIMA. Dilewati bila sudah
                # ada sapuan terjadwal tepat sebelumnya (talang baru saja dikosongkan / botol sebelumnya masih diproses).
                # 28 Okt (koreksi di mesin): DULU dijadwalkan tiba-1,5 s -> botol sudah mendarat lebih awal dan IKUT terbuang.
                # KINI segera saat keputusan (botol masih di bawah kamera, jauh dari talang). Dilewati bila masih ada botol
                # sebelumnya yang menunggu dimiringkan (tiket mana pun belum jalan) -> botol itu tidak ikut dibuang.
                pre = self.odo
                if not self.tiket:
                    self.tiket.append(dict(odo_tiba=pre, jalur="BUANG", id=b.get("id"), alasan="kosongkan_talang"))
            if jalur == "TERIMA" or config.SORTIR_BUANG_TIKET:
                self.tiket.append(dict(odo_tiba=tiba, jalur=jalur, id=b.get("id"), alasan=b.get("reason") or ""))
            self.terakhir = (b.get("id"), jalur, b.get("reason"))
        print("[sortir] #%s -> %s, tiba di V pada jam belt %.1f s%s" % (
            b.get("id"), jalur, tiba, " (" + b["reason"] + ")" if b.get("reason") else ""), flush=True)

    def status(self):
        with self.lock:
            return dict(aktif=config.SORTIR_ENABLE, posisi=self.posisi, jam_belt=round(self.odo, 1),
                        antrean=[dict(id=t["id"], jalur=t["jalur"], tiba_dalam=round(t["odo_tiba"] - self.odo, 1)) for t in self.tiket],
                        sapuan=dict(self.n), tiba_s=config.SORTIR_TRAVEL_SEC, buang_tiap_s=config.SORTIR_BUANG_EVERY,
                        serial=config.SORTIR_SERIAL or "-", berkala=config.SORTIR_BUANG_PERIODIK)

    # ------------------------------------------------------------------ penjadwal
    def _belt_jalan(self):
        return bool(self.conveyor is not None and self.conveyor.running)

    def _run(self):
        while not self._stop.is_set():
            try:
                self._tick(time.time())
            except Exception as e:                   # gangguan penjadwal tidak boleh menghentikan mesin
                print("[sortir] galat:", e, flush=True)
            self._stop.wait(0.02)

    def _tick(self, now):
        dt = now - self._last; self._last = now
        aksi = []
        with self.lock:
            if self.balik_wall is not None and now >= self.balik_wall:
                self.balik_wall = None; aksi.append(("TENGAH", None, "kembali"))
            if self._belt_jalan():
                self.odo += dt
            sibuk = self.balik_wall is not None
            due = [t for t in self.tiket if t["odo_tiba"] <= self.odo]
            if due and not sibuk:
                t = min(due, key=lambda z: z["odo_tiba"]); self.tiket.remove(t)
                if t["jalur"] == "TERIMA" and any(abs(x - t["odo_tiba"]) <= config.SORTIR_GUARD_SEC for x in self.buang_tiba):
                    t["jalur"] = "BUANG"; t["alasan"] = "konflik_dekat_benda_tolak"   # wadah terima tetap bersih; poin pelanggan tetap
                    print("[sortir] #%s TERIMA -> BUANG: tiba berdekatan dengan benda yang ditolak" % t.get("id"), flush=True)
                aksi.append((t["jalur"], t, "tiket"))
                if t["jalur"] == "TERIMA":
                    self.last_terima = self.odo
                self.n["terima" if t["jalur"] == "TERIMA" else "buang_tiket"] += 1
                sibuk = True
            if config.SORTIR_BUANG_PERIODIK and self.odo >= self.next_buang:
                g = config.SORTIR_GUARD_SEC
                # jendela aman botol diterima: +-GUARD dari jadwal yang menunggu, dan GUARD sesudah sapuan TERIMA terakhir
                akhir = [t["odo_tiba"] + g for t in self.tiket if t["jalur"] == "TERIMA" and abs(t["odo_tiba"] - self.odo) <= g]
                if self.last_terima is not None and self.odo - self.last_terima < g:
                    akhir.append(self.last_terima + g)
                if akhir:
                    self.next_buang = max(akhir) + 0.01; self.n["dilewati"] += 1      # tunda sampai jendela aman lewat
                elif sibuk:
                    self.next_buang = self.odo + 0.1                                   # servo masih menyapu: coba sebentar lagi
                else:
                    self.next_buang = self.odo + config.SORTIR_BUANG_EVERY
                    aksi.append(("BUANG", None, "berkala")); self.n["buang_berkala"] += 1; sibuk = True
            for pos, t, why in aksi:
                if pos != "TENGAH":
                    self.balik_wall = now + config.SORTIR_SWEEP_SEC
                self.posisi = pos
        for pos, t, why in aksi:
            self._kirim(pos, t, why)

    def _kirim(self, posisi, t=None, why=""):
        """Perintah ke servo penyapu V. Ganti / lengkapi di sini sesuai alat tim."""
        if why != "kembali" and (why != "berkala" or config.SORTIR_LOG_BERKALA):
            print("[sortir] V -> %s (%s%s) jam belt %.1f s" % (posisi, why, " #%s" % t["id"] if t else "", self.odo), flush=True)
        if self.ser is not None:
            try:
                if posisi in POS_KODE:
                    self.ser.write(POS_KODE[posisi])
            except Exception as e:
                print("[sortir] serial tulis gagal:", e, flush=True)
        if config.SCAN_SAVE_DIR and not self.sim and why != "kembali" and (why != "berkala" or config.SORTIR_LOG_BERKALA):
            try:
                import datetime
                d = config.scan_dir()
                p = os.path.join(d, "SORTIR.csv"); new = not os.path.exists(p)
                with open(p, "a") as fh:
                    if new:
                        fh.write("waktu,jam_belt_s,posisi,sebab,botol,alasan\n")
                    fh.write("%s,%.2f,%s,%s,%s,%s\n" % (datetime.datetime.now().strftime("%H%M%S.%f")[:9], self.odo, posisi, why,
                                                         t["id"] if t else "", t["alasan"] if t else ""))
            except Exception as e:
                print("[sortir] SORTIR.csv gagal:", e, flush=True)
