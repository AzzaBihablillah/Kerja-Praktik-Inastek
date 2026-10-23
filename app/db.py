"""Penyimpanan SQLite: sesi, botol, klaim poin, akun, buku poin (ledger), kampanye donasi, hadiah simulasi.
Satu koneksi + kunci thread (dipakai bersama oleh thread analisis dan handler FastAPI)."""
import hashlib
import hmac
import os
import secrets
import sqlite3
import threading
import time

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY, hp TEXT UNIQUE NOT NULL, nama TEXT NOT NULL, pin_hash TEXT NOT NULL, salt TEXT NOT NULL,
  created REAL NOT NULL, fails INTEGER DEFAULT 0, locked_until REAL DEFAULT 0);
CREATE TABLE IF NOT EXISTS auth(token TEXT PRIMARY KEY, user_id INTEGER NOT NULL, created REAL NOT NULL);
CREATE TABLE IF NOT EXISTS sessions(
  id INTEGER PRIMARY KEY, started REAL NOT NULL, ended REAL, end_reason TEXT,
  valid INTEGER DEFAULT 0, invalid INTEGER DEFAULT 0, points INTEGER DEFAULT 0, grams INTEGER DEFAULT 0);
CREATE TABLE IF NOT EXISTS bottles(
  id INTEGER PRIMARY KEY, session_id INTEGER, ts REAL NOT NULL, decision TEXT NOT NULL, reason TEXT,
  size TEXT, points INTEGER, grams INTEGER, p_liquid REAL, cap TEXT, cap_votes TEXT, area REAL, n_frames INTEGER);
CREATE TABLE IF NOT EXISTS claims(
  code TEXT PRIMARY KEY, session_id INTEGER, points INTEGER NOT NULL, grams INTEGER DEFAULT 0, created REAL NOT NULL,
  expires REAL NOT NULL, status TEXT NOT NULL DEFAULT 'open', user_id INTEGER, campaign_id INTEGER, done_ts REAL);
CREATE TABLE IF NOT EXISTS ledger(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, delta INTEGER NOT NULL, kind TEXT NOT NULL, note TEXT, ref TEXT, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS campaigns(
  id INTEGER PRIMARY KEY, slug TEXT UNIQUE, nama TEXT, ringkas TEXT, target INTEGER, seed_points INTEGER DEFAULT 0, sort INTEGER);
CREATE TABLE IF NOT EXISTS donations(
  id INTEGER PRIMARY KEY, campaign_id INTEGER NOT NULL, points INTEGER NOT NULL, user_id INTEGER, source TEXT, ts REAL NOT NULL);
CREATE TABLE IF NOT EXISTS rewards(
  id INTEGER PRIMARY KEY, slug TEXT UNIQUE, nama TEXT, ringkas TEXT, cost INTEGER, stok INTEGER, sort INTEGER);
CREATE TABLE IF NOT EXISTS redemptions(
  id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, reward_id INTEGER NOT NULL, cost INTEGER, kode TEXT, ts REAL NOT NULL);
CREATE INDEX IF NOT EXISTS ix_ledger_user ON ledger(user_id);
CREATE INDEX IF NOT EXISTS ix_bottles_ts ON bottles(ts);
"""

# Data contoh (simulasi). Kampanye & hadiah BUKAN program nyata; tampil dengan label "simulasi".
CAMPAIGNS = [
    ("mangrove", "Mangrove Pesisir Selatan", "Tiap 500 poin = 1 bibit mangrove ditanam di pantai Bantul.", 20000, 6815, 1),
    ("tong-pilah", "Tong Pilah untuk Sekolah", "Tong sampah terpilah untuk 12 SD negeri di Sleman.", 15000, 3270, 2),
    ("sungai-code", "Bersih Sungai Code", "Alat dan konsumsi relawan susur sungai tiap bulan.", 8000, 1945, 3),
]
DEFAULT_CAMPAIGN = "mangrove"   # tujuan poin yang tidak diklaim dalam CLAIM_HOURS
REWARDS = [
    ("pulsa-5k", "Pulsa Rp5.000", "Voucher pulsa semua operator.", 400, 50, 1),
    ("ewallet-10k", "Saldo e-wallet Rp10.000", "Kode isi ulang dompet digital.", 750, 30, 2),
    ("tumbler", "Tumbler bambu 450 ml", "Ambil di meja operator.", 1250, 12, 3),
    ("tote", "Tote bag kanvas", "Tas belanja dari kanvas daur ulang.", 900, 20, 4),
]
CODE_ALPHABET = "23456789ABCDEFGHJKLMNPQRSTUVWXYZ"   # tanpa 0/O/1/I agar mudah diketik


def _hash_pin(pin, salt):
    return hashlib.pbkdf2_hmac("sha256", pin.encode(), salt.encode(), 120_000).hex()


def new_code(n=6):
    return "".join(secrets.choice(CODE_ALPHABET) for _ in range(n))


class AppError(Exception):
    """Galat yang pesannya aman ditampilkan ke pengguna."""

    def __init__(self, msg, status=400):
        super().__init__(msg)
        self.msg, self.status = msg, status


class DB:
    def __init__(self, path=config.DB_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.c = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.c.row_factory = sqlite3.Row
        self.c.execute("PRAGMA journal_mode=WAL")
        self.lock = threading.RLock()
        with self.lock:
            self.c.executescript(SCHEMA)
            for r in CAMPAIGNS:
                self.c.execute("INSERT OR IGNORE INTO campaigns(slug,nama,ringkas,target,seed_points,sort) VALUES(?,?,?,?,?,?)", r)
            for r in REWARDS:
                self.c.execute("INSERT OR IGNORE INTO rewards(slug,nama,ringkas,cost,stok,sort) VALUES(?,?,?,?,?,?)", r)

    # ------------------------------------------------------------------ util
    def q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.c.execute(sql, args).fetchall()]

    def one(self, sql, args=()):
        with self.lock:
            r = self.c.execute(sql, args).fetchone()
            return dict(r) if r else None

    def x(self, sql, args=()):
        with self.lock:
            return self.c.execute(sql, args).lastrowid

    def tx(self):
        """Transaksi: `with db.tx(): ...` (kunci dipegang selama transaksi)."""
        db = self

        class _T:
            def __enter__(self):
                db.lock.acquire(); db.c.execute("BEGIN IMMEDIATE")

            def __exit__(self, et, ev, tb):
                try:
                    db.c.execute("ROLLBACK" if et else "COMMIT")
                finally:
                    db.lock.release()
        return _T()

    # ------------------------------------------------------------------ sesi & botol
    def session_start(self):
        return self.x("INSERT INTO sessions(started) VALUES(?)", (time.time(),))

    def bottle_add(self, sid, b):
        with self.tx():
            self.c.execute(
                "INSERT INTO bottles(session_id,ts,decision,reason,size,points,grams,p_liquid,cap,cap_votes,area,n_frames)"
                " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (sid, b["ts"], b["decision"], b.get("reason"), b["size"], b["points"], b["grams"], b["p_liquid"], b["cap"],
                 b["cap_votes"], b["area"], b["n_frames"]))
            col = "valid" if b["decision"] == "terima" else "invalid"
            self.c.execute(f"UPDATE sessions SET {col}={col}+1, points=points+?, grams=grams+? WHERE id=?",
                           (b["points"], b["grams"], sid))

    def session_end(self, sid, reason):
        """Tutup sesi; bila ada poin, buat klaim (kode QR). Mengembalikan (ringkasan sesi, klaim | None)."""
        now = time.time()
        with self.tx():
            self.c.execute("UPDATE sessions SET ended=?, end_reason=? WHERE id=?", (now, reason, sid))
            s = dict(self.c.execute("SELECT * FROM sessions WHERE id=?", (sid,)).fetchone())
            claim = None
            if s["points"] > 0:
                for _ in range(20):
                    code = new_code()
                    if not self.c.execute("SELECT 1 FROM claims WHERE code=?", (code,)).fetchone():
                        break
                self.c.execute("INSERT INTO claims(code,session_id,points,grams,created,expires) VALUES(?,?,?,?,?,?)",
                               (code, sid, s["points"], s["grams"], now, now + config.CLAIM_HOURS * 3600))
                claim = dict(self.c.execute("SELECT * FROM claims WHERE code=?", (code,)).fetchone())
        return s, claim

    def recover_sessions(self, min_idle, skip_id=0):
        """Sesi yang tidak pernah ditutup (server mati / kamera putus) dan sudah diam >= min_idle detik:
        ditutup dengan alasan 'pemulihan'; bila ada poin, dibuatkan kode klaim (terlihat di panel operator).
        Ambang diam mencegah sesi yang masih berjalan (mis. server lain di database yang sama) ikut ditutup."""
        now = time.time()
        rows = self.q("SELECT s.id, MAX(s.started, COALESCE((SELECT MAX(ts) FROM bottles b WHERE b.session_id=s.id), 0)) AS last"
                      " FROM sessions s WHERE s.ended IS NULL")
        out = []
        for r in rows:
            if r["id"] != skip_id and r["last"] < now - min_idle:
                out.append(self.session_end(r["id"], "pemulihan"))
        return out

    def session_bottles(self, sid):
        return self.q("SELECT * FROM bottles WHERE session_id=? ORDER BY id", (sid,))

    # ------------------------------------------------------------------ klaim
    def claim_get(self, code):
        return self.one("SELECT * FROM claims WHERE code=?", ((code or "").upper().replace("-", "").strip(),))

    def _claim_open(self, code):
        cl = self.c.execute("SELECT * FROM claims WHERE code=?", (code.upper().replace("-", "").strip(),)).fetchone()
        if not cl:
            raise AppError("Kode klaim tidak ditemukan. Periksa lagi hurufnya.", 404)
        cl = dict(cl)
        if cl["status"] != "open":
            raise AppError("Poin dari kode ini sudah " + ("disimpan ke akun." if cl["status"] == "claimed" else "didonasikan."), 409)
        if cl["expires"] < time.time():
            raise AppError("Kode klaim sudah kedaluwarsa.", 410)
        return cl

    def claim_to_user(self, code, uid):
        with self.tx():
            cl = self._claim_open(code)
            now = time.time()
            self.c.execute("UPDATE claims SET status='claimed', user_id=?, done_ts=? WHERE code=?", (uid, now, cl["code"]))
            self.c.execute("INSERT INTO ledger(user_id,delta,kind,note,ref,ts) VALUES(?,?,?,?,?,?)",
                           (uid, cl["points"], "setor", "Setor botol", cl["code"], now))
        return cl

    def claim_donate(self, code, campaign_id, source="kiosk", uid=None):
        with self.tx():
            cl = self._claim_open(code)
            camp = self.c.execute("SELECT * FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            if not camp:
                raise AppError("Kampanye tidak ditemukan.", 404)
            now = time.time()
            self.c.execute("UPDATE claims SET status='donated', campaign_id=?, user_id=?, done_ts=? WHERE code=?",
                           (campaign_id, uid, now, cl["code"]))
            self.c.execute("INSERT INTO donations(campaign_id,points,user_id,source,ts) VALUES(?,?,?,?,?)",
                           (campaign_id, cl["points"], uid, source, now))
        return cl, dict(camp)

    def expire_claims(self):
        """Klaim kedaluwarsa -> didonasikan otomatis ke kampanye default. Mengembalikan jumlah klaim."""
        now = time.time()
        with self.tx():
            camp = self.c.execute("SELECT id FROM campaigns WHERE slug=?", (DEFAULT_CAMPAIGN,)).fetchone()
            rows = self.c.execute("SELECT code, points FROM claims WHERE status='open' AND expires<?", (now,)).fetchall()
            for r in rows:
                self.c.execute("UPDATE claims SET status='donated', campaign_id=?, done_ts=? WHERE code=?", (camp["id"], now, r["code"]))
                self.c.execute("INSERT INTO donations(campaign_id,points,user_id,source,ts) VALUES(?,?,?,?,?)",
                               (camp["id"], r["points"], None, "kedaluwarsa", now))
        return len(rows)

    # ------------------------------------------------------------------ akun
    @staticmethod
    def norm_hp(hp):
        d = "".join(ch for ch in (hp or "") if ch.isdigit())
        if d.startswith("62"):
            d = "0" + d[2:]
        if not (d.startswith("08") and 10 <= len(d) <= 13):
            raise AppError("Nomor HP tidak valid. Contoh: 0812 3456 7890.")
        return d

    @staticmethod
    def check_pin(pin):
        if not (pin and pin.isdigit() and len(pin) == config.PIN_LEN):
            raise AppError(f"PIN harus {config.PIN_LEN} angka.")
        if len(set(pin)) == 1 or pin in ("123456", "654321", "012345"):
            raise AppError("PIN terlalu mudah ditebak. Pakai kombinasi lain.")

    def register(self, hp, nama, pin):
        hp = self.norm_hp(hp)
        nama = " ".join((nama or "").split())[:24]
        if len(nama) < 2:
            raise AppError("Isi nama panggilan (minimal 2 huruf).")
        self.check_pin(pin)
        salt = secrets.token_hex(8)
        with self.tx():
            if self.c.execute("SELECT 1 FROM users WHERE hp=?", (hp,)).fetchone():
                raise AppError("Nomor ini sudah terdaftar. Silakan masuk.", 409)
            uid = self.c.execute("INSERT INTO users(hp,nama,pin_hash,salt,created) VALUES(?,?,?,?,?)",
                                 (hp, nama, _hash_pin(pin, salt), salt, time.time())).lastrowid
        return self._new_token(uid)

    def login(self, hp, pin):
        hp = self.norm_hp(hp)
        u = self.one("SELECT * FROM users WHERE hp=?", (hp,))
        if not u:
            raise AppError("Nomor belum terdaftar. Buat akun dulu.", 404)
        now = time.time()
        if u["locked_until"] > now:
            raise AppError(f"Terlalu banyak percobaan. Coba lagi dalam {int(u['locked_until'] - now) // 60 + 1} menit.", 429)
        if not hmac.compare_digest(_hash_pin(pin or "", u["salt"]), u["pin_hash"]):
            fails = u["fails"] + 1
            lock = now + config.PIN_LOCK_SEC if fails >= config.PIN_MAX_FAIL else 0
            self.x("UPDATE users SET fails=?, locked_until=? WHERE id=?", (0 if lock else fails, lock, u["id"]))
            left = config.PIN_MAX_FAIL - fails
            raise AppError("PIN salah." + (f" Sisa {left} percobaan." if left > 0 else " Akun dikunci 5 menit."), 401)
        self.x("UPDATE users SET fails=0, locked_until=0 WHERE id=?", (u["id"],))
        return self._new_token(u["id"])

    def _new_token(self, uid):
        tok = secrets.token_urlsafe(24)
        self.x("INSERT INTO auth(token,user_id,created) VALUES(?,?,?)", (tok, uid, time.time()))
        return tok

    def user_by_token(self, tok):
        if not tok:
            return None
        return self.one("SELECT u.* FROM auth a JOIN users u ON u.id=a.user_id WHERE a.token=? AND a.created>?",
                        (tok, time.time() - config.AUTH_DAYS * 86400))

    def logout(self, tok):
        self.x("DELETE FROM auth WHERE token=?", (tok,))

    def balance(self, uid):
        r = self.one("SELECT COALESCE(SUM(delta),0) AS b FROM ledger WHERE user_id=?", (uid,))
        return int(r["b"])

    def profile(self, uid):
        u = self.one("SELECT id,hp,nama,created FROM users WHERE id=?", (uid,))
        hist = self.q("SELECT delta,kind,note,ref,ts FROM ledger WHERE user_id=? ORDER BY id DESC LIMIT 30", (uid,))
        st = self.one("SELECT COUNT(*) n, COALESCE(SUM(c.points),0) p, COALESCE(SUM(c.grams),0) g FROM claims c WHERE c.user_id=?",
                      (uid,))
        don = self.one("SELECT COALESCE(SUM(points),0) p FROM donations WHERE user_id=?", (uid,))
        reds = self.q("SELECT r.nama, x.kode, x.cost, x.ts FROM redemptions x JOIN rewards r ON r.id=x.reward_id"
                      " WHERE x.user_id=? ORDER BY x.id DESC LIMIT 10", (uid,))
        hp = u["hp"]
        return dict(nama=u["nama"], hp_mask=hp[:4] + "•" * (len(hp) - 7) + hp[-3:], saldo=self.balance(uid), riwayat=hist,
                    total_setor=st["p"], total_gram=st["g"], total_sesi=st["n"], total_donasi=don["p"], penukaran=reds)

    def donate_from_balance(self, uid, campaign_id, points):
        try:
            points = int(points)
        except (TypeError, ValueError):
            raise AppError("Jumlah poin tidak valid.")
        if points < 10:
            raise AppError("Donasi minimal 10 poin.")
        with self.tx():
            camp = self.c.execute("SELECT * FROM campaigns WHERE id=?", (campaign_id,)).fetchone()
            if not camp:
                raise AppError("Kampanye tidak ditemukan.", 404)
            bal = self.c.execute("SELECT COALESCE(SUM(delta),0) FROM ledger WHERE user_id=?", (uid,)).fetchone()[0]
            if points > bal:
                raise AppError(f"Saldo tidak cukup (saldo {bal} poin).")
            now = time.time()
            self.c.execute("INSERT INTO ledger(user_id,delta,kind,note,ref,ts) VALUES(?,?,?,?,?,?)",
                           (uid, -points, "donasi", camp["nama"], camp["slug"], now))
            self.c.execute("INSERT INTO donations(campaign_id,points,user_id,source,ts) VALUES(?,?,?,?,?)",
                           (campaign_id, points, uid, "akun", now))
        return dict(camp)

    def redeem(self, uid, reward_id):
        with self.tx():
            r = self.c.execute("SELECT * FROM rewards WHERE id=?", (reward_id,)).fetchone()
            if not r:
                raise AppError("Hadiah tidak ditemukan.", 404)
            if r["stok"] <= 0:
                raise AppError("Stok hadiah ini habis.")
            bal = self.c.execute("SELECT COALESCE(SUM(delta),0) FROM ledger WHERE user_id=?", (uid,)).fetchone()[0]
            if r["cost"] > bal:
                raise AppError(f"Poin belum cukup. Kurang {r['cost'] - bal} poin.")
            kode = "DEMO-" + new_code(4) + "-" + new_code(4)
            now = time.time()
            self.c.execute("UPDATE rewards SET stok=stok-1 WHERE id=?", (reward_id,))
            self.c.execute("INSERT INTO ledger(user_id,delta,kind,note,ref,ts) VALUES(?,?,?,?,?,?)",
                           (uid, -r["cost"], "tukar", r["nama"], kode, now))
            self.c.execute("INSERT INTO redemptions(user_id,reward_id,cost,kode,ts) VALUES(?,?,?,?,?)",
                           (uid, reward_id, r["cost"], kode, now))
        return dict(nama=r["nama"], kode=kode, cost=r["cost"])

    # ------------------------------------------------------------------ katalog & statistik
    def campaigns(self):
        return self.q("SELECT c.id, c.slug, c.nama, c.ringkas, c.target, c.seed_points + COALESCE(SUM(d.points),0) AS terkumpul,"
                      " COUNT(d.id) AS n_donasi FROM campaigns c LEFT JOIN donations d ON d.campaign_id=c.id"
                      " GROUP BY c.id ORDER BY c.sort")

    def rewards(self):
        return self.q("SELECT id, slug, nama, ringkas, cost, stok FROM rewards ORDER BY sort")

    def stats(self):
        midnight = time.mktime(time.localtime()[:3] + (0, 0, 0, 0, 0, -1))
        tot = self.one("SELECT COUNT(*) n, COALESCE(SUM(grams),0) g, COALESCE(SUM(points),0) p FROM bottles WHERE decision='terima'")
        today = self.one("SELECT COUNT(*) n, COALESCE(SUM(grams),0) g FROM bottles WHERE decision='terima' AND ts>=?", (midnight,))
        recent = self.q("SELECT decision, reason, size, points, ts FROM bottles ORDER BY id DESC LIMIT 8")
        users = self.one("SELECT COUNT(*) n FROM users")
        return dict(botol_total=tot["n"], gram_total=tot["g"], poin_total=tot["p"], botol_hari_ini=today["n"],
                    gram_hari_ini=today["g"], terbaru=recent, pengguna=users["n"])
