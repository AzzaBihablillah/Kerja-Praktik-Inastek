"""Deteksi "ada benda" berbasis patokan BELT KOSONG — mode CATAT SAJA (26 Okt). Tidak menghentikan belt, tidak memengaruhi
keputusan/poin. Tujuan: mengumpulkan bukti di belt nyata untuk menutup celah benda yang TIDAK terlihat detektor
(uji 22 Okt: 15/86 non-botol tanpa kotak apa pun).

Alur per sesi (bukan simulasi):
  1. Patokan: frame tanpa kotak detektor apa pun (botol / non_bottle) dikumpulkan sampai REF_FRAMES -> median per piksel.
     Gambar patokan disimpan: <scan>/benda/belt_kosong_<jam>.jpg.  (Mulai sesi dengan belt KOSONG beberapa detik.)
  2. Tiap 2 frame: gumpalan beda-dari-patokan di zona periksa. Satu benda yang lewat = satu KEJADIAN (dari gumpalan pertama
     sampai zona kosong 3 kali berturut-turut). Tiap frame kejadian diberi label dari detektor:
        botol      = gumpalan tertutup kotak botol
        non_botol  = gumpalan tertutup kotak non_bottle
        tak_dikenal= gumpalan TIDAK tertutup kotak apa pun   <- celah yang ingin ditutup
     Frame dengan kotak detektor di zona tetapi TANPA gumpalan dihitung 'luput_patokan' (patokan gagal melihat benda).
  3. Akhir kejadian -> baris <scan>/BENDA.csv + gambar <scan>/benda/E<jam>.jpg (frame terbesar + topeng merah).
  4. Patokan diperbarui pelan hanya saat seluruh frame kosong (tahan perubahan cahaya). Beda tanpa kotak detektor yang DIAM
     > STATIC_SEC (benda lewat selalu bergerak) = latar berubah (kamera tergeser, cahaya, noda) -> dicatat 'latar_berubah',
     patokan dibangun ulang dari frame kosong berikutnya."""
import datetime
import os
from collections import Counter

import cv2
import numpy as np

from . import config

STATIC_SEC = 8.0     # benda lewat selalu bergerak; beda tanpa kotak detektor yang bertahan > ini = latar berubah, bukan benda
COLS = ["waktu", "sesi", "durasi_s", "kategori", "f_botol", "f_non_botol", "f_tak_dikenal", "f_luput_patokan", "luas_maks", "file"]


class PencatatBenda(object):
    def __init__(self, sim=False):
        from rvm_kp.benda import BeltRef
        self.sim = sim; self.ref = BeltRef(); self.ev = None; self.empty = 0; self.i = 0; self.geser = 0
        self.n = Counter(); self.ref_file = ""

    @property
    def aktif(self):
        return config.PRESENCE_MODE == "catat" and not self.sim and bool(config.SCAN_SAVE_DIR)

    def reset(self):
        self._finish()
        self.ref.reset(); self.ev = None; self.empty = 0; self.ref_file = ""

    def status(self):
        return dict(mode=config.PRESENCE_MODE if not self.sim else "mati (simulasi)", patokan_siap=self.ref.ready,
                    patokan_kumpul="%d/%d" % (len(self.ref.buf), self.ref.n), kejadian=dict(self.n), file_patokan=self.ref_file)

    def _dir(self):
        d = os.path.join(config.scan_dir(), "benda")
        os.makedirs(d, exist_ok=True)
        return d

    def step(self, frame, bottles, nonbottles, now, sid):
        if not self.aktif:
            return
        try:
            self._step(frame, [b.box for b in bottles], [b.box for b in nonbottles], now, sid)
        except Exception as e:                        # pencatat tidak boleh mengganggu mesin
            print("[benda] galat:", e, flush=True)

    def _step(self, frame, bb, nb, now, sid):
        from rvm_kp.benda import outside
        W = frame.shape[1]; z0, z1 = config.ZONE_X[0] * W, config.ZONE_X[1] * W
        in_zone = lambda b: z0 <= (b[0] + b[2]) / 2 <= z1
        if not self.ref.ready:
            if not bb and not nb:
                self.ref.feed_empty(frame)
                if self.ref.ready:
                    self.ref_file = os.path.join(self._dir(), "belt_kosong_%s.jpg" % datetime.datetime.now().strftime("%H%M%S"))
                    cv2.imwrite(self.ref_file, self.ref.ref_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    print("[benda] patokan belt kosong siap -> %s" % self.ref_file, flush=True)
            return
        self.i += 1
        if self.i % 2:
            return
        sm, m = self.ref.mask(frame)
        det = [b for b in bb + nb if in_zone(b)]
        if m is not None and not bb and not nb and m.mean() / 255.0 > 0.25:   # >25% frame beda tanpa kotak apa pun:
            self.geser += 1                                                   # kamera tergeser / cahaya berubah besar
            if self.geser >= 20:
                print("[benda] PERINGATAN: patokan tidak cocok lagi (kamera tergeser / cahaya berubah) -> dibangun ulang", flush=True)
                self._finish(); self.ref.reset(); self.geser = 0; self.n["patokan_ulang"] += 1
            return
        self.geser = 0
        blobs = self.ref.objects(frame, zone=config.ZONE_X, mask=m)
        if not blobs and not det:
            self.empty += 1
            if self.empty >= 3:
                self._finish()
                if m is not None and m.mean() / 255.0 < 0.005:          # seluruh frame (praktis) kosong: perbarui patokan
                    self.ref.feed_empty(frame)
            return
        self.empty = 0
        if self.ev is None:
            self.ev = dict(t0=now, t1=now, sid=sid, lab=Counter(), area=-1.0, img=None, det=False)
        ev = self.ev; ev["t1"] = now; ev["det"] = ev["det"] or bool(det)
        if not ev["det"] and now - ev["t0"] > STATIC_SEC:            # beda diam lama tanpa kotak detektor = latar berubah
            print("[benda] beda diam %.0f s tanpa kotak detektor -> latar berubah (kamera tergeser / cahaya / noda), patokan dibangun ulang"
                  % (now - ev["t0"]), flush=True)
            ev["lab"] = Counter(latar_berubah=1); self._finish(); self.ref.reset()
            return
        if not blobs:
            ev["lab"]["luput_patokan"] += 1
        for g in blobs:
            if bb and outside(g, bb) <= 0.5:
                ev["lab"]["botol"] += 1
            elif nb and outside(g, nb) <= 0.5:
                ev["lab"]["non_botol"] += 1
            else:
                ev["lab"]["tak_dikenal"] += 1
            if g[4] > ev["area"]:
                ev["area"] = g[4]; v = sm.copy()
                if m is not None:
                    v[m > 0] = (0.5 * v[m > 0] + (0, 0, 127)).astype(np.uint8)
                k = sm.shape[1] / float(W)
                for b in bb:
                    cv2.rectangle(v, (int(b[0] * k), int(b[1] * k)), (int(b[2] * k), int(b[3] * k)), (60, 220, 60), 2)
                for b in nb:
                    cv2.rectangle(v, (int(b[0] * k), int(b[1] * k)), (int(b[2] * k), int(b[3] * k)), (40, 140, 255), 2)
                ev["img"] = np.hstack([sm, v])

    def _finish(self):
        ev, self.ev = self.ev, None
        if ev is None or (ev["area"] < 0 and not ev["lab"]):
            return
        lab = ev["lab"]
        kat = ("latar_berubah" if lab["latar_berubah"] else "botol" if lab["botol"] else "non_botol" if lab["non_botol"]
               else "tak_dikenal" if lab["tak_dikenal"] else "luput_patokan")
        self.n[kat] += 1
        stamp = datetime.datetime.fromtimestamp(ev["t0"]).strftime("%H%M%S")
        f = ""
        try:
            d = self._dir()
            if ev["img"] is not None:
                f = os.path.join(d, "E%s_%s.jpg" % (stamp, kat)); cv2.imwrite(f, ev["img"], [cv2.IMWRITE_JPEG_QUALITY, 80])
            from .engine import csv_add
            csv_add(os.path.join(os.path.dirname(d), "BENDA.csv"), dict(
                waktu=stamp, sesi=ev["sid"], durasi_s=round(ev["t1"] - ev["t0"], 2), kategori=kat, f_botol=lab["botol"],
                f_non_botol=lab["non_botol"], f_tak_dikenal=lab["tak_dikenal"], f_luput_patokan=lab["luput_patokan"],
                luas_maks=round(max(ev["area"], 0.0), 4), file=os.path.basename(f)), COLS)
        except Exception as e:
            print("[benda] simpan gagal:", e, flush=True)
        if kat == "tak_dikenal":
            print("[benda] KEJADIAN tak dikenal detektor (sesi %s, %s) -> %s" % (ev["sid"], stamp, os.path.basename(f)), flush=True)
