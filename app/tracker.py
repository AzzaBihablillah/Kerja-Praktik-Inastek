"""Pelacak botol antar-frame + keputusan akhir per botol (voting).

Satu botol di belt terlihat di beberapa frame. Tiap frame memberi satu "suara" (tutup, p_liquid, ukuran) selama pusat botol
berada di zona periksa. Botol diputuskan SEKALI: saat keluar zona setelah cukup suara, atau saat hilang dari frame."""
import time
from collections import Counter
from statistics import median

from . import config

REASONS = {
    "tutup": ("Tutup masih terpasang", "Lepas tutupnya, lalu masukkan lagi. Cincin di leher botol boleh tertinggal."),
    "cairan": ("Masih ada isinya", "Habiskan atau buang isinya dulu, lalu masukkan lagi."),
    "tutup_cairan": ("Masih bertutup dan masih ada isinya", "Habiskan atau buang isinya, lepas tutupnya, lalu masukkan lagi."),
    "leher": ("Leher botol tidak terlihat", "Baringkan botol di belt dengan leher menghadap samping, lalu coba lagi."),
}
SIZE_TEXT = {"kecil": "Kecil", "tanggung": "Tanggung", "besar": "Besar", "?": "Ukuran tak terbaca"}


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def center(b):
    return (b[0] + b[2]) / 2, (b[1] + b[3]) / 2


class Track:
    _next = 1

    def __init__(self, r, now):
        self.id = Track._next; Track._next += 1
        self.box = r.box; self.first = self.last = now
        self.votes = []          # (cap, cap_conf, p_liquid, size, area)
        self.decided = False
        self.was_in_zone = False
        self.cur = r             # hasil frame terakhir (untuk gambar pratinjau)

    def vote(self, r):
        self.votes.append((r.cap, r.cap_conf, r.p_liquid, r.size, r.area_1920))


class Tracker:
    def __init__(self, zone=config.ZONE_X, min_votes=config.MIN_VOTES, lost_sec=config.LOST_SEC):
        self.zone, self.min_votes, self.lost_sec = zone, min_votes, lost_sec
        self.tracks = []

    def reset(self):
        self.tracks = []

    def update(self, results, width, now=None):
        """results: list BottleResult (rvm_kp). Mengembalikan list Track yang BARU diputuskan pada frame ini."""
        now = time.time() if now is None else now
        z0, z1 = self.zone[0] * width, self.zone[1] * width
        free = list(self.tracks); done = []
        for r in sorted(results, key=lambda r: -r.conf):
            best, score = None, 0.0
            cx, cy = center(r.box); size = max(r.box[2] - r.box[0], r.box[3] - r.box[1])
            for t in free:
                s = iou(r.box, t.box)
                tx, ty = center(t.box)
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if s < 0.1:
                    if d >= 0.6 * size:
                        continue
                    s = 0.1 * (1 - d / (0.6 * size)) + 0.001   # frame jarang (Nano): IoU kecil tapi pusat masih dekat
                if s > score:
                    best, score = t, s
            if best is None:
                best = Track(r, now); self.tracks.append(best)
            else:
                free.remove(best)
            best.box, best.last, best.cur = r.box, now, r
            in_zone = z0 <= cx <= z1
            if in_zone and not best.decided:
                best.vote(r); best.was_in_zone = True
            elif best.was_in_zone and not best.decided and len(best.votes) >= self.min_votes:
                best.decided = True; done.append(best)      # keluar zona setelah cukup suara
        keep = []
        for t in self.tracks:
            if now - t.last > self.lost_sec:
                if not t.decided and len(t.votes) >= self.min_votes:
                    t.decided = True; done.append(t)
            else:
                keep.append(t)
        self.tracks = keep
        return done

    def flush(self):
        """Akhir sesi: putuskan botol yang masih terlihat dan sudah cukup suara."""
        done = [t for t in self.tracks if not t.decided and len(t.votes) >= self.min_votes]
        for t in done:
            t.decided = True
        self.tracks = []
        return done


def decide(track, liquid_thr):
    """Keputusan akhir satu botol dari suara semua frame di zona."""
    caps = Counter(v[0] for v in track.votes)
    n_seen = caps["tutup"] + caps["tanpa_tutup"]
    p_liq = median(v[2] for v in track.votes)
    sizes = Counter(v[3] for v in track.votes if v[3] != "?")
    size = sizes.most_common(1)[0][0] if sizes else "?"
    area = median(v[4] for v in track.votes)

    if n_seen == 0:
        reason = "leher"                                   # tanpa deteksi leher yang sah: jangan dianggap tanpa tutup
    elif caps["tanpa_tutup"] < config.NOCAP_FRAC * n_seen:
        reason = "tutup_cairan" if p_liq >= liquid_thr else "tutup"     # 22 Okt: cairan tetap ditampilkan walau bertutup
    elif p_liq >= liquid_thr:
        reason = "cairan"
    else:
        reason = None
    ok = reason is None
    pts = (config.POINTS_BY_SIZE.get(size, config.POINTS_UNKNOWN_SIZE) + config.BRAND_BONUS) if ok else 0
    out = dict(
        id=track.id, ts=time.time(), decision="terima" if ok else "tolak", reason=reason,
        size=size, size_text=SIZE_TEXT[size], points=pts, grams=config.GRAMS_BY_SIZE.get(size, 16) if ok else 0,
        p_liquid=round(float(p_liq), 3), cap=("tanpa_tutup" if n_seen and reason not in ("tutup", "tutup_cairan") else ("tutup" if n_seen else "?")),
        cap_votes="%d/%d/%d" % (caps["tanpa_tutup"], caps["tutup"], caps["?"]), area=round(float(area)), n_frames=len(track.votes),
    )
    if reason:
        out["reason_text"], out["tip"] = REASONS[reason]
    return out
