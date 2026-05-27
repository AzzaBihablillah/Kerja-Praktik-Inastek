"""
Session Manager untuk RVM.

Mengelola state sesi aktif: daftar botol yang sudah diproses, hitungan valid/invalid,
dan total poin reward.  Dirancang thread-safe agar bisa diakses bersamaan oleh
inference thread dan FastAPI server.

Alasan invalid yang didukung saat ini:
    - "bukan_botol_pet"  : model bottle detection mengembalikan class not_bottle
    - "berisi_cairan"    : (future) load cell mendeteksi botol tidak kosong
    - "tutup_terpasang"  : (future) model cap detection
    - "botol_rusak"      : (future) model condition mendeteksi Deformed

Sistem poin sementara (Capstone 1):
    - Default 50 poin per botol valid
    - Nanti: beda per brand (aqua / le_mineral / coca_cola)
"""

from __future__ import annotations

import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

# ── Konstanta poin ────────────────────────────────────────────────────────────

POINTS_DEFAULT = 50          # poin per botol valid (tanpa brand detection)

# Nantinya bisa diganti dict per brand:
# POINTS_BY_BRAND = {"aqua": 50, "le_mineral": 40, "coca_cola": 45, "unknown": 30}

# ── Katalog SKU barcode → nama produk ─────────────────────────────────────────
# Isi saat barcode scanner sudah terintegrasi.
# Key  : string SKU dari hasil scan barcode (mis. "8996001101055")
# Value: nama produk yang ditampilkan di UI
#
# Cara menambahkan produk baru:
#   1. Scan botol dengan barcode scanner, catat SKU-nya
#   2. Tambahkan entry: "SKU_NUMBER": "Nama Produk Volume",
#
SKU_CATALOG: Dict[str, str] = {
    # Contoh (uncomment dan sesuaikan SKU yang sebenarnya):
    # "8996001101055": "Aqua 600ml",
    # "8996001102052": "Aqua 1500ml",
    # "8998007300022": "Le Mineral 600ml",
    # "8998007301029": "Le Mineral 1500ml",
    # "8995004001023": "Coca-Cola 390ml",
    # "8995004002020": "Coca-Cola 1000ml",
}

# ── Alasan invalid yang dikenal ───────────────────────────────────────────────

REASON_BUKAN_BOTOL     = "bukan_botol_pet"     # ✅ aktif sekarang
REASON_BERISI_CAIRAN   = "berisi_cairan"        # 🔜 load cell
REASON_TUTUP_TERPASANG = "tutup_terpasang"      # 🔜 cap model
REASON_BOTOL_RUSAK     = "botol_rusak"          # 🔜 condition model

# Label Indonesia untuk ditampilkan di UI
REASON_LABELS: Dict[str, str] = {
    REASON_BUKAN_BOTOL:     "Bukan botol PET",
    REASON_BERISI_CAIRAN:   "Botol berisi cairan",
    REASON_TUTUP_TERPASANG: "Tutup masih terpasang",
    REASON_BOTOL_RUSAK:     "Botol rusak / penyok",
}


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class BottleResult:
    """Hasil akhir satu botol yang sudah selesai diproses."""

    track_id: int
    decision: str           # "valid" | "invalid"
    reason: Optional[str]   # None jika valid; salah satu REASON_* jika invalid
    confidence: float       # confidence deteksi terakhir (0–1)
    brand: Optional[str]    # None untuk Capstone 1; diisi nanti oleh visual brand model
    points: int             # poin yang diperoleh (0 jika invalid)
    sku: Optional[str] = None          # raw barcode scan result (mis. "8996001101055")
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def reason_label(self) -> str:
        """Label Indonesia dari reason code, untuk ditampilkan di UI."""
        if self.reason is None:
            return ""
        return REASON_LABELS.get(self.reason, self.reason)

    @property
    def display_name(self) -> str:
        """
        Nama identitas botol untuk ditampilkan di UI.

        Fallback chain (prioritas tertinggi ke terendah):
          1. SKU di-lookup di SKU_CATALOG → nama produk lengkap ("Aqua 600ml")
          2. SKU ada tapi tidak dikenal    → tampilkan SKU mentah ("SKU 8996001101055")
          3. Brand dari visual model       → nama brand ("Aqua")
          4. Tidak ada info apapun         → nomor urut botol ("Botol #3")

        Untuk menambahkan produk baru, isi SKU_CATALOG di bagian atas file ini.
        """
        if self.sku and self.sku in SKU_CATALOG:
            return SKU_CATALOG[self.sku]
        if self.sku:
            return f"SKU {self.sku}"
        if self.brand:
            return self.brand.replace("_", " ").title()
        return f"Botol #{self.track_id}"

    def to_dict(self) -> dict:
        """Serialisasi ke dict siap-JSON untuk WebSocket / REST."""
        return {
            "track_id":     self.track_id,
            "decision":     self.decision,
            "reason":       self.reason,
            "reason_label": self.reason_label,
            "confidence":   round(self.confidence, 3),
            "brand":        self.brand,
            "sku":          self.sku,
            "display_name": self.display_name,
            "points":       self.points,
            "timestamp":    self.timestamp.isoformat(),
        }


@dataclass
class SessionSummary:
    """Ringkasan lengkap satu sesi, dikembalikan saat sesi diakhiri."""

    session_id: str
    started_at: datetime
    ended_at: datetime
    bottles: List[BottleResult]
    total_valid: int
    total_invalid: int
    total_points: int

    def to_dict(self) -> dict:
        return {
            "session_id":    self.session_id,
            "started_at":    self.started_at.isoformat(),
            "ended_at":      self.ended_at.isoformat(),
            "total_valid":   self.total_valid,
            "total_invalid": self.total_invalid,
            "total_points":  self.total_points,
            "bottles":       [b.to_dict() for b in self.bottles],
        }


# ── Session Manager ───────────────────────────────────────────────────────────

class SessionManager:
    """
    Thread-safe state manager untuk satu sesi RVM.

    Cara pakai:
        sm = SessionManager()
        sm.start_session()

        # Dari inference thread, setiap botol selesai:
        result = sm.add_bottle(track_id=7, decision=0, confidence=0.93)

        # Dari FastAPI:
        state = sm.get_live_state()

        # Saat user tekan SELESAI:
        summary = sm.end_session()
    """

    def __init__(self, points_per_bottle: int = POINTS_DEFAULT) -> None:
        self._lock             = threading.Lock()
        self._active           = False
        self._session_id       = ""
        self._started_at: Optional[datetime] = None
        self._bottles: List[BottleResult]    = []
        self._total_valid      = 0
        self._total_invalid    = 0
        self._total_points     = 0
        self.points_per_bottle = points_per_bottle

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    @property
    def is_active(self) -> bool:
        return self._active

    def start_session(self) -> str:
        """
        Mulai sesi baru.  Reset semua counter.

        :return: session_id (UUID string)
        """
        with self._lock:
            self._active       = True
            self._session_id   = str(uuid.uuid4())
            self._started_at   = datetime.now()
            self._bottles      = []
            self._total_valid  = 0
            self._total_invalid = 0
            self._total_points = 0
        return self._session_id

    def add_bottle(
        self,
        track_id: int,
        decision: int,                   # 0 = VALID, 1 = INVALID
        confidence: float,
        reason: Optional[str] = None,    # wajib diisi jika decision == 1
        brand: Optional[str] = None,     # dari visual brand model (future)
        sku: Optional[str] = None,       # dari barcode scanner (future)
    ) -> BottleResult:
        """
        Catat satu botol yang baru saja selesai diproses oleh tracker.

        :param track_id:   ID track dari BottleTracker
        :param decision:   0 = VALID, 1 = INVALID
        :param confidence: confidence deteksi terakhir
        :param reason:     kode alasan invalid (salah satu REASON_*); None jika valid
        :param brand:      nama brand dari visual model; None untuk Capstone 1
        :param sku:        hasil scan barcode; None sampai barcode scanner terpasang
        :return:           BottleResult yang baru dicatat
        """
        is_valid = (decision == 0)

        # Tentukan alasan jika invalid dan belum di-set
        if not is_valid and reason is None:
            reason = REASON_BUKAN_BOTOL

        points = self.points_per_bottle if is_valid else 0

        result = BottleResult(
            track_id=track_id,
            decision="valid" if is_valid else "invalid",
            reason=reason if not is_valid else None,
            confidence=confidence,
            brand=brand,
            sku=sku,
            points=points,
        )

        with self._lock:
            self._bottles.append(result)
            if is_valid:
                self._total_valid  += 1
                self._total_points += points
            else:
                self._total_invalid += 1

        return result

    def end_session(self) -> SessionSummary:
        """
        Akhiri sesi aktif dan kembalikan ringkasan lengkap.

        :raises RuntimeError: jika tidak ada sesi aktif
        """
        with self._lock:
            if not self._active:
                raise RuntimeError("Tidak ada sesi aktif untuk diakhiri.")

            summary = SessionSummary(
                session_id    = self._session_id,
                started_at    = self._started_at,
                ended_at      = datetime.now(),
                bottles       = list(self._bottles),
                total_valid   = self._total_valid,
                total_invalid = self._total_invalid,
                total_points  = self._total_points,
            )
            self._active = False

        return summary

    def get_live_state(self) -> dict:
        """
        State sesi saat ini untuk dikirim ke WebSocket/REST.

        :return: dict dengan total_valid, total_invalid, total_points,
                 is_active, session_id, dan daftar botol terakhir (maks 20)
        """
        with self._lock:
            return {
                "is_active":     self._active,
                "session_id":    self._session_id,
                "total_valid":   self._total_valid,
                "total_invalid": self._total_invalid,
                "total_points":  self._total_points,
                # Kirim 20 botol terakhir ke UI (terbaru di atas)
                "recent_bottles": [b.to_dict() for b in reversed(self._bottles[-20:])],
            }
