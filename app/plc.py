"""Kendali konveyor lewat PLC Siemens (snap7), bit MB10.0 = jalan (sama dengan server PLC lama).
Opsional: tanpa --plc, atau bila snap7 tidak terpasang / PLC tidak terhubung, semua perintah hanya dicatat (mode kering)."""
import threading

from . import config


class Conveyor:
    def __init__(self, enabled=False, ip=config.PLC_IP):
        self.enabled, self.ip = enabled, ip
        self.client = None; self.running = False; self.note = "nonaktif (tanpa --plc)"
        self._lock = threading.Lock()
        if not enabled:
            return
        try:
            import snap7
            self.client = snap7.client.Client()
            self.client.connect(ip, 0, 1)
            self.note = "terhubung" if self.client.get_connected() else "gagal terhubung"
        except Exception as e:                       # snap7 tidak ada di Nano / PLC mati: aplikasi tetap jalan
            self.client = None; self.note = "galat: %s" % e
        print("[plc]", self.ip, self.note)

    @property
    def connected(self):
        try:
            return bool(self.client and self.client.get_connected())
        except Exception:
            return False

    def _reconnect(self):
        try:
            self.client.disconnect()
        except Exception:
            pass
        try:
            self.client.connect(self.ip, 0, 1)
            self.note = "terhubung ulang" if self.client.get_connected() else "gagal terhubung ulang"
        except Exception as e:
            self.note = "gagal terhubung ulang: %s" % e
        print("[plc]", self.note, flush=True)

    def run(self, on):
        """Tulis bit jalan. Bila koneksi putus/gagal (PLC kadang hilang dari jaringan), sambung ulang SEKALI lalu coba lagi.
        True = tertulis; False = PERINTAH TIDAK SAMPAI (anggap keadaan konveyor tidak pasti)."""
        with self._lock:
            self.running = bool(on)
            if self.client is None:
                return False
            for attempt in (1, 2):
                try:
                    if not self.connected:
                        raise RuntimeError("tidak terhubung")
                    from snap7.util import set_bool
                    data = bytearray(self.client.mb_read(config.PLC_BYTE_RUN, 1))
                    set_bool(data, 0, config.PLC_BIT_RUN, bool(on))
                    self.client.mb_write(config.PLC_BYTE_RUN, 1, data)
                    return True
                except Exception as e:
                    print("[plc] tulis gagal (percobaan %d): %s" % (attempt, e), flush=True)
                    if attempt == 1:
                        self._reconnect()
            return False

    def close(self):
        if self.connected:
            self.run(False)
            try:
                self.client.disconnect()
            except Exception:
                pass

    def status(self):
        return dict(aktif=self.enabled, terhubung=self.connected, jalan=self.running, ket=self.note, ip=self.ip)
