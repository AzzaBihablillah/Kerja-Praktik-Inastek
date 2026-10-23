"""python -m app [--sim] [--backend pytorch|tensorrt|openvino|onnx] [--cam N] [--plc] [--terus] [--jeda S] [--kandidat] [--port 8000] [--url http://IP:8000]
Jalankan dari folder proyek (induk app/ dan rvm_kp/)."""
import argparse
import os

from . import config


def main():
    ap = argparse.ArgumentParser(prog="python -m app", description="Kiosk RVM: layar mesin, klaim poin lewat QR, akun & donasi.")
    ap.add_argument("--sim", action="store_true", help="mode simulasi: tanpa kamera/model/PLC")
    ap.add_argument("--backend", default=config.BACKEND, help="pytorch (laptop) | tensorrt (Jetson Nano) | openvino | onnx")
    ap.add_argument("--profil", default=config.PROFILE, help="profil rvm_kp: nano (default) | laptop")
    ap.add_argument("--cam", type=int, default=config.CAMERA_INDEX, help="index kamera (laptop: 1 = C922; Nano: 0)")
    ap.add_argument("--plc", action="store_true", help="kendalikan konveyor lewat PLC. HANYA setelah area belt dipastikan aman")
    ap.add_argument("--terus", action="store_true", help="konveyor jalan terus (tanpa berhenti-lalu-pindai)")
    ap.add_argument("--jeda", type=float, default=config.STOP_SETTLE_SEC, help="detik menunggu botol diam setelah konveyor berhenti")
    ap.add_argument("--no-simpan", action="store_true", help="jangan simpan frame diam tiap botol")
    ap.add_argument("--kandidat", action="store_true", help="nilai juga dengan model kandidat (bayangan, hanya dicatat; layar tetap model aktif)")
    ap.add_argument("--host", default=config.HOST)
    ap.add_argument("--port", type=int, default=config.PORT)
    ap.add_argument("--url", default=config.PUBLIC_BASE_URL, help="alamat untuk QR klaim, mis. http://192.168.0.20:8000")
    a = ap.parse_args()

    if a.terus:
        config.STOP_SCAN = False
    config.STOP_SETTLE_SEC = a.jeda
    if a.no_simpan:
        config.SCAN_SAVE_DIR = ""
    if a.kandidat:
        config.SHADOW = True
    os.environ.setdefault("YOLO_AUTOINSTALL", "False")       # cegah Ultralytics memasang paket sendiri
    from . import server
    server.OPTS.update(sim=a.sim, backend=a.backend, profile=a.profil, cam=a.cam, plc=a.plc, port=a.port, public_url=a.url)
    import uvicorn
    uvicorn.run(server.app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
