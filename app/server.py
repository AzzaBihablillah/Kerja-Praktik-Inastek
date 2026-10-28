"""Server aplikasi kiosk RVM (FastAPI). Tiga antarmuka dari satu server:
  /            kiosk layar sentuh di mesin (WebSocket /ws untuk kejadian langsung)
  /k/<kode>    halaman HP setelah memindai QR: simpan poin ke akun atau donasikan
  /akun        dompet poin, tukar hadiah (simulasi), donasi
  /operator    status mesin, data kalibrasi per botol, tombol simulasi
Jalankan lewat `python -m app` (lihat app/README.md)."""
import asyncio
import json
import os
import socket
import time
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config
from .db import DB, DEFAULT_CAMPAIGN, AppError
from .engine import Machine
from .plc import Conveyor

try:
    import segno
except ImportError:                     # tanpa segno: kiosk tetap menampilkan kode klaim untuk diketik
    segno = None

COOKIE = "rvm_auth"
OPTS = dict(sim=False, backend=config.BACKEND, profile=config.PROFILE, cam=config.CAMERA_INDEX, plc=False,
            port=config.PORT, public_url=config.PUBLIC_BASE_URL)

db = None          # type: Optional[DB]
machine = None     # type: Optional[Machine]
conveyor = None    # type: Optional[Conveyor]
_clients = set()
_queue = None
_loop = None


# ---------------------------------------------------------------------------- util
def lan_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))     # tidak mengirim paket; hanya memilih antarmuka jaringan
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def base_url():
    return (OPTS["public_url"] or "http://%s:%d" % (lan_ip(), OPTS["port"])).rstrip("/")


def qr_svg(text):
    if segno is None:
        return ""
    return segno.make(text, error="m").svg_inline(omitsize=True, border=0, dark="#16201b", light=None)


def claim_public(cl, with_qr=False):
    if not cl:
        return None
    d = dict(code=cl["code"], code_fmt=cl["code"][:3] + "-" + cl["code"][3:], points=cl["points"], grams=cl["grams"],
             status=cl["status"], expires=cl["expires"], url="%s/k/%s" % (base_url(), cl["code"]))
    if cl["status"] == "open" and cl["expires"] < time.time():
        d["status"] = "expired"
    if cl.get("campaign_id"):
        c = db.one("SELECT nama FROM campaigns WHERE id=?", (cl["campaign_id"],))
        d["campaign"] = c["nama"] if c else None
    if with_qr:
        d["qr"] = qr_svg(d["url"])
    return d


def emit(ev):
    """Dipanggil dari thread mesin: titipkan ke loop asyncio."""
    if _loop is not None and _queue is not None:
        if ev.get("type") == "session_end" and ev.get("claim"):
            ev = dict(ev, claim=claim_public(ev["claim"], with_qr=True))
        _loop.call_soon_threadsafe(_queue.put_nowait, ev)


async def _broadcast_loop():
    while True:
        ev = await _queue.get()
        msg = json.dumps(ev, ensure_ascii=False, default=float)
        for ws in list(_clients):
            try:
                await ws.send_text(msg)
            except Exception:
                _clients.discard(ws)


async def _housekeeping():
    while True:
        try:
            s = machine.session
            for row, cl in db.recover_sessions(config.RECOVER_IDLE_SEC, skip_id=s["id"] if s else 0):
                print("[pemulihan] sesi #%d ditutup (%d poin)%s" % (row["id"], row["points"], ", kode klaim " + cl["code"] if cl else ""))
            n = db.expire_claims()
            if n:
                print("[klaim] %d klaim kedaluwarsa didonasikan otomatis" % n)
        except Exception as e:
            print("[klaim] galat:", e)
        await asyncio.sleep(60)


@asynccontextmanager
async def lifespan(app):
    global db, machine, conveyor, _queue, _loop
    _loop = asyncio.get_running_loop(); _queue = asyncio.Queue()
    db = DB()
    conveyor = Conveyor(enabled=OPTS["plc"])
    machine = Machine(db, conveyor, emit, sim=OPTS["sim"], backend=OPTS["backend"], profile=OPTS["profile"], cam=OPTS["cam"])
    machine.start()
    tasks = [asyncio.ensure_future(_broadcast_loop()), asyncio.ensure_future(_housekeeping())]
    print("[app] kiosk    : http://localhost:%d/" % OPTS["port"])
    print("[app] operator : http://localhost:%d/operator" % OPTS["port"])
    print("[app] QR klaim : %s/k/<kode>  (HP harus satu jaringan)" % base_url())
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        await _loop.run_in_executor(None, machine.stop)
        conveyor.close()


app = FastAPI(title="RVM Kiosk", lifespan=lifespan, docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")


@app.middleware("http")
async def _static_no_cache(request, call_next):
    """CSS/JS selalu divalidasi ulang (ETag): kiosk tidak menampilkan versi lama setelah aplikasi diperbarui."""
    resp = await call_next(request)
    if request.url.path.startswith("/static/"):
        resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.exception_handler(AppError)
async def _app_error(request, exc):
    return JSONResponse({"ok": False, "pesan": exc.msg}, status_code=exc.status)


def page(name):
    return FileResponse(os.path.join(config.STATIC_DIR, name), headers={"Cache-Control": "no-cache"})


def need_user(request):
    u = db.user_by_token(request.cookies.get(COOKIE))
    if not u:
        raise AppError("Silakan masuk dulu.", 401)
    return u


def set_auth(resp, tok):
    resp.set_cookie(COOKIE, tok, max_age=config.AUTH_DAYS * 86400, httponly=True, samesite="lax")
    return resp


# ---------------------------------------------------------------------------- halaman
@app.get("/")
def kiosk_page():
    return page("kiosk.html")


@app.get("/operator")
def operator_page():
    return page("operator.html")


@app.get("/k")
@app.get("/k/{code}")
@app.get("/akun")
def mobile_page(code: str = ""):
    return page("m.html")


@app.get("/video_feed")
async def video_feed():
    async def gen():
        last = None
        while True:
            f = machine.jpeg
            if f and f is not last:
                last = f
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n"
            await asyncio.sleep(1.0 / config.PREVIEW_FPS)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept(); _clients.add(ws)
    try:
        summ = machine.last_summary
        if summ and time.time() - summ["ts"] > config.SUMMARY_SEC:
            summ = None
        if summ and summ.get("claim"):
            summ = dict(summ, claim=claim_public(db.claim_get(summ["claim"]["code"]), with_qr=True))
        await ws.send_text(json.dumps(dict(type="init", machine=machine.machine_status(), session=machine.session_public(),
                                           summary=summ, idle_max=config.IDLE_TIMEOUT, summary_sec=config.SUMMARY_SEC,
                                           claim_link_sec=config.CLAIM_LINK_SEC, poin=config.POINTS_BY_SIZE),
                                      ensure_ascii=False, default=float))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        _clients.discard(ws)


# ---------------------------------------------------------------------------- kiosk
@app.get("/api/kiosk")
def kiosk_info():
    return dict(stats=db.stats(), kampanye=db.campaigns(), poin=config.POINTS_BY_SIZE, mesin=machine.machine_status(),
                kampanye_default=DEFAULT_CAMPAIGN)


@app.post("/api/sesi/mulai")
def sesi_mulai():
    s, err = machine.start_session()
    if err:
        raise AppError(err, 409)
    return dict(ok=True, session=s)


@app.post("/api/sesi/selesai")
def sesi_selesai():
    summ = machine.end_session("manual")
    if summ is None:
        raise AppError("Tidak ada sesi yang berjalan.", 409)
    return dict(ok=True, session=summ["session"], claim=claim_public(summ["claim"]))


# ---------------------------------------------------------------------------- klaim
class DonasiIn(BaseModel):
    campaign_id: int
    points: Optional[int] = None


@app.get("/api/klaim/{code}")
def klaim_info(code: str):
    cl = db.claim_get(code)
    if not cl:
        raise AppError("Kode klaim tidak ditemukan. Periksa lagi hurufnya.", 404)
    k = claim_public(cl)
    if k["status"] == "open":
        emit(dict(type="claim_opened", code=k["code"]))     # kiosk: HP sudah terhubung, tunggu lebih lama
    return dict(ok=True, klaim=k, kampanye=db.campaigns())


@app.get("/api/klaim/{code}/qr.svg")
def klaim_qr(code: str):
    cl = db.claim_get(code)
    if not cl or segno is None:
        raise AppError("QR tidak tersedia.", 404)
    return Response(qr_svg("%s/k/%s" % (base_url(), cl["code"])), media_type="image/svg+xml")


@app.post("/api/klaim/{code}/simpan")
def klaim_simpan(code: str, request: Request):
    u = need_user(request)
    cl = db.claim_to_user(code, u["id"])
    _notify_claim(cl["code"])
    return dict(ok=True, points=cl["points"], saldo=db.balance(u["id"]))


@app.post("/api/klaim/{code}/donasi")
def klaim_donasi(code: str, body: DonasiIn, request: Request):
    u = db.user_by_token(request.cookies.get(COOKIE))
    src = "hp" if "/k/" in (request.headers.get("referer") or "") else "kiosk"
    cl, camp = db.claim_donate(code, body.campaign_id, source=src, uid=u["id"] if u else None)
    _notify_claim(cl["code"])
    return dict(ok=True, points=cl["points"], kampanye=camp["nama"])


def _notify_claim(code):
    """Beri tahu kiosk bahwa QR di layar sudah dipakai (layar ringkasan bisa langsung berterima kasih)."""
    emit(dict(type="claim_update", claim=claim_public(db.claim_get(code))))


# ---------------------------------------------------------------------------- akun
class DaftarIn(BaseModel):
    hp: str
    nama: str
    pin: str


class MasukIn(BaseModel):
    hp: str
    pin: str


@app.post("/api/akun/daftar")
def akun_daftar(body: DaftarIn):
    tok = db.register(body.hp, body.nama, body.pin)
    return set_auth(JSONResponse(dict(ok=True)), tok)


@app.post("/api/akun/masuk")
def akun_masuk(body: MasukIn):
    tok = db.login(body.hp, body.pin)
    return set_auth(JSONResponse(dict(ok=True)), tok)


@app.post("/api/akun/keluar")
def akun_keluar(request: Request):
    db.logout(request.cookies.get(COOKIE))
    r = JSONResponse(dict(ok=True)); r.delete_cookie(COOKIE)
    return r


@app.get("/api/akun")
def akun(request: Request):
    u = db.user_by_token(request.cookies.get(COOKIE))
    if not u:
        return dict(ok=True, masuk=False)
    return dict(ok=True, masuk=True, profil=db.profile(u["id"]))


@app.post("/api/akun/donasi")
def akun_donasi(body: DonasiIn, request: Request):
    u = need_user(request)
    camp = db.donate_from_balance(u["id"], body.campaign_id, body.points)
    return dict(ok=True, kampanye=camp["nama"], saldo=db.balance(u["id"]))


@app.get("/api/hadiah")
def hadiah():
    return dict(ok=True, hadiah=db.rewards(), kampanye=db.campaigns())


@app.post("/api/hadiah/{rid}/tukar")
def hadiah_tukar(rid: int, request: Request):
    u = need_user(request)
    r = db.redeem(u["id"], rid)
    return dict(ok=True, hadiah=r, saldo=db.balance(u["id"]))


# ---------------------------------------------------------------------------- operator
class SimIn(BaseModel):
    size: Optional[str] = None
    cap: bool = False
    liquid: bool = False
    benda: bool = False


class AutoIn(BaseModel):
    on: bool


class KelompokIn(BaseModel):
    isi: str
    tutup: str
    produk: str = ""
    ml: str = ""
    catatan: str = ""


@app.post("/api/kelompok")
def kelompok(body: KelompokIn):
    """Kiosk capturer: label kelompok untuk semua botol berikutnya."""
    if not config.DATA_MODE:
        raise AppError("Hanya tersedia di mode kiosk capturer (--ambil-data).", 400)
    try:
        return dict(ok=True, kelompok=machine.set_batch(body.isi, body.tutup, body.produk, body.ml, body.catatan))
    except ValueError as e:
        raise AppError(str(e))


@app.get("/api/operator")
def operator():
    rows = db.q("SELECT b.*, s.started FROM bottles b LEFT JOIN sessions s ON s.id=b.session_id ORDER BY b.id DESC LIMIT 40")
    sess = db.q("SELECT * FROM sessions ORDER BY id DESC LIMIT 10")
    claims = db.q("SELECT code, points, status, created, expires FROM claims ORDER BY created DESC LIMIT 10")
    return dict(mesin=machine.machine_status(), plc=conveyor.status(), sortir=machine.sortir.status(), benda=machine.benda.status(), timings=getattr(machine.an, "last_timings", {}),
                botol=rows, sesi=sess, klaim=claims, stats=db.stats(), url_publik=base_url(),
                aturan=dict(zona=config.ZONE_X, min_suara=config.MIN_VOTES, nocap_frac=config.NOCAP_FRAC,
                            idle=config.IDLE_TIMEOUT, poin=config.POINTS_BY_SIZE),
                capturer=dict(aktif=config.DATA_MODE, kelompok=machine.batch_public(), isi=config.KELOMPOK_ISI, tutup=config.KELOMPOK_TUTUP,
                              folder=config.SESI_DIR or config.SCAN_SAVE_DIR, kandidat=config.SHADOW))


@app.post("/api/sim/botol")
def sim_botol(body: SimIn):
    if not machine.sim:
        raise AppError("Hanya tersedia di mode simulasi (--sim).", 400)
    if body.size not in (None, "kecil", "tanggung", "besar"):
        raise AppError("Ukuran tidak dikenal.")
    machine.an.spawn(body.size, body.cap, body.liquid, body.benda)
    return dict(ok=True)


@app.post("/api/sim/auto")
def sim_auto(body: AutoIn):
    if not machine.sim:
        raise AppError("Hanya tersedia di mode simulasi (--sim).", 400)
    machine.an.auto = body.on
    return dict(ok=True, auto=body.on)


@app.get("/health")
def health():
    return dict(ok=True, mesin=machine.machine_status(), plc=conveyor.status(), ws=len(_clients))
