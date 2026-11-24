# app.py - MammoScan AI (complete, consolidated)
# Pages : /  (login)  /app  /exam  /bulk  /settings
# API   : /login /logout /health /config /predict /predict4 /export
#         /api/settings  /api/feedback  /api/feedback/export
#         /api/report4  /api/case/{report_id}

import os, io, re, json, time, hmac, hashlib, secrets, tempfile, base64, asyncio
from typing import Optional
from urllib.parse import quote

import numpy as np, cv2, onnxruntime as ort
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from preprocess import preprocess
from heatmap import make_overlay
from report import build_report


# =====================================================================
# Configuration
# =====================================================================
MODEL_FILE      = os.getenv("MODEL_FILE", "mammo_ensemble_cam.onnx")
MODEL_VERSION   = os.getenv("MODEL_VERSION", "effnetv2s-ensemble5-cam-v1")
COOKIE_SECURE   = os.getenv("COOKIE_SECURE", "true").lower() == "true"   # set false for local http
SETTINGS_BUCKET = os.getenv("SETTINGS_BUCKET", "")                       # set on Cloud Run only
APP_BASE_URL    = os.getenv("APP_BASE_URL", "").rstrip("/")              # e.g. https://ai.leyapharmacz.com
MAX_UPLOAD_MB   = 30

APP_USER    = os.getenv("APP_USER", "leya_admin")
APP_PASS    = os.getenv("APP_PASS", "Leya@0000")
SECRET      = os.getenv("APP_SECRET", "change-this-to-a-long-random-string").encode()
SESSION_HRS = 8

VIEW_KEYS    = ("rcc", "lcc", "rmlo", "lmlo")
REPORT_ID_RE = re.compile(r"MS-\d{8}-\d{6}-[0-9a-f]{8}")
REVIEW_ID_RE = re.compile(r"[A-Za-z0-9-]{8,64}")

sess = ort.InferenceSession(MODEL_FILE, providers=["CPUExecutionProvider"])


# =====================================================================
# Storage helpers (Cloud Storage bucket on Cloud Run, local folder on PC)
# =====================================================================
def _bucket():
    from google.cloud import storage
    return storage.Client().bucket(SETTINGS_BUCKET)


def store_write(name: str, text: str):
    if SETTINGS_BUCKET:
        _bucket().blob(name).upload_from_string(text, content_type="application/json")
    else:
        folder = os.path.dirname(name)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(name, "w", encoding="utf-8") as f:
            f.write(text)


def store_read(name: str):
    if SETTINGS_BUCKET:
        blob = _bucket().blob(name)
        return blob.download_as_text() if blob.exists() else None
    if not os.path.exists(name):
        return None
    with open(name, encoding="utf-8") as f:
        return f.read()


def store_list(prefix: str):
    """Returns the text of every .json file under prefix/."""
    out = []
    if SETTINGS_BUCKET:
        for b in _bucket().list_blobs(prefix=prefix + "/"):
            if b.name.endswith(".json"):
                out.append(b.download_as_text())
    elif os.path.isdir(prefix):
        for fn in os.listdir(prefix):
            if fn.endswith(".json"):
                with open(os.path.join(prefix, fn), encoding="utf-8") as f:
                    out.append(f.read())
    return out


# =====================================================================
# Settings (thresholds)
# =====================================================================
with open("config.json") as f:
    CFG = json.load(f)
DEFAULTS = {"threshold": float(CFG.get("threshold", 0.1875)),
            "high_priority": float(CFG.get("high_priority", 0.80))}
_settings_cache = {"t": 0.0, "v": None}


def get_settings() -> dict:
    # refresh every 30 s so all Cloud Run instances pick up changes
    if _settings_cache.get("v") is None or time.time() - _settings_cache.get("t") > 30:
        try:
            raw = store_read("settings.json")
            saved = json.loads(raw) if raw else {}
            merged = dict(DEFAULTS)
            merged.update(saved)
            _settings_cache.update(v=merged)
        except Exception:
            _settings_cache.update(v=_settings_cache.get("v") or dict(DEFAULTS))
        _settings_cache.update(t=time.time())
    return _settings_cache.get("v")


def save_settings(s: dict) -> dict:
    s = dict(s)
    s.update(updated_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    store_write("settings.json", json.dumps(s))
    _settings_cache.update(t=time.time(), v=s)
    return s


def band_for(p: float, s: Optional[dict] = None):
    s = s or get_settings()
    if p >= s.get("high_priority"):
        return ("high", "High priority – review first",
                "Score is in the high-priority band. Prioritise this case for radiologist review.")
    if p >= s.get("threshold"):
        return ("review", "Flag for radiologist review",
                "Score is above the screening threshold. Please review this case.")
    return ("low", "Low suspicion – routine reading",
            "Score is below the screening threshold. Routine radiologist reading is still required.")


# =====================================================================
# Model helpers
# =====================================================================
def to_data_url(image: np.ndarray) -> str:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def to_input(imgs):
    """List of HxW float images (0-1) -> N x 3 x H x W normalised batch."""
    batch = np.stack([(im - 0.5) / 0.25 for im in imgs])
    return np.repeat(batch.reshape(len(imgs), 1, *batch.shape[1:]), 3, axis=1).astype(np.float32)


def run_batch(imgs):
    """Returns (list of probabilities, cams array or None)."""
    outputs = sess.run(None, {"image": to_input(imgs)})
    probs = [float(v) for v in np.ravel(outputs.__getitem__(0))]
    cams = outputs.__getitem__(1) if len(outputs) > 1 else None
    return probs, cams


def preprocess_bytes(data: bytes, filename: str):
    suffix = os.path.splitext(filename or "")[1] or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        path = f.name
    try:
        return preprocess(path).astype(np.float32) / 65535
    finally:
        os.remove(path)


# =====================================================================
# Auth
# =====================================================================
def make_token(user: str) -> str:
    exp = str(int(time.time()) + SESSION_HRS * 3600)
    sig = hmac.new(SECRET, (user + "|" + exp).encode(), hashlib.sha256).hexdigest()
    return user + "|" + exp + "|" + sig


def session_user(request: Request):
    parts = request.cookies.get("session", "").split("|")
    if len(parts) != 3:
        return None
    user, exp, sig = parts
    good = hmac.new(SECRET, (user + "|" + exp).encode(), hashlib.sha256).hexdigest()
    try:
        if hmac.compare_digest(sig, good) and int(exp) > time.time():
            return user
    except ValueError:
        pass
    return None


def logged_in(request: Request) -> bool:
    return session_user(request) is not None


def require_login(request: Request) -> str:
    user = session_user(request)
    if not user:
        raise HTTPException(401, "Please sign in again.")
    return user


def safe_next(n) -> str:
    n = n or ""
    return n if n.startswith("/") and not n.startswith("//") else "/app"


def page(request: Request, file: str):
    if not logged_in(request):
        target = request.url.path
        if request.url.query:
            target += "?" + request.url.query
        return RedirectResponse("/?next=" + quote(target, safe=""))
    return FileResponse("static/" + file)


class Login(BaseModel):
    username: str
    password: str


app = FastAPI(title="MammoScan AI")


# =====================================================================
# Pages
# =====================================================================
@app.get("/")
def login_page(request: Request):
    if logged_in(request):
        return RedirectResponse(safe_next(request.query_params.get("next")))
    return FileResponse("static/login.html")


@app.post("/login")
def login(body: Login):
    ok = hmac.compare_digest(body.username, APP_USER) and hmac.compare_digest(body.password, APP_PASS)
    if not ok:
        raise HTTPException(401, "Invalid credentials")
    res = JSONResponse({"ok": True})
    res.set_cookie("session", make_token(body.username), httponly=True, secure=COOKIE_SECURE,
                   samesite="lax", max_age=SESSION_HRS * 3600)
    return res


@app.get("/logout")
def logout():
    res = RedirectResponse("/")
    res.delete_cookie("session")
    return res


@app.get("/app")
def main_page(request: Request):
    return page(request, "app.html")


@app.get("/exam")
def exam_page(request: Request):
    return page(request, "exam.html")


@app.get("/bulk")
def bulk_page(request: Request):
    return page(request, "bulk.html")


@app.get("/settings")
def settings_page(request: Request):
    return page(request, "settings.html")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION,
            "storage": "bucket" if SETTINGS_BUCKET else "local"}


@app.get("/config")
def config(request: Request):
    require_login(request)
    s = get_settings()
    return {"threshold": s.get("threshold"), "high_priority": s.get("high_priority")}


# =====================================================================
# Settings API
# =====================================================================
class SettingsIn(BaseModel):
    threshold: float
    high_priority: float


@app.get("/api/settings")
def read_settings(request: Request):
    require_login(request)
    out = dict(get_settings())
    out.update(defaults=DEFAULTS)
    return out


@app.post("/api/settings")
def write_settings(request: Request, body: SettingsIn):
    require_login(request)
    t, h = round(body.threshold, 4), round(body.high_priority, 4)
    if not (0.01 <= t < h <= 0.99):
        raise HTTPException(400, "Review threshold must be lower than the high-priority threshold (both between 1% and 99%).")
    out = dict(save_settings({"threshold": t, "high_priority": h}))
    out.update(defaults=DEFAULTS)
    return out


# =====================================================================
# Single-image prediction  (/predict?light=1 is used by bulk analysis)
# =====================================================================
@app.post("/predict")
async def predict(request: Request, file: UploadFile = File(...)):
    require_login(request)
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, "File too large (max %d MB)." % MAX_UPLOAD_MB)
    case_id = hashlib.sha256(data).hexdigest().__getitem__(slice(0, 20))

    try:
        img = await run_in_threadpool(preprocess_bytes, data, file.filename)
    except Exception:
        raise HTTPException(400, "Could not read this image. Upload a DICOM, PNG, JPG or TIFF mammogram.")

    probs, cams = await run_in_threadpool(run_batch, [img])
    p = probs.__getitem__(0)
    s = get_settings()
    band, title, message = band_for(p, s)

    result = {
        "case_id": case_id,
        "model_version": MODEL_VERSION,
        "abnormal_probability": round(p, 4),
        "band": band,
        "title": title,
        "message": message,
        "flagged": p >= s.get("threshold"),
        "threshold": s.get("threshold"),
        "high_priority": s.get("high_priority"),
    }
    if request.query_params.get("light") == "1":
        return result

    result.update(preview=to_data_url((img * 255).astype(np.uint8)))
    if cams is not None:
        overlay = await run_in_threadpool(make_overlay, img, cams.__getitem__(0))
        result.update(heatmap=to_data_url(overlay))
    return result


# =====================================================================
# 4-view exam  (parallel preprocessing + one batched model call)
# =====================================================================
def _prep_view(key, data, filename):
    try:
        return key, preprocess_bytes(data, filename)
    except Exception:
        return key, None


def _render_view(key, img, cam, p, s):
    gray = (img * 255).astype(np.uint8)
    heat = make_overlay(img, cam) if cam is not None else None
    if key.startswith("r"):                       # hanging protocol: mirror right breast
        gray = cv2.flip(gray, 1)
        if heat is not None:
            heat = cv2.flip(heat, 1)
    out = {"score": round(p, 4), "band": band_for(p, s).__getitem__(0), "preview": to_data_url(gray)}
    if heat is not None:
        out.update(heatmap=to_data_url(heat))
    return key, out


@app.post("/predict4")
async def predict4(request: Request,
                   rcc: Optional[UploadFile] = File(None),
                   lcc: Optional[UploadFile] = File(None),
                   rmlo: Optional[UploadFile] = File(None),
                   lmlo: Optional[UploadFile] = File(None)):
    require_login(request)
    s = get_settings()
    uploads = {"rcc": rcc, "lcc": lcc, "rmlo": rmlo, "lmlo": lmlo}

    raw = []
    for key, up in uploads.items():
        if up is None:
            continue
        data = await up.read()
        if not data:
            continue
        if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
            raise HTTPException(413, key.upper() + ": file too large (max %d MB)." % MAX_UPLOAD_MB)
        raw.append((key, data, up.filename))
    if not raw:
        raise HTTPException(400, "Please upload at least one view.")

    prepped = await asyncio.gather(*(run_in_threadpool(_prep_view, k, d, fn) for k, d, fn in raw))
    bad = [k.upper() for k, im in prepped if im is None]
    if bad:
        raise HTTPException(400, ", ".join(bad) + ": could not read this image.")
    keys = [k for k, _ in prepped]
    imgs = [im for _, im in prepped]

    probs, cams = await run_in_threadpool(run_batch, imgs)

    jobs = []
    for i, key in enumerate(keys):
        cam = cams.__getitem__(i) if cams is not None else None
        jobs.append(run_in_threadpool(_render_view, key, imgs.__getitem__(i), cam, probs.__getitem__(i), s))
    views = dict(await asyncio.gather(*jobs))

    breasts = {}
    for name, pair in (("right", ("rcc", "rmlo")), ("left", ("lcc", "lmlo"))):
        scores = [views.get(k).get("score") for k in pair if k in views]
        if scores:
            bp = max(scores)
            breasts.update({name: {"score": bp, "band": band_for(bp, s).__getitem__(0)}})

    ep = max(b.get("score") for b in breasts.values())
    band, title, message = band_for(ep, s)
    return {
        "views": views,
        "breasts": breasts,
        "exam": {"score": ep, "band": band, "title": title, "message": message},
        "threshold": s.get("threshold"),
        "high_priority": s.get("high_priority"),
        "model_version": MODEL_VERSION,
    }


# =====================================================================
# Excel helpers
# =====================================================================
def _style_sheet(ws):
    from openpyxl.styles import Font, PatternFill, Alignment
    pink = PatternFill("solid", fgColor="E75480")
    for col in range(1, ws.max_column + 1):
        c = ws.cell(row=1, column=col)
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = pink
        c.alignment = Alignment(horizontal="center")
        width = 10
        for row in range(1, ws.max_row + 1):
            width = max(width, len(str(ws.cell(row=row, column=col).value or "")) + 2)
        ws.column_dimensions.__getitem__(c.column_letter).width = min(70, width)
    ws.freeze_panes = "A2"


def _xlsx_response(wb, prefix):
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    name = "%s_%s.xlsx" % (prefix, time.strftime("%Y%m%d_%H%M%S"))
    return StreamingResponse(buf, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": 'attachment; filename="%s"' % name})


# =====================================================================
# Bulk export
# =====================================================================
class ExportIn(BaseModel):
    rows: list


@app.post("/export")
def export_excel(request: Request, body: ExportIn):
    require_login(request)
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill

    s = get_settings()
    band_fill = {"high": "FEE4E2", "review": "FEF0C7", "low": "D1FADF"}
    band_name = {"high": "High priority", "review": "Review", "low": "Low suspicion"}

    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    ws.append(["#", "File name", "Score (%)", "Band", "Status", "Error"])
    for i, r in enumerate(body.rows, 1):
        r = r if isinstance(r, dict) else {}
        sc = r.get("score")
        ws.append([i, r.get("file", ""), round(sc * 100, 1) if isinstance(sc, (int, float)) else None,
                   band_name.get(r.get("band"), ""), r.get("status", ""), r.get("error", "")])
        if r.get("band") in band_fill:
            ws.cell(row=i + 1, column=4).fill = PatternFill("solid", fgColor=band_fill.get(r.get("band")))

    rows = [r for r in body.rows if isinstance(r, dict)]
    counts = {b: sum(1 for r in rows if r.get("band") == b) for b in ("high", "review", "low")}
    ws2 = wb.create_sheet("Summary")
    for row in (["Item", "Value"],
                ["Generated (UTC)", time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())],
                ["Images", len(rows)],
                ["High priority", counts.get("high")], ["Review", counts.get("review")],
                ["Low suspicion", counts.get("low")],
                ["Errors", sum(1 for r in rows if r.get("status") == "error")],
                ["Review threshold (%)", round(s.get("threshold") * 100, 2)],
                ["High-priority threshold (%)", round(s.get("high_priority") * 100, 2)],
                ["Note", "Decision support only. All images require radiologist review."]):
        ws2.append(row)

    _style_sheet(ws)
    _style_sheet(ws2)
    return _xlsx_response(wb, "MammoScan_Bulk")


# =====================================================================
# Radiologist feedback (single image + 4-view exam)
# =====================================================================
FEEDBACK_FIELDS = ("review_id", "file_name", "report_id", "ai_score", "ai_band", "threshold", "high_priority",
                   "verdict", "birads", "density", "abnormal_views", "ai_views", "reviewer", "comment")


@app.post("/api/feedback")
async def save_feedback(request: Request):
    account = require_login(request)
    body = await request.json()
    if not isinstance(body, dict):
        raise HTTPException(400, "Invalid review.")
    if body.get("verdict") not in ("agree", "disagree"):
        raise HTTPException(400, "Please choose Agree or Disagree.")
    reviewer = str(body.get("reviewer") or "").strip()
    if not reviewer:
        raise HTTPException(400, "Please enter the reviewer name.")
    review_id = str(body.get("review_id") or "")
    if not REVIEW_ID_RE.fullmatch(review_id):
        raise HTTPException(400, "Invalid review ID.")

    rec = {k: body.get(k) for k in FEEDBACK_FIELDS if k in body}
    rec.update(reviewer=reviewer.__getitem__(slice(0, 80)),
               comment=str(body.get("comment") or "").strip().__getitem__(slice(0, 2000)),
               account=account, model_version=MODEL_VERSION,
               saved_at=time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()))
    await run_in_threadpool(store_write, "feedback/" + review_id + ".json", json.dumps(rec))
    return {"ok": True, "saved_at": rec.get("saved_at")}


@app.get("/api/feedback/export")
def export_feedback(request: Request):
    if not logged_in(request):
        return RedirectResponse("/?next=" + quote("/api/feedback/export", safe=""))
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill

    recs = []
    for txt in store_list("feedback"):
        try:
            recs.append(json.loads(txt))
        except Exception:
            pass
    recs.sort(key=lambda r: r.get("saved_at", ""), reverse=True)

    cols = (("saved_at", "Saved (UTC)"), ("reviewer", "Reviewer"), ("file_name", "Case / file"),
            ("ai_score", "AI score (%)"), ("ai_band", "AI band"), ("verdict", "Radiologist verdict"),
            ("birads", "Radiologist assessment"), ("density", "Breast density"),
            ("abnormal_views", "Radiologist abnormal views"), ("ai_views", "AI view scores"),
            ("comment", "Comments"), ("threshold", "Review threshold (%)"),
            ("high_priority", "High-priority threshold (%)"), ("model_version", "Model version"))
    pct_cols = {"ai_score", "threshold", "high_priority"}
    band_name = {"high": "High priority", "review": "Review", "low": "Low suspicion"}

    wb = Workbook()
    ws = wb.active
    ws.title = "Radiologist reviews"
    ws.append([label for _, label in cols])
    for r in recs:
        row = []
        for key, _ in cols:
            v = r.get(key, "")
            if key in pct_cols and isinstance(v, (int, float)):
                v = round(v * 100, 1)
            if key == "ai_band":
                v = band_name.get(v, v)
            if key == "verdict":
                v = {"agree": "Agree", "disagree": "Disagree"}.get(v, v)
            row.append(v)
        ws.append(row)
        fill = "D1FADF" if r.get("verdict") == "agree" else "FEE4E2"
        ws.cell(row=ws.max_row, column=6).fill = PatternFill("solid", fgColor=fill)

    _style_sheet(ws)
    return _xlsx_response(wb, "MammoScan_Reviews")


# =====================================================================
# PDF pre-read report + saved cases (link from PDF back to the app)
# =====================================================================
def _base_url(request: Request) -> str:
    if APP_BASE_URL:
        return APP_BASE_URL
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    host = request.headers.get("x-forwarded-host", request.headers.get("host", ""))
    return proto + "://" + host


@app.post("/api/report4")
async def report4(request: Request):
    require_login(request)
    d = await request.json()
    if not isinstance(d, dict) or not d.get("views"):
        raise HTTPException(400, "Run the analysis before generating the report.")

    report_id = time.strftime("MS-%Y%m%d-%H%M%S-") + secrets.token_hex(4)
    ref = str(d.get("case_ref") or "").strip().__getitem__(slice(0, 40))
    case = {
        "report_id": report_id,
        "case_ref": ref,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "model_version": MODEL_VERSION,
        "views": d.get("views"),
        "breasts": d.get("breasts"),
        "exam": d.get("exam"),
        "threshold": d.get("threshold"),
        "high_priority": d.get("high_priority"),
    }
    await run_in_threadpool(store_write, "cases/" + report_id + ".json", json.dumps(case))

    pdf_data = dict(case)
    pdf_data.update(include_original=bool(d.get("include_original", True)),
                    case_url=_base_url(request) + "/exam?report=" + report_id)
    try:
        pdf, _ = await run_in_threadpool(build_report, pdf_data)
    except Exception as e:
        raise HTTPException(500, "Could not build the report (%s: %s)." % (type(e).__name__, e))
    return Response(pdf, media_type="application/pdf",
                    headers={"Content-Disposition": 'attachment; filename="MammoScan_Report_%s.pdf"' % report_id,
                             "X-Report-Id": report_id,
                             "Access-Control-Expose-Headers": "Content-Disposition, X-Report-Id"})


@app.get("/api/case/{report_id}")
async def get_case(report_id: str, request: Request):
    require_login(request)
    if not REPORT_ID_RE.fullmatch(report_id):
        raise HTTPException(400, "Invalid report ID.")
    raw = await run_in_threadpool(store_read, "cases/" + report_id + ".json")
    if not raw:
        raise HTTPException(404, "This case was not found. It may have been deleted.")
    return json.loads(raw)


# ---------- Google Search Console verification (must be public, no login) ----------
@app.get("/googlea592d3488418fff3.html")
def google_site_verification():
    return Response("google-site-verification: googlea592d3488418fff3.html",
                    media_type="text/html")

app.mount("/static", StaticFiles(directory="static"), name="static")
