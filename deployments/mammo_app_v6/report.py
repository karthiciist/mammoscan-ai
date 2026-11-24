# report.py - builds the AI pre-read PDF for a 4-view exam
import io, os, base64, time
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

PINK = colors.HexColor("#E75480")
PINK_DEEP = colors.HexColor("#C2185B")
PINK_SOFT = colors.HexColor("#FDF0F5")
INK = colors.HexColor("#1D2939")
MUTED = colors.HexColor("#667085")
LINE = colors.HexColor("#F3E1E8")
DARK = colors.HexColor("#0B0B0F")
BAND_COLOR = {"high": colors.HexColor("#D92D20"), "review": colors.HexColor("#F79009"), "low": colors.HexColor("#12B76A")}
BAND_SOFT = {"high": colors.HexColor("#FEF3F2"), "review": colors.HexColor("#FFFAEB"), "low": colors.HexColor("#ECFDF3")}
BAND_NAME = {"high": "High priority", "review": "Review", "low": "Low suspicion"}
KEYS = ("rcc", "lcc", "rmlo", "lmlo")
LABEL = {"rcc": "RCC", "lcc": "LCC", "rmlo": "RMLO", "lmlo": "LMLO"}
W, H = A4
M = 14 * mm


def _img(data_url):
    if not data_url or "," not in data_url:
        return None
    try:
        raw = base64.b64decode(data_url.split(",", 1).__getitem__(1))
        return ImageReader(io.BytesIO(raw))
    except Exception:
        return None


def _pct(v):
    return "%.1f%%" % (float(v or 0) * 100)


def _wrap(c, text, x, y, width, font="Helvetica", size=8, lead=10, color=MUTED):
    c.setFont(font, size)
    c.setFillColor(color)
    words, line = str(text).split(), ""
    for w in words:
        test = (line + " " + w).strip()
        if c.stringWidth(test, font, size) > width:
            c.drawString(x, y, line)
            y -= lead
            line = w
        else:
            line = test
    if line:
        c.drawString(x, y, line)
        y -= lead
    return y


def _header(c, title, page, report_id):
    logo = "static/logo.jpeg"
    if os.path.exists(logo):
        img = ImageReader(logo)
        iw, ih = img.getSize()
        h = 9 * mm
        c.drawImage(img, M, H - M - h, width=h * iw / ih, height=h, mask="auto")
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(PINK)
    c.drawRightString(W - M, H - M - 4 * mm, "PROPRIETARY CLINICAL AI PLATFORM")
    c.setFont("Helvetica", 7.5)
    c.setFillColor(MUTED)
    c.drawRightString(W - M, H - M - 8 * mm, "Report ID " + report_id)

    y = H - M - 14 * mm
    c.setFillColor(PINK)
    c.rect(0, y - 16 * mm, W, 16 * mm, stroke=0, fill=1)
    c.setFillColor(PINK_DEEP)
    c.rect(W * 0.62, y - 16 * mm, W * 0.38, 16 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(M, y - 7 * mm, title)
    c.setFont("Helvetica", 8.5)
    c.drawString(M, y - 12 * mm, "AI pre-read generated before radiologist review - not a diagnosis")
    c.setFont("Helvetica-Bold", 8)
    c.drawRightString(W - M, y - 9.5 * mm, "PAGE %d" % page)
    return y - 16 * mm


def _footer(c):
    c.setStrokeColor(PINK)
    c.setLineWidth(1.5)
    c.line(M, 16 * mm, W - M, 16 * mm)
    _wrap(c, "Decision-support output only. All findings, including low-suspicion results, must be reviewed and "
             "confirmed by a qualified radiologist. This report has not been reviewed by a radiologist.",
          M, 12.5 * mm, W - 2 * M, size=6.8, lead=8)
    c.setFont("Helvetica", 6.5)
    c.setFillColor(MUTED)
    c.drawString(M, 6 * mm, "Proprietary & confidential - Leya Art Ntelznx Private Limited. Unauthorised use or duplication is prohibited.")
    c.drawRightString(W - M, 6 * mm, "(c) %s Leya Art Ntelznx Pvt. Ltd." % time.strftime("%Y"))


def _box(c, x, y, w, h, fill=colors.white, stroke=LINE, r=3 * mm):
    c.setFillColor(fill)
    c.setStrokeColor(stroke)
    c.setLineWidth(0.8)
    c.roundRect(x, y, w, h, r, stroke=1, fill=1)


def _section(c, x, y, text):
    c.setFillColor(PINK)
    c.rect(x, y + 1, 5 * mm, 1.2, stroke=0, fill=1)
    c.setFillColor(colors.HexColor("#57585C"))
    c.setFont("Helvetica-Bold", 8.5)
    c.drawString(x + 7 * mm, y, text.upper())


def _image_grid(c, d, y_top, height, kind):
    gw = (W - 2 * M) / 4
    _box(c, M, y_top - height, W - 2 * M, height, fill=DARK, stroke=DARK, r=2 * mm)
    for i, k in enumerate(KEYS):
        x = M + i * gw
        v = (d.get("views") or {}).get(k)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 8)
        c.drawString(x + 3 * mm, y_top - 5 * mm, LABEL.get(k))
        if not v:
            c.setFillColor(colors.HexColor("#8A8F98"))
            c.setFont("Helvetica", 8)
            c.drawCentredString(x + gw / 2, y_top - height / 2, "Not uploaded")
            continue
        img = _img(v.get(kind) or v.get("preview"))
        box_h = height - 18 * mm
        if img:
            iw, ih = img.getSize()
            s = min((gw - 4 * mm) / iw, box_h / ih)
            dw, dh = iw * s, ih * s
            dx = x + gw - 2 * mm - dw if k.startswith("r") else x + 2 * mm
            c.drawImage(img, dx, y_top - 8 * mm - box_h + (box_h - dh) / 2, width=dw, height=dh)
        band = v.get("band", "low")
        if band in ("high", "review"):
            c.setStrokeColor(BAND_COLOR.get(band))
            c.setLineWidth(2)
            c.rect(x + 1, y_top - height + 1, gw - 2, height - 2, stroke=1, fill=0)
        cw = 34 * mm
        cx = x + (gw - cw) / 2
        c.setFillColor(colors.HexColor("#16161D"))
        c.roundRect(cx, y_top - height + 3 * mm, cw, 6.5 * mm, 3 * mm, stroke=0, fill=1)
        c.setFillColor(BAND_COLOR.get(band))
        c.circle(cx + 4 * mm, y_top - height + 6.25 * mm, 1.2 * mm, stroke=0, fill=1)
        c.setFillColor(colors.white)
        c.setFont("Helvetica-Bold", 7.5)
        c.drawString(cx + 7 * mm, y_top - height + 5.3 * mm, "%s - %s" % (_pct(v.get("score", 0)), BAND_NAME.get(band)))
        if i < 3:
            c.setStrokeColor(colors.HexColor("#2A2A33"))
            c.setLineWidth(3 if i == 1 else 0.6)
            c.line(x + gw, y_top - height, x + gw, y_top)


def _link_button(c, x, y, w, h, url, text):
    """Pink clickable button that opens the case in the app."""
    c.setFillColor(PINK)
    c.roundRect(x, y, w, h, h / 2, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawCentredString(x + w / 2, y + h / 2 - 3, text)
    c.linkURL(url, (x, y, x + w, y + h), relative=0, thickness=0)


def build_report(d):
    report_id = d.get("report_id") or time.strftime("MS-%Y%m%d-%H%M%S")
    url = d.get("case_url") or ""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    c.setTitle("MammoScan AI - Preliminary report " + report_id)
    c.setAuthor("MammoScan AI")

    # ---------------- Page 1 ----------------
    y = _header(c, "MammoScan AI - Preliminary Report", 1, report_id)
    y -= 6 * mm

    gen = time.strftime("%d %b %Y, %H:%M UTC", time.gmtime())
    rows = [("Case reference", d.get("case_ref") or "Not provided"),
            ("Generated", gen),
            ("Views analysed", ", ".join(LABEL.get(k) for k in KEYS if (d.get("views") or {}).get(k)) or "-"),
            ("Model version", d.get("model_version") or "-"),
            ("Thresholds", "Review >= %s   |   High priority >= %s" % (_pct(d.get("threshold", 0)), _pct(d.get("high_priority", 0))))]
    bh = 6.2 * mm * len(rows) + 4 * mm
    _box(c, M, y - bh, (W - 2 * M) * 0.55, bh)
    ry = y - 6 * mm
    for k, v in rows:
        c.setFont("Helvetica", 7.8); c.setFillColor(MUTED); c.drawString(M + 4 * mm, ry, k)
        c.setFont("Helvetica-Bold", 7.8); c.setFillColor(INK); c.drawString(M + 32 * mm, ry, str(v).__getitem__(slice(0, 60)))
        ry -= 6.2 * mm

    ex = d.get("exam") or {}
    band = ex.get("band", "low")
    vx = M + (W - 2 * M) * 0.55 + 4 * mm
    vw = W - M - vx
    _box(c, vx, y - bh, vw, bh, fill=BAND_SOFT.get(band), stroke=BAND_COLOR.get(band))
    c.setFillColor(BAND_COLOR.get(band))
    c.setFont("Helvetica-Bold", 7.5)
    c.drawString(vx + 4 * mm, y - 6 * mm, "OVERALL AI RESULT")
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(INK)
    c.drawString(vx + 4 * mm, y - 16 * mm, _pct(ex.get("score", 0)))
    c.setFillColor(BAND_COLOR.get(band))
    c.roundRect(vx + 4 * mm, y - 23 * mm, 30 * mm, 5 * mm, 2.5 * mm, stroke=0, fill=1)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 7)
    c.drawCentredString(vx + 19 * mm, y - 21.4 * mm, BAND_NAME.get(band).upper())
    _wrap(c, ex.get("title", ""), vx + 4 * mm, y - 28.5 * mm, vw - 8 * mm, font="Helvetica-Bold", size=8, color=INK)
    y -= bh + 8 * mm

    _section(c, M, y, "Heat map - areas that most influenced the score")
    y -= 4 * mm
    grid_h = 92 * mm
    _image_grid(c, d, y, grid_h, "heatmap")
    y -= grid_h + 8 * mm

    _section(c, M, y, "Scores by breast and view")
    y -= 5 * mm
    col_w = (W - 2 * M - 4 * mm) / 2
    for bi, (bname, pair) in enumerate((("Right breast", ("rcc", "rmlo")), ("Left breast", ("lcc", "lmlo")))):
        bx = M + bi * (col_w + 4 * mm)
        b = (d.get("breasts") or {}).get("right" if bi == 0 else "left")
        _box(c, bx, y - 30 * mm, col_w, 30 * mm)
        c.setFont("Helvetica-Bold", 9); c.setFillColor(INK); c.drawString(bx + 4 * mm, y - 6 * mm, bname)
        if b:
            bb = b.get("band", "low")
            c.setFont("Helvetica-Bold", 11); c.drawRightString(bx + col_w - 4 * mm, y - 6.3 * mm, _pct(b.get("score", 0)))
            c.setFillColor(BAND_COLOR.get(bb)); c.setFont("Helvetica-Bold", 7)
            c.drawRightString(bx + col_w - 24 * mm, y - 6 * mm, BAND_NAME.get(bb).upper())
        ly = y - 14 * mm
        for k in pair:
            v = (d.get("views") or {}).get(k)
            c.setFont("Helvetica-Bold", 8); c.setFillColor(INK); c.drawString(bx + 4 * mm, ly, LABEL.get(k))
            tx, tw = bx + 18 * mm, col_w - 40 * mm
            c.setFillColor(colors.HexColor("#F2F4F7")); c.roundRect(tx, ly - 0.5 * mm, tw, 2.6 * mm, 1.3 * mm, stroke=0, fill=1)
            if v:
                vb = v.get("band", "low")
                c.setFillColor(BAND_COLOR.get(vb))
                c.roundRect(tx, ly - 0.5 * mm, max(2.6 * mm, tw * float(v.get("score", 0))), 2.6 * mm, 1.3 * mm, stroke=0, fill=1)
                c.setFont("Helvetica-Bold", 8); c.drawRightString(bx + col_w - 4 * mm, ly, _pct(v.get("score", 0)))
            else:
                c.setFont("Helvetica", 7.5); c.setFillColor(MUTED); c.drawRightString(bx + col_w - 4 * mm, ly, "Not uploaded")
            ly -= 8 * mm
    y -= 38 * mm

    _section(c, M, y, "Radiologist review")
    if url:
        _link_button(c, W - M - 62 * mm, y - 2.2 * mm, 62 * mm, 7.5 * mm, url, "Open case in MammoScan AI  >")
    y -= 6 * mm
    _box(c, M, y - (33 if url else 28) * mm, W - 2 * M, (33 if url else 28) * mm, fill=PINK_SOFT)
    c.setFont("Helvetica-Bold", 9); c.setFillColor(PINK_DEEP)
    c.drawString(M + 4 * mm, y - 6.5 * mm, "PENDING - to be completed by the reporting radiologist")
    c.setFont("Helvetica", 8); c.setFillColor(INK)
    fields = (("Agree with AI:   Yes  /  No", "BI-RADS: ________"),
              ("Abnormal views: ______________________", "Density (A-D): ______"),
              ("Radiologist: ______________________", "Signature / date: ______________________"))
    fy = y - 13 * mm
    for a, b in fields:
        c.drawString(M + 4 * mm, fy, a)
        c.drawString(M + (W - 2 * M) / 2 + 4 * mm, fy, b)
        fy -= 6 * mm
    if url:
        lead = "Review online: "
        c.setFont("Helvetica", 6.8)
        c.setFillColor(MUTED)
        c.drawString(M + 4 * mm, y - 30.5 * mm, lead)
        x0 = M + 4 * mm + c.stringWidth(lead, "Helvetica", 6.8)
        c.setFillColor(PINK_DEEP)
        c.drawString(x0, y - 30.5 * mm, url)
        tw = c.stringWidth(url, "Helvetica", 6.8)
        c.linkURL(url, (x0, y - 31.5 * mm, x0 + tw, y - 28 * mm), relative=0, thickness=0)

    _footer(c)
    c.showPage()

    # ---------------- Page 2: originals ----------------
    if d.get("include_original", True):
        y = _header(c, "MammoScan AI - Original Views", 2, report_id)
        y -= 6 * mm
        _section(c, M, y, "Processed images as analysed by the model (no overlay)")
        y -= 4 * mm
        big_h = 150 * mm
        _image_grid(c, d, y, big_h, "preview")
        y -= big_h + 8 * mm
        _section(c, M, y, "How to read this report")
        y -= 6 * mm
        notes = ("Scores are the model's estimated probability that a view is abnormal. Each breast takes the higher of its CC and MLO scores; the exam takes the higher breast.",
                 "Heat maps show regions that most influenced the score. They are coarse attention maps, not lesion outlines, and always highlight something even when the score is low.",
                 "Bands: Low suspicion - below the review threshold, routine reading still required. Review - radiologist review recommended. High priority - review first.",
                 "Images are downscaled for analysis; fine detail such as microcalcifications must be assessed on the original diagnostic images.")
        for n in notes:
            c.setFillColor(PINK); c.circle(M + 1.5 * mm, y + 1 * mm, 0.8 * mm, stroke=0, fill=1)
            y = _wrap(c, n, M + 4 * mm, y, W - 2 * M - 4 * mm, size=8, lead=10.5, color=INK) - 2 * mm
        _footer(c)
        c.showPage()

    c.save()
    return buf.getvalue(), report_id
