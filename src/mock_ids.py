"""Procedural generator for synthetic, PII-free identity documents.

Every document is drawn from scratch with OpenCV: gradient background,
guilloche line work, a cartoon avatar (no photos of real people), random
syllable-based names, MRZ lines with ICAO-style check digits, barcode
strip, hologram and signature scribble.

`generate_mock_id` returns the BGR image plus ROI hints (portrait, MRZ,
text fields ...) which the forgery generator uses to place tampered
regions over semantically meaningful parts of the document.
"""

from __future__ import annotations

import string
from dataclasses import dataclass, field

import cv2
import numpy as np

WIDTH = 768
HEIGHT = 480

STATES = ["USA", "GBR", "FRA", "DEU", "IND", "JPN", "BRA", "CAN", "AUS", "ESP"]
ORGS = [
    "BUREAU OF CIVIL IDENTIFICATION",
    "NATIONAL DOCUMENT AUTHORITY",
    "INSTITUTE OF IDENTITY RECORDS",
    "CENTRAL REGISTRY OF PERSONS",
    "MINISTRY OF INTERNAL AFFAIRS",
    "OFFICE OF SECURE DOCUMENTS",
]
DOC_CODES = ["ID", "P", "DL", "AN"]
SEXES = ["M", "F", "X"]

_SYLLABLES = [
    "mar", "go", "rin", "va", "shi", "tha", "ach", "an", "ta", "leo", "ver",
    "di", "car", "men", "sof", "ia", "joh", "nn", "smi", "th", "wei", "ling",
    "kum", "ar", "no", "to", "mi", "sa", "to", " brow", "ny", "chen", "yam",
    "ada", "mu", "rez", "garc", "ia", "kle", "in", "ha", "mil", "ton", "zho",
    "ng", "pri", "ya", "rao", "kel", "ly", "dur", "and", "san", "tos",
]

SKIN_TONES = [
    (180, 200, 225), (150, 175, 205), (120, 150, 190),
    (100, 130, 170), (80, 110, 150), (65, 95, 135),
]
HAIR_COLORS = [
    (30, 30, 40), (40, 50, 70), (60, 70, 90), (90, 100, 120),
    (40, 60, 100), (70, 70, 160), (110, 110, 130),
]


def _rand_name(rng: np.random.Generator, n_syll: int = 2) -> str:
    parts = [rng.choice(_SYLLABLES) for _ in range(n_syll + 1)]
    name = "".join(parts).replace(" ", "")
    return name[:1].upper() + name[1:].lower()


def _check_digit(s: str) -> str:
    """ICAO 9303 check digit (weights 7,3,1)."""
    weights = [7, 3, 1]
    total = 0
    for i, ch in enumerate(s):
        if ch.isdigit():
            v = int(ch)
        elif ch == "<":
            v = 0
        else:
            v = ord(ch) - ord("A") + 10
        total += v * weights[i % 3]
    return str(total % 10)


def _mrz(rng: np.random.Generator, state: str, surname: str, given: str,
          docnum: str, dob: str, exp: str, sex: str) -> list[str]:
    def pad(s: str) -> str:
        return (s + "<" * 30)[:30]

    def field9(s: str) -> str:
        # ICAO document-number field: 9 chars, '<' filled, never spaces
        return (s + "<<" * 5)[:9]

    line1 = pad(f"ID{state}<<{surname}<<{given}")
    doc_field = field9(docnum)
    # doc#(9) + cd(1) + nat(3) + dob(6) + sex(1) + exp(6) + filler(4) = 30
    l2 = doc_field + _check_digit(doc_field)
    l2 += state + dob + sex + exp
    l2 += "".join(rng.choice(list(string.ascii_uppercase + "<"), size=4))
    line2 = pad(l2[:30])
    line3 = pad(doc_field + _check_digit(doc_field) + "<" * 18
                + _check_digit(l2))
    return [line1, line2, line3]


def _fitted_text(img: np.ndarray, text: str, org: tuple[int, int], max_w: int,
                 font: int, color: tuple, thickness: int = 1,
                 font_scale: float = 1.0) -> tuple[int, int]:
    """Draw text scaled horizontally so it fits within max_w pixels."""
    x, y = org
    (tw, th), base = cv2.getTextSize(text, font, font_scale, thickness)
    if tw > max_w and tw > 0:
        sx = max_w / tw
        m = np.array([[sx, 0, x], [0, 1, y]], dtype=np.float64)
        cv2.putText(img, text, (0, 0), font, font_scale, color, thickness,
                    cv2.LINE_AA, m)
        return int(x + max_w), y
    cv2.putText(img, text, org, font, font_scale, color, thickness, cv2.LINE_AA)
    return x + tw, y


def _guilloche(h: int, w: int, rng: np.random.Generator, n: int = 8) -> np.ndarray:
    canvas = np.zeros((h, w), np.uint8)
    xs = np.arange(0, w, 2)
    for _ in range(n):
        a1, a2 = rng.uniform(3.0, 14.0, 2)
        f1, f2 = rng.uniform(0.008, 0.045, 2)
        p1, p2 = rng.uniform(0, 2 * np.pi, 2)
        y0 = rng.uniform(0, h)
        ys = y0 + a1 * np.sin(f1 * xs + p1) + a2 * np.sin(f2 * xs + p2)
        pts = np.stack([xs, ys], 1).astype(np.int32)
        cv2.polylines(canvas, [pts], False, 255, 1, cv2.LINE_AA)
    return canvas


def _background(rng: np.random.Generator) -> np.ndarray:
    c1 = rng.integers(170, 245, 3).astype(np.float64)
    c2 = rng.integers(150, 230, 3).astype(np.float64)
    t = np.linspace(0, 1, WIDTH)[None, :, None]
    img = (c1[None, None, :] * (1 - t) + c2[None, None, :] * t)
    img = np.repeat(img, HEIGHT, axis=0)
    # diagonal tint
    yy, xx = np.mgrid[0:HEIGHT, 0:WIDTH]
    tint = (xx / WIDTH + yy / HEIGHT) / 2
    img = img * (0.92 + 0.08 * tint[..., None])
    return np.clip(img, 0, 255).astype(np.uint8)


def _draw_avatar(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    """Cartoon portrait placeholder (no real person)."""
    canvas = np.zeros((h, w, 3), np.uint8)
    bg = rng.integers(200, 250, 3).astype(int)
    canvas[:] = bg
    skin = SKIN_TONES[int(rng.integers(0, len(SKIN_TONES)))]
    hair = HAIR_COLORS[int(rng.integers(0, len(HAIR_COLORS)))]
    shirt = rng.integers(40, 190, 3).astype(int)

    cx = w // 2
    head_w, head_h = int(w * rng.uniform(0.42, 0.52)), int(h * 0.30)
    head_cy = int(h * rng.uniform(0.34, 0.42))

    # shoulders
    cv2.ellipse(canvas, (cx, h + int(h * 0.25)),
                (int(w * 0.55), int(h * 0.45)), 0, 0, 360,
                tuple(int(v) for v in shirt), -1)
    # neck
    cv2.rectangle(canvas, (cx - int(w * 0.10), head_cy + head_h - 10),
                  (cx + int(w * 0.10), int(h * 0.75)),
                  tuple(int(v) for v in skin), -1)
    # hair back layer (long styles)
    style = int(rng.integers(0, 4))
    if style in (1, 2):
        cv2.ellipse(canvas, (cx, head_cy + int(head_h * 0.35)),
                    (int(head_w * 0.72), int(head_h * 1.05)), 0, 0, 360,
                    tuple(int(v) for v in hair), -1)
    # head
    cv2.ellipse(canvas, (cx, head_cy), (head_w // 2, head_h), 0, 0, 360,
                tuple(int(v) for v in skin), -1)
    # hair cap
    if style != 3:
        cv2.ellipse(canvas, (cx, head_cy - int(head_h * 0.35)),
                    (int(head_w * 0.53), int(head_h * 0.72)), 0, 180, 360,
                    tuple(int(v) for v in hair), -1)
    # eyes
    eye_dx = int(head_w * 0.20)
    eye_y = head_cy + int(head_h * 0.05)
    r = max(2, int(w * 0.028))
    for sx in (-1, 1):
        cv2.circle(canvas, (cx + sx * eye_dx, eye_y), r, (40, 45, 55), -1)
    # brows
    for sx in (-1, 1):
        cv2.line(canvas, (cx + sx * eye_dx - r * 2, eye_y - r * 3),
                 (cx + sx * eye_dx + r * 2, eye_y - r * 3),
                 tuple(int(v) for v in hair), 2, cv2.LINE_AA)
    # nose + mouth
    cv2.line(canvas, (cx, eye_y + r * 2), (cx - r, eye_y + r * 5),
             (60, 70, 90), 1, cv2.LINE_AA)
    cv2.ellipse(canvas, (cx, head_cy + int(head_h * 0.52)),
                (int(head_w * 0.20), int(head_h * 0.12)), 0, 0, 180,
                (50, 60, 110), 2, cv2.LINE_AA)
    # occasional glasses
    if rng.random() < 0.3:
        for sx in (-1, 1):
            x0 = cx + sx * eye_dx - r * 3
            cv2.rectangle(canvas, (x0, eye_y - r * 3),
                          (x0 + r * 6, eye_y + r * 3), (35, 40, 50), 2)
        cv2.line(canvas, (cx - r * 3 + 0, eye_y), (cx + r * 3, eye_y),
                 (35, 40, 50), 2)
    # soft vignette
    vig = np.linspace(1.0, 0.75, w)[None, :, None]
    canvas = np.clip(canvas.astype(np.float64) * vig, 0, 255).astype(np.uint8)
    return canvas


def _barcode(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    canvas = np.full((h, w, 3), 255, np.uint8)
    x = 4
    while x < w - 4:
        bw = int(rng.integers(1, 5))
        if rng.random() < 0.55:
            cv2.rectangle(canvas, (x, 3), (x + bw, h - 3), (15, 15, 20), -1)
        x += bw + int(rng.integers(1, 4))
    return canvas


def _signature(rng: np.random.Generator, w: int, h: int) -> np.ndarray:
    canvas = np.zeros((h, w, 3), np.uint8)
    x = int(rng.uniform(0.1, 0.3) * w)
    y = int(rng.uniform(0.5, 0.7) * h)
    pts = [(x, y)]
    for _ in range(int(rng.integers(10, 22))):
        x = int(np.clip(x + rng.uniform(-0.2, 0.3) * w, 2, w - 2))
        y = int(np.clip(y + rng.uniform(-0.25, 0.25) * h, 2, h - 2))
        pts.append((x, y))
    pts = np.array(pts, np.int32)
    color = tuple(int(v) for v in rng.integers(20, 80, 3))
    for i in range(len(pts) - 1):
        cv2.line(canvas, tuple(pts[i]), tuple(pts[i + 1]), color, 2, cv2.LINE_AA)
    return canvas


def _paste(dst: np.ndarray, src: np.ndarray, x: int, y: int) -> None:
    h, w = src.shape[:2]
    dst[y:y + h, x:x + w] = src


@dataclass
class MockIDData:
    """Rendered document + semantic region hints."""
    image: np.ndarray
    rois: dict = field(default_factory=dict)


def generate_mock_id(rng: np.random.Generator | None = None) -> MockIDData:
    if rng is None:
        rng = np.random.default_rng()
    img = _background(rng)

    # guilloche security pattern
    gu = _guilloche(HEIGHT, WIDTH, rng, n=int(rng.integers(5, 12)))
    ink = np.array(rng.integers(70, 160, 3), dtype=np.float64)
    mask = gu > 0
    img[mask] = (img[mask].astype(np.float64) * 0.72 + ink * 0.28).astype(np.uint8)

    state = STATES[int(rng.integers(0, len(STATES)))]
    org = ORGS[int(rng.integers(0, len(ORGS)))]
    code = DOC_CODES[int(rng.integers(0, len(DOC_CODES)))]
    surname, given = _rand_name(rng, 2), _rand_name(rng, 1)
    docnum = "".join(rng.choice(list(string.digits), size=8))
    yy = int(rng.integers(50, 99))
    mm, dd = int(rng.integers(1, 13)), int(rng.integers(1, 29))
    dob = f"{yy:02d}{mm:02d}{dd:02d}"
    ey = int(rng.integers(25, 40))
    exp = f"{ey:02d}{int(rng.integers(1, 13)):02d}{int(rng.integers(1, 29)):02d}"
    sex = SEXES[int(rng.integers(0, len(SEXES)))]
    height = f"{rng.integers(150, 195)}cm"
    eyes = rng.choice(["BRN", "BLU", "GRN", "HAZ"])

    rois: dict = {}
    layout = int(rng.integers(0, 4))
    accent = tuple(int(v) for v in rng.integers(30, 170, 3))
    dark = tuple(int(v * 0.55) for v in accent)
    _ = dark

    # ---- header ----
    header_h = int(rng.uniform(56, 76))
    cv2.rectangle(img, (0, 0), (WIDTH, header_h), accent, -1)
    if layout in (0, 2):
        _fitted_text(img, org, (18, header_h // 2 + 7), WIDTH - 160,
                     cv2.FONT_HERSHEY_DUPLEX, (255, 255, 255), 2, 0.75)
    else:
        _fitted_text(img, org, (90, header_h // 2 + 7), WIDTH - 190,
                     cv2.FONT_HERSHEY_DUPLEX, (255, 255, 255), 2, 0.7)
        cv2.circle(img, (46, header_h // 2), 26, (250, 250, 250), 2, cv2.LINE_AA)
        cv2.circle(img, (46, header_h // 2), 17, accent, -1)
        cv2.circle(img, (46, header_h // 2), 8, (250, 250, 250), -1)
    rois["header"] = (0, 0, WIDTH, header_h)

    # ---- portrait ----
    ph = int(rng.uniform(170, 205))
    pw = int(ph * rng.uniform(0.72, 0.84))
    py = header_h + int(rng.uniform(14, 30))
    px = 26 if layout in (0, 2) else WIDTH - pw - 26
    frame_color = tuple(int(v) for v in rng.integers(40, 120, 3))
    cv2.rectangle(img, (px - 5, py - 5), (px + pw + 5, py + ph + 5),
                  (250, 250, 250), -1)
    _paste(img, _draw_avatar(pw, ph, rng), px, py)
    cv2.rectangle(img, (px - 5, py - 5), (px + pw + 5, py + ph + 5),
                  frame_color, 2)
    rois["portrait"] = (px - 5, py - 5, pw + 10, ph + 10)

    # ghost portrait (secondary, faint, right side)
    if layout == 2:
        gw, gh = int(pw * 0.55), int(ph * 0.55)
        gx, gy = WIDTH - gw - 40, py + 30
        ghost = _draw_avatar(gw, gh, rng)
        region = img[gy:gy + gh, gx:gx + gw].astype(np.float64)
        img[gy:gy + gh, gx:gx + gw] = np.clip(
            region * 0.65 + ghost.astype(np.float64) * 0.35, 0, 255
        ).astype(np.uint8)
        rois["ghost"] = (gx, gy, gw, gh)

    # ---- data fields ----
    fx = px + pw + 30 if px < WIDTH // 2 else 30
    if layout == 2:
        fx = px + pw + 30
    fw = WIDTH - fx - (px + pw + 30 if px + pw + 30 < WIDTH else 30) - 200
    fw = max(200, min(fw, WIDTH - fx - 210))
    fields = [
        ("SURNAME / NAME", surname.upper()),
        ("GIVEN NAMES", given.upper()),
        ("DOCUMENT NO", docnum),
        ("DATE OF BIRTH", f"{'19' if yy > 24 else '20'}{dob[:2]}/{dob[2:4]}/{dob[4:]}"),
        ("DATE OF EXPIRY", f"20{exp[:2]}/{exp[2:4]}/{exp[4:]}"),
        ("SEX / HEIGHT / EYES", f"{sex}  {height}  {eyes}"),
        ("NATIONALITY", state),
        ("AUTHORITY", f"{state}-{docnum[:4]}"),
    ]
    fy0 = py + 4
    line_h = 26
    for i, (lab, val) in enumerate(fields):
        y = fy0 + i * line_h
        if y > HEIGHT - 120:
            break
        cv2.putText(img, lab, (fx, y), cv2.FONT_HERSHEY_PLAIN, 1.05,
                    (70, 80, 100), 1, cv2.LINE_AA)
        cv2.putText(img, val, (fx + 190, y), cv2.FONT_HERSHEY_DUPLEX, 0.72,
                    (25, 30, 45), 1, cv2.LINE_AA)
    rois["fields"] = (fx, max(0, fy0 - 20),
                      min(fw + 190, WIDTH - fx - 10),
                      len(fields) * line_h)

    # ---- barcode ----
    bw, bh = int(rng.uniform(150, 230)), 46
    bx = WIDTH - bw - 24 if layout in (0, 2) else 24
    by = py + ph - bh - 6
    bc = _barcode(rng, bw, bh)
    _paste(img, bc, bx, by)
    rois["barcode"] = (bx, by, bw, bh)

    # ---- signature ----
    sw, sh = 170, 44
    sx = 30 if layout in (0, 2) else WIDTH - sw - 30
    sy = HEIGHT - 108
    sig = _signature(rng, sw, sh)
    m = sig.sum(axis=2) > 0
    reg = img[sy:sy + sh, sx:sx + sw]
    reg[m] = sig[m]
    cv2.line(img, (sx, sy + sh - 4), (sx + sw, sy + sh - 4),
             (60, 70, 90), 1, cv2.LINE_AA)
    rois["signature"] = (sx, sy, sw, sh)

    # ---- hologram overlay ----
    hx = int(rng.uniform(0.55, 0.8) * WIDTH)
    hy = int(rng.uniform(0.3, 0.6) * HEIGHT)
    hroi = np.zeros_like(img)
    for k in range(9):
        cv2.ellipse(hroi, (hx, hy), (18 + k * 9, 14 + k * 7),
                    int(rng.integers(0, 180)), 0, 360,
                    (int(60 + 60 * k), 255 - k * 15, int(120 + 40 * k)), 2,
                    cv2.LINE_AA)
    ring_mask = hroi.sum(axis=2) > 0
    reg = img[hy - 80:hy + 80, hx - 100:hx + 100]
    rm = ring_mask[hy - 80:hy + 80, hx - 100:hx + 100]
    reg[rm] = np.clip(reg[rm].astype(np.float64) * 0.5
                      + hroi[hy - 80:hy + 80, hx - 100:hx + 100][rm] * 0.5,
                      0, 255).astype(np.uint8)
    rois["hologram"] = (hx - 100, hy - 80, 200, 160)

    # ---- MRZ ----
    mrz_lines = _mrz(rng, state, surname.upper(), given.upper(), docnum,
                     dob, exp, sex)
    mrz_h = 96
    mrz_y0 = HEIGHT - mrz_h - 8
    cv2.rectangle(img, (8, mrz_y0 - 6), (WIDTH - 8, HEIGHT - 6),
                  (245, 246, 240), -1)
    cv2.rectangle(img, (8, mrz_y0 - 6), (WIDTH - 8, HEIGHT - 6),
                  (150, 155, 160), 1)
    for i, line in enumerate(mrz_lines):
        y = mrz_y0 + 22 + i * 26
        _fitted_text(img, line, (22, y), WIDTH - 44,
                     cv2.FONT_HERSHEY_PLAIN, (20, 22, 30), 2, 0.95)
    rois["mrz"] = (8, mrz_y0 - 6, WIDTH - 16, mrz_h + 12)

    # ---- paper texture noise ----
    noise = rng.normal(0, 3.5, img.shape)
    img = np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    # sanity: keep rois inside bounds
    for k, (x, y, w, h) in list(rois.items()):
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(WIDTH, x + w), min(HEIGHT, y + h)
        rois[k] = (x0, y0, max(1, x1 - x0), max(1, y1 - y0))

    return MockIDData(image=img, rois=rois)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for i in range(4):
        d = generate_mock_id(rng)
        cv2.imwrite(f"mock_id_{i}.jpg", d.image, [cv2.IMWRITE_JPEG_QUALITY, 93])
    print("wrote mock_id_0..3.jpg")
