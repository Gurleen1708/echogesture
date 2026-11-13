"""Streaming decoder, per-stage timing, and the on-screen HUD used by the live demo and the demo reel."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from .features import clip_to_input
from .landmarks import stack_frames

# ---------------------------------------------------------------------------
# decoding
# ---------------------------------------------------------------------------


class StreamDecoder:
    """Sliding window over the last `window` frames -> words, smoothed so captions do not flicker.

    push() is called once per camera frame with that frame's raw landmarks and returns a word
    when one is confirmed. A word is confirmed when the top class has stayed the same for
    `stable` consecutive model runs with EMA confidence >= `threshold`. When no hand is seen
    for `idle_frames`, the window is cleared so the next sign starts fresh.
    """

    def __init__(self, classifier, aspect: float, window: int = 40, stride: int = 4,
                 threshold: float = 0.6, stable: int = 3, idle_frames: int = 12):
        self.clf, self.aspect = classifier, aspect
        self.buf: deque = deque(maxlen=window)
        self.stride, self.threshold, self.stable, self.idle_frames = stride, threshold, stable, idle_frames
        self.count = self.streak = self.idle = 0
        self.ema = None
        self.candidate = self.last_word = None
        self.top3: list[tuple[str, float]] = []
        self.model_ms = 0.0
        self.signing = False

    @property
    def fill(self) -> float:
        """How full the analysis window is (0..1); drawn as the thin progress line in the HUD."""
        return len(self.buf) / float(self.buf.maxlen)

    def reset(self):
        self.buf.clear()
        self.ema, self.candidate, self.last_word, self.streak = None, None, None, 0
        self.top3, self.signing = [], False

    def push(self, frame_raw: dict) -> str | None:
        self.buf.append(frame_raw)
        self.count += 1
        if frame_raw["hand_present"].max() > 0:
            self.idle = 0
        else:
            self.idle += 1
        if self.idle >= self.idle_frames:      # hands down -> sign finished
            self.reset()
            return None
        seen = np.mean([f["hand_present"].max() for f in self.buf])
        self.signing = len(self.buf) >= 12 and seen >= 0.5
        if not self.signing or self.count % self.stride:
            return None
        probs, self.model_ms = self.clf.classify(clip_to_input(stack_frames(list(self.buf), self.aspect)))
        self.ema = probs if self.ema is None else 0.6 * self.ema + 0.4 * probs
        order = np.argsort(-self.ema)[:3]
        self.top3 = [(self.clf.vocab[int(i)], float(self.ema[i])) for i in order]
        word, conf = self.top3[0]
        self.streak = self.streak + 1 if word == self.candidate else 1
        self.candidate = word
        if conf >= self.threshold and self.streak >= self.stable and word != self.last_word:
            self.last_word = word
            return word
        return None


class Stats:
    """Per-stage timing over the last `n` frames: landmark ms, model ms, end-to-end ms, FPS."""

    def __init__(self, n: int = 60):
        self.lm: deque = deque(maxlen=n)
        self.model: deque = deque(maxlen=n)
        self.total: deque = deque(maxlen=n)
        self.frames = 0

    def add(self, landmark_ms: float, model_ms: float, total_ms: float):
        self.lm.append(landmark_ms)
        self.model.append(model_ms)
        self.total.append(total_ms)
        self.frames += 1

    def _m(self, d):
        return float(np.mean(d)) if d else 0.0

    @property
    def landmark_ms(self):
        return self._m(self.lm)

    @property
    def model_ms(self):
        return self._m(self.model)

    @property
    def end_to_end_ms(self):
        return self._m(self.total)

    @property
    def fps(self):
        return 1000.0 / self.end_to_end_ms if self.end_to_end_ms > 0 else 0.0

    def summary(self) -> dict:
        return {"fps": round(self.fps, 1), "landmark_ms": round(self.landmark_ms, 1),
                "model_ms": round(self.model_ms, 2), "end_to_end_ms": round(self.end_to_end_ms, 1),
                "frames": self.frames}


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]

# BGR palette
INK = (14, 14, 18)
PANEL = (22, 22, 28)
WHITE = (245, 245, 245)
GREY = (160, 160, 170)
MINT = (120, 230, 150)
AMBER = (70, 190, 255)
BLUE = (255, 190, 90)
RED = (80, 80, 240)
FONT = cv2.FONT_HERSHEY_DUPLEX


def _rrect(img, x0, y0, x1, y1, color, r=10, alpha=1.0):
    """Filled rounded rectangle; alpha < 1 blends it over the frame."""
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    r = max(1, min(r, (x1 - x0) // 2, (y1 - y0) // 2))
    layer = img if alpha >= 1.0 else img.copy()
    cv2.rectangle(layer, (x0 + r, y0), (x1 - r, y1), color, -1)
    cv2.rectangle(layer, (x0, y0 + r), (x1, y1 - r), color, -1)
    for cx, cy in ((x0 + r, y0 + r), (x1 - r, y0 + r), (x0 + r, y1 - r), (x1 - r, y1 - r)):
        cv2.circle(layer, (cx, cy), r, color, -1, cv2.LINE_AA)
    if alpha < 1.0:
        cv2.addWeighted(layer, alpha, img, 1 - alpha, 0, img)


def _text(img, s, x, y, scale, color=WHITE, thick=1):
    cv2.putText(img, s, (int(x), int(y)), FONT, scale, color, thick, cv2.LINE_AA)


def _tw(s, scale, thick=1):
    return cv2.getTextSize(s, FONT, scale, thick)[0][0]


def _pill(img, s, x, y, scale, fg, bg, pad=10):
    w = _tw(s, scale) + 2 * pad
    h = int(26 * scale / 0.55)
    _rrect(img, x, y - h * 0.72, x + w, y + h * 0.28, bg, r=h // 2)
    _text(img, s, x + pad, y, scale, fg)
    return w


def draw_hands(frame, raw, s):
    h, w = frame.shape[:2]
    for slot, colour, label in ((0, BLUE, "L"), (1, MINT, "R")):
        if raw["hand_present"][slot] <= 0:
            continue
        pts = (raw["hands_img"][slot, :, :2] * [w, h]).astype(int)
        for a, b in HAND_EDGES:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), colour, max(1, int(2 * s)), cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), max(2, int(3 * s)), WHITE, -1, cv2.LINE_AA)
        wx, wy = pts[0]
        _rrect(frame, wx - 14 * s, wy + 8 * s, wx + 14 * s, wy + 30 * s, colour, r=6)
        _text(frame, label, wx - 6 * s, wy + 25 * s, 0.5 * s, INK)


def draw_hud(frame: np.ndarray, raw: dict, decoder: StreamDecoder, words: list[str], stats: Stats,
             model_label: str = "", expected: str | None = None, paused: bool = False,
             title: str = "EchoGesture  ·  Team Mudra") -> np.ndarray:
    """Draw the full overlay in place and return the frame."""
    h, w = frame.shape[:2]
    s = max(0.5, h / 720.0)
    draw_hands(frame, raw, s)

    # ---- top bar -----------------------------------------------------------
    top = int(54 * s)
    _rrect(frame, 0, 0, w, top, INK, r=0, alpha=0.78)
    if paused:
        dot, state = GREY, "PAUSED"
    elif decoder.signing:
        dot, state = MINT, "SIGNING"
    else:
        dot, state = GREY, "WAITING FOR HANDS"
    cv2.circle(frame, (int(22 * s), int(27 * s)), int(7 * s), dot, -1, cv2.LINE_AA)
    _text(frame, state, 38 * s, 34 * s, 0.6 * s, WHITE)
    tw = _tw(title, 0.6 * s)
    _text(frame, title, (w - tw) / 2, 34 * s, 0.6 * s, GREY)
    pw = _tw("ON-DEVICE  ·  NO CLOUD", 0.5 * s) + 20 * s
    _pill(frame, "ON-DEVICE  ·  NO CLOUD", w - pw - 14 * s, 33 * s, 0.5 * s, INK, MINT)

    # ---- left: live candidates ---------------------------------------------
    x0, y0, pw_ = int(14 * s), int(top + 14 * s), int(290 * s)
    if decoder.top3:
        _rrect(frame, x0, y0, x0 + pw_, y0 + int(124 * s), PANEL, r=int(12 * s), alpha=0.75)
        _text(frame, "LIVE PREDICTION", x0 + 14 * s, y0 + 24 * s, 0.42 * s, GREY)
        for i, (word, conf) in enumerate(decoder.top3):
            y = y0 + int((52 + 32 * i) * s)
            col = MINT if i == 0 else GREY
            _text(frame, word, x0 + 14 * s, y, (0.62 if i == 0 else 0.5) * s, WHITE if i == 0 else GREY)
            bx0, bx1 = x0 + int(150 * s), x0 + pw_ - int(50 * s)
            _rrect(frame, bx0, y - 9 * s, bx1, y - 2 * s, (50, 50, 60), r=3)
            _rrect(frame, bx0, y - 9 * s, bx0 + (bx1 - bx0) * max(0.02, conf), y - 2 * s, col, r=3)
            _text(frame, f"{conf * 100:3.0f}%", x0 + pw_ - 44 * s, y, 0.45 * s, col)
    if expected:
        ey = y0 + int((138 if decoder.top3 else 0) * s)
        _pill(frame, f"EXPECTED  {expected}", x0, ey + 24 * s, 0.5 * s, INK, AMBER)

    # ---- right: on-device stats --------------------------------------------
    rw, rh = int(250 * s), int(150 * s)
    rx0, ry0 = w - rw - int(14 * s), int(top + 14 * s)
    _rrect(frame, rx0, ry0, rx0 + rw, ry0 + rh, PANEL, r=int(12 * s), alpha=0.75)
    _text(frame, "ON-DEVICE PIPELINE", rx0 + 14 * s, ry0 + 24 * s, 0.42 * s, GREY)
    rows = [("fps", f"{stats.fps:5.1f}"), ("landmarks", f"{stats.landmark_ms:5.1f} ms"),
            ("classifier", f"{stats.model_ms:5.2f} ms"), ("end-to-end", f"{stats.end_to_end_ms:5.1f} ms")]
    for i, (k, v) in enumerate(rows):
        y = ry0 + int((48 + 24 * i) * s)
        _text(frame, k, rx0 + 14 * s, y, 0.45 * s, GREY)
        _text(frame, v, rx0 + rw - _tw(v, 0.45 * s) - 14 * s, y, 0.45 * s, MINT if k == "fps" else WHITE)
    if model_label:
        _text(frame, model_label, rx0 + 14 * s, ry0 + rh - 10 * s, 0.38 * s, GREY)
    # sparkline of end-to-end ms (last 60 frames)
    if len(stats.total) > 2:
        sx0, sx1, sy0, sy1 = rx0 + 14 * s, rx0 + rw - 14 * s, ry0 + rh + 8 * s, ry0 + rh + 30 * s
        _rrect(frame, sx0 - 6 * s, sy0 - 4 * s, sx1 + 6 * s, sy1 + 4 * s, PANEL, r=int(6 * s), alpha=0.6)
        vals = np.array(stats.total, np.float32)
        vmax = max(vals.max(), 1.0)
        xs = np.linspace(sx0, sx1, len(vals))
        ys = sy1 - (vals / vmax) * (sy1 - sy0)
        pts = np.stack([xs, ys], axis=1).astype(np.int32)
        cv2.polylines(frame, [pts], False, MINT, 1, cv2.LINE_AA)

    # ---- bottom: caption bar -----------------------------------------------
    bar = int(96 * s)
    _rrect(frame, 0, h - bar, w, h, INK, r=0, alpha=0.82)
    # window-fill progress line
    cv2.line(frame, (0, h - bar), (int(w * decoder.fill), h - bar), MINT if decoder.signing else GREY,
             max(2, int(3 * s)))
    if words:
        scale, gap, pad = 0.9 * s, int(12 * s), int(14 * s)
        shown = words[-8:]
        widths = [_tw(x, scale, 2) + 2 * pad for x in shown]
        while sum(widths) + gap * (len(shown) - 1) > w - 40 * s and len(shown) > 1:
            shown, widths = shown[1:], widths[1:]
        x = (w - (sum(widths) + gap * (len(shown) - 1))) // 2
        cy = h - bar // 2
        for i, (word, ww) in enumerate(zip(shown, widths)):
            last = i == len(shown) - 1
            _rrect(frame, x, cy - 26 * s, x + ww, cy + 18 * s, MINT if last else (48, 48, 58), r=int(10 * s))
            _text(frame, word, x + pad, cy + 8 * s, scale, INK if last else WHITE, 2)
            x += ww + gap
    else:
        hint = "sign a word to start the caption"
        _text(frame, hint, (w - _tw(hint, 0.6 * s)) / 2, h - bar // 2 + 8 * s, 0.6 * s, GREY)
    keys = "Q quit   C clear   S screenshot   SPACE pause"
    _text(frame, keys, w - _tw(keys, 0.38 * s) - 14 * s, h - 10 * s, 0.38 * s, (110, 110, 120))
    return frame


def title_card(w: int, h: int, title: str, subtitle: str, lines: list[str]) -> np.ndarray:
    """Dark slide used by the demo reel for the intro and the summary."""
    img = np.full((h, w, 3), INK, np.uint8)
    s = max(0.5, h / 720.0)
    _text(img, title, (w - _tw(title, 1.5 * s, 2)) / 2, h * 0.36, 1.5 * s, WHITE, 2)
    _text(img, subtitle, (w - _tw(subtitle, 0.7 * s)) / 2, h * 0.36 + 48 * s, 0.7 * s, MINT)
    for i, line in enumerate(lines):
        _text(img, line, (w - _tw(line, 0.6 * s)) / 2, h * 0.36 + (100 + 34 * i) * s, 0.6 * s, GREY)
    return img
