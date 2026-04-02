import cv2
import numpy as np
from typing import Sequence
from collections import deque


# ── Zone colour map (BGR, semi-transparent) ───────────────────────────────────
_ZONE_COLORS = {
    "yorker": (50,  50, 200),
    "full":   (50, 160, 200),
    "good":   (50, 200,  80),
    "short":  (200, 80,  50),
}

_HUD_BG      = (20, 20, 20)
_HUD_TEXT    = (230, 230, 230)
_BOUNCE_DOT  = (30, 30, 220)
_TRAJ_COLOR  = (0, 220, 255)
_BALL_RING   = {
    "high":   (50, 220, 50),
    "mid":    (30, 160, 255),
    "low":    (50,  50, 200),
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


# ── Already in your file ──────────────────────────────────────────────────────

def draw_pitch_zones(frame, zones, alpha=0.20):
    overlay = frame.copy()
    h, w = frame.shape[:2]
    for name, (y0, y1) in zones.items():
        color = _ZONE_COLORS.get(name.lower(), (128, 128, 128))
        cv2.rectangle(overlay, (0, y0), (w, y1), color, -1)
        ly = y0 + (y1 - y0) // 2 + 5
        cv2.putText(overlay, name.upper(), (6, ly), _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    return frame


def draw_trajectory(frame, trajectory, max_points=40, thickness=2):
    if len(trajectory) < 2:
        return frame
    pts = [(int(p[0]), int(p[1])) for p in trajectory[-max_points:]]
    n = len(pts)
    overlay = frame.copy()
    for i in range(1, n):
        alpha_i = i / n
        color = tuple(int(c * alpha_i) for c in _TRAJ_COLOR)
        cv2.line(overlay, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    if n >= 2:
        cv2.line(frame, pts[-2], pts[-1], _TRAJ_COLOR, thickness, cv2.LINE_AA)
    return frame


def draw_ball(frame, x, y, radius, confidence=1.0, label=""):
    cx, cy, r = int(x), int(y), max(1, int(radius))
    ring_color = _BALL_RING["high"] if confidence >= 0.65 else \
                 _BALL_RING["mid"]  if confidence >= 0.35 else _BALL_RING["low"]
    cv2.circle(frame, (cx, cy), r + 3, ring_color, 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), r, (255, 255, 255), 1, cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (cx + r + 5, cy), _FONT, 0.45, ring_color, 1, cv2.LINE_AA)
    return frame


def draw_bounce_marker(frame, pt_or_x, label_or_y=None, *args, **kwargs):
    """
    Flexible signature — processor.py calls it as:
        draw_bounce_marker(vis, (x, y), "Ball 1")
    The new API also supports:
        draw_bounce_marker(frame, x, y, frame_idx, bounce_frame)
    """
    # Old processor.py call: draw_bounce_marker(vis, (x,y), label_str)
    if isinstance(pt_or_x, (tuple, list)):
        x, y = pt_or_x
        cx, cy = int(x), int(y)
        cv2.circle(frame, (cx, cy), 6, _BOUNCE_DOT, -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, _BOUNCE_DOT, 1, cv2.LINE_AA)
        label = label_or_y or ""
        if label:
            cv2.putText(frame, str(label), (cx + 10, cy - 8),
                        _FONT, 0.45, _BOUNCE_DOT, 1, cv2.LINE_AA)
        return frame

    # New API: draw_bounce_marker(frame, x, y, frame_idx, bounce_frame, pulse_frames=30)
    x = pt_or_x
    y = label_or_y
    frame_idx  = args[0] if len(args) > 0 else kwargs.get("frame_idx", 0)
    bounce_frm = args[1] if len(args) > 1 else kwargs.get("bounce_frame", 0)
    pulse      = args[2] if len(args) > 2 else kwargs.get("pulse_frames", 30)

    age = frame_idx - bounce_frm
    if age < 0 or age > pulse:
        return frame
    alpha = 1.0 - age / pulse
    cx, cy = int(x), int(y)
    color = tuple(int(c * alpha) for c in _BOUNCE_DOT)
    cv2.circle(frame, (cx, cy), 6, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 8 + age,
               tuple(int(c * alpha * 0.7) for c in _BOUNCE_DOT), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, completed_or_frame_idx, ball_no_or_bounce=None,
             bounce_frame=None, length=None, confidence=None, pos="top-left"):
    """
    Flexible HUD — supports both call signatures:

    Old (processor.py):
        draw_hud(vis, completed_list, current_ball_no)

    New API:
        draw_hud(frame, frame_idx, bounce_detected, bounce_frame, length, confidence)
    """
    h, w = frame.shape[:2]

    # ── Detect which call style ───────────────────────────────────
    if isinstance(completed_or_frame_idx, list):
        # Old style: draw_hud(vis, completed, ball_no)
        completed = completed_or_frame_idx
        ball_no   = ball_no_or_bounce or 0

        bounced_list = [r for r in completed if r.bounced]
        last = completed[-1] if completed else None

        lines = [
            f"Ball No  : {ball_no}",
            f"Pitched  : {len(bounced_list)}",
            f"Length   : {last.length.upper() if last and last.length else '—'}",
            f"Bounce   : {'YES' if last and last.bounced else 'NO'}",
        ]
    else:
        # New style
        frame_idx       = completed_or_frame_idx
        bounce_detected = bool(ball_no_or_bounce)
        lines = [
            f"Frame    : {frame_idx}",
            f"Bounce   : {'YES (fr ' + str(bounce_frame) + ')' if bounce_detected else 'NO'}",
            f"Length   : {length.upper() if length else '—'}",
            f"Conf     : {confidence:.2f}" if confidence is not None else "Conf     : —",
        ]

    pad, line_h, box_w = 8, 22, 220
    box_h = pad * 2 + line_h * len(lines)

    if pos == "top-right":
        x0, y0 = w - box_w - 10, 10
    elif pos == "bottom-left":
        x0, y0 = 10, h - box_h - 10
    elif pos == "bottom-right":
        x0, y0 = w - box_w - 10, h - box_h - 10
    else:
        x0, y0 = 10, 10

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + box_w, y0 + box_h), _HUD_BG, -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
    cv2.rectangle(frame, (x0, y0), (x0 + box_w, y0 + box_h), (80, 80, 80), 1)

    for i, line in enumerate(lines):
        ty = y0 + pad + (i + 1) * line_h - 4
        color = _HUD_TEXT
        if "YES" in line:
            color = _BOUNCE_DOT
        cv2.putText(frame, line, (x0 + pad, ty), _FONT, 0.45, color, 1, cv2.LINE_AA)

    return frame


# ── NEW functions that processor.py needs ────────────────────────────────────

def draw_ball_trail(frame: np.ndarray,
                    trail,
                    max_points: int = 50,
                    thickness: int = 2) -> np.ndarray:
    """
    Draw the ball trail from a deque/list of (x, y) tuples.
    Older points fade out.

    Called in processor.py as:  draw_ball_trail(vis, trail)
    """
    pts = list(trail)[-max_points:]
    if len(pts) < 2:
        return frame

    n = len(pts)
    for i in range(1, n):
        alpha_i = i / n
        color = tuple(int(c * alpha_i) for c in _TRAJ_COLOR)
        p1 = (int(pts[i - 1][0]), int(pts[i - 1][1]))
        p2 = (int(pts[i][0]),     int(pts[i][1]))
        cv2.line(frame, p1, p2, color, thickness, cv2.LINE_AA)

    return frame


def draw_length_banner(frame: np.ndarray,
                       result,
                       frame_no: int,
                       banner_until: dict,
                       display_frames: int = 60) -> np.ndarray:
    """
    Show a large centred length banner for a short time after a delivery.

    Called in processor.py as:
        draw_length_banner(vis, res, frame_no, banner_until)

    `result`       – a DeliveryResult with .ball_no, .length, .bounced
    `banner_until` – dict[ball_no → last_frame] maintained by processor.py
    """
    ball_no = result.ball_no

    # Register the banner end-frame the first time we see this delivery
    if ball_no not in banner_until:
        banner_until[ball_no] = frame_no + display_frames

    if frame_no > banner_until[ball_no]:
        return frame

    h, w = frame.shape[:2]
    length = result.length or "—"
    b_text = "BOUNCE" if result.bounced else "FULL TOSS"

    bg_color = _ZONE_COLORS.get(length.lower(), (60, 60, 60))

    # Banner background strip
    by0, by1 = h // 2 - 30, h // 2 + 30
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, by0), (w, by1), bg_color, -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    # Length text (large, centred)
    text = f"{length.upper()}  |  {b_text}"
    (tw, th), _ = cv2.getTextSize(text, _FONT, 1.1, 2)
    tx = (w - tw) // 2
    ty = h // 2 + th // 2
    cv2.putText(frame, text, (tx, ty), _FONT, 1.1, (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def draw_pitch_zones_side(frame: np.ndarray,
                          alpha: float = 0.15) -> np.ndarray:
    """
    Draw standard cricket pitch length zones for a side-view camera.

    Zones are expressed as fractions of frame height so they adapt to
    any resolution automatically.

    Called in processor.py as:  draw_pitch_zones_side(vis)
    """
    h, w = frame.shape[:2]

    # Approximate zone boundaries (fraction of frame height, top=0)
    zone_fractions = {
        "yorker": (0.75, 0.88),
        "full":   (0.62, 0.75),
        "good":   (0.48, 0.62),
        "short":  (0.30, 0.48),
    }

    zones_px = {name: (int(f0 * h), int(f1 * h))
                for name, (f0, f1) in zone_fractions.items()}

    return draw_pitch_zones(frame, zones_px, alpha=alpha)