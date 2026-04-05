import cv2
import numpy as np
from collections import deque
from src.config import CFG, LENGTH_COLORS
from src.models.delivery_result import DeliveryResult


# ── Trail history per ball (persists across frames) ──────────────
_ball_trails: dict[int, list] = {}


def add_to_trail(ball_no: int, pt: tuple, maxlen: int = 80):
    if ball_no not in _ball_trails:
        _ball_trails[ball_no] = []
    _ball_trails[ball_no].append(pt)
    if len(_ball_trails[ball_no]) > maxlen:
        _ball_trails[ball_no].pop(0)


def clear_trail(ball_no: int):
    _ball_trails[ball_no] = []


def reset_all_trails():
    _ball_trails.clear()


# ── HUD ──────────────────────────────────────────────────────────
def draw_hud(frame, results_so_far, current_ball_no, total_balls=6):
    h, w = frame.shape[:2]
    rows        = max(len(results_so_far), 1)
    box_w       = 280
    box_h       = 48 + 30 * rows + 12
    pad         = 10

    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h),
                  (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Border line
    cv2.rectangle(frame, (pad, pad), (pad + box_w, pad + box_h),
                  (80, 80, 80), 1, cv2.LINE_AA)

    # Title
    cv2.putText(frame, "DELIVERY  ANALYSIS", (pad + 12, pad + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (230, 230, 230), 1, cv2.LINE_AA)

    # Divider
    cv2.line(frame, (pad + 8, pad + 30), (pad + box_w - 8, pad + 30),
             (70, 70, 70), 1, cv2.LINE_AA)

    for i, res in enumerate(results_so_far):
        y = pad + 52 + i * 30

        # Row highlight for latest ball
        if res.ball_no == current_ball_no - 1:
            hl = frame.copy()
            cv2.rectangle(hl, (pad + 4, y - 14), (pad + box_w - 4, y + 8),
                          (40, 40, 40), -1)
            cv2.addWeighted(hl, 0.5, frame, 0.5, 0, frame)

        b_label = "BOUNCED" if res.bounced else "FULL TOSS"
        b_color = CFG["color_bounce"] if res.bounced else CFG["color_no_bounce"]
        l_color = LENGTH_COLORS.get(res.length, (200, 200, 200))

        # Ball number
        cv2.putText(frame, f"B{res.ball_no}", (pad + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (180, 180, 180), 1, cv2.LINE_AA)

        # Bounce status dot + label
        dot_x = pad + 52
        cv2.circle(frame, (dot_x, y - 4), 5, b_color, -1, cv2.LINE_AA)
        cv2.putText(frame, b_label, (dot_x + 10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, b_color, 1, cv2.LINE_AA)

        # Length tag with background pill
        l_text  = res.length
        (tw, th), _ = cv2.getTextSize(l_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        pill_x  = pad + box_w - tw - 22
        cv2.rectangle(frame, (pill_x - 4, y - th - 2), (pill_x + tw + 4, y + 3),
                      l_color, -1, cv2.LINE_AA)
        cv2.putText(frame, l_text, (pill_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (10, 10, 10), 1, cv2.LINE_AA)


# ── Ball trail ───────────────────────────────────────────────────
def draw_ball_trail(frame, trail, ball_no: int = 0):
    """
    Draw a fading, thick trail with a glow effect.
    Accepts either the live deque (from processor) or the per-ball trail dict.
    """
    pts = list(trail)
    if len(pts) < 2:
        return

    n = len(pts)
    for i in range(1, n):
        alpha   = i / n                          # 0=old → 1=newest
        base_c  = CFG["color_trail"]

        # Colour fades from dim to full
        color = (
            int(base_c[0] * alpha),
            int(base_c[1] * alpha),
            int(base_c[2] * alpha),
        )

        # Thickness grows toward the newest point
        thickness = max(1, int(1 + 3 * alpha))

        cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

    # Glow on the last few points
    glow_pts = pts[max(0, n - 8):]
    for i in range(1, len(glow_pts)):
        cv2.line(frame, glow_pts[i - 1], glow_pts[i],
                 (255, 255, 255), 1, cv2.LINE_AA)

    # Bright dot at the very tip
    if pts:
        cv2.circle(frame, pts[-1], 4, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, pts[-1], 6, CFG["color_ball"], 1, cv2.LINE_AA)


# ── Bounce marker ────────────────────────────────────────────────
def draw_bounce_marker(frame, pt, label=""):
    x, y = pt

    # Outer pulse ring
    cv2.circle(frame, (x, y), 18, CFG["color_bounce"],  1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 12, CFG["color_bounce"],  2, cv2.LINE_AA)

    # Inner filled dot
    cv2.circle(frame, (x, y),  4, CFG["color_bounce"], -1, cv2.LINE_AA)

    # Cross-hair lines
    cv2.line(frame, (x - 20, y), (x - 13, y), CFG["color_bounce"], 1, cv2.LINE_AA)
    cv2.line(frame, (x + 13, y), (x + 20, y), CFG["color_bounce"], 1, cv2.LINE_AA)
    cv2.line(frame, (x, y - 20), (x, y - 13), CFG["color_bounce"], 1, cv2.LINE_AA)
    cv2.line(frame, (x, y + 13), (x, y + 20), CFG["color_bounce"], 1, cv2.LINE_AA)

    # Label with background
    if label:
        txt = "BOUNCE"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        lx, ly = x + 22, y - 6
        cv2.rectangle(frame, (lx - 3, ly - th - 2), (lx + tw + 3, ly + 3),
                      (10, 10, 10), -1)
        cv2.putText(frame, txt, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, CFG["color_bounce"], 1, cv2.LINE_AA)


# ── Bottom banner ────────────────────────────────────────────────
def draw_length_banner(frame, result: DeliveryResult, frame_no: int,
                       banner_until: dict):
    if result is None:
        return
    if frame_no > banner_until.get(result.ball_no, 0):
        return

    h, w = frame.shape[:2]
    color  = LENGTH_COLORS.get(result.length, (200, 200, 200))
    b_str  = "BOUNCED" if result.bounced else "FULL TOSS"
    label  = f"Ball {result.ball_no}   {b_str}   {result.length.upper()}"

    # Banner background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 60), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Coloured accent bar at top of banner
    cv2.rectangle(frame, (0, h - 60), (w, h - 56), color, -1)

    # Centred text
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.72, 2)
    tx = max(8, (w - tw) // 2)
    cv2.putText(frame, label, (tx, h - 22),
                cv2.FONT_HERSHEY_DUPLEX, 0.72, color, 2, cv2.LINE_AA)


# ── Pitch zone overlay (side view) ──────────────────────────────
def draw_pitch_zones_side(frame):
    h, w = frame.shape[:2]

    overlay = frame.copy()

    for name, (lo, hi) in CFG["length_zones_side"].items():
        x1    = int(lo * w)
        x2    = int(hi * w)
        color = LENGTH_COLORS.get(name, (200, 200, 200))

        # Subtle full-height shaded column
        cv2.rectangle(overlay, (x1, int(h * 0.30)), (x2, int(h * 0.82)),
                      color, -1)

    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

    # Zone boundary lines + labels at top
    for name, (lo, hi) in CFG["length_zones_side"].items():
        x1    = int(lo * w)
        x2    = int(hi * w)
        mid_x = (x1 + x2) // 2
        color = LENGTH_COLORS.get(name, (200, 200, 200))

        # Boundary line
        cv2.line(frame, (x1, int(h * 0.28)), (x1, int(h * 0.84)),
                 (*color, 120), 1, cv2.LINE_AA)

        # Zone name label
        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        lx = mid_x - tw // 2
        ly = int(h * 0.27)

        # Label background
        cv2.rectangle(frame, (lx - 3, ly - th - 2), (lx + tw + 3, ly + 2),
                      (15, 15, 15), -1)
        cv2.putText(frame, name, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)