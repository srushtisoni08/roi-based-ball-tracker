import cv2
import numpy as np
from collections import deque
from src.config import CFG, LENGTH_COLORS
from src.models.delivery_result import DeliveryResult


# ── Ball trail ───────────────────────────────────────────────────
def draw_ball_trail(frame, trail, ball_no: int = 0):
    """
    Draw a fading, anti-aliased trail with a glow tip.
    Accepts a deque or list of (x, y) tuples.
    """
    pts = list(trail)
    if len(pts) < 2:
        return

    n = len(pts)
    for i in range(1, n):
        alpha = i / n                       # 0 = oldest, 1 = newest
        base  = CFG["color_trail"]
        color = (
            int(base[0] * alpha),
            int(base[1] * alpha),
            int(base[2] * alpha),
        )
        thickness = max(1, int(1 + 3 * alpha))
        cv2.line(frame, pts[i - 1], pts[i], color, thickness, cv2.LINE_AA)

    # Glow on newest 8 points
    glow = pts[max(0, n - 8):]
    for i in range(1, len(glow)):
        cv2.line(frame, glow[i - 1], glow[i], (255, 255, 255), 1, cv2.LINE_AA)

    # Bright dot at tip
    tip = pts[-1]
    cv2.circle(frame, tip, 4, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, tip, 6, CFG["color_ball"],  1, cv2.LINE_AA)


# ── Bounce marker ────────────────────────────────────────────────
def draw_bounce_marker(frame, pt, label: str = ""):
    """
    Prominent crosshair + rings at the bounce location.
    Stays visible on screen for as long as the caller renders it.
    """
    x, y = int(pt[0]), int(pt[1])

    # Outer pulse rings
    cv2.circle(frame, (x, y), 20, CFG["color_bounce"],  1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 13, CFG["color_bounce"],  2, cv2.LINE_AA)
    # Inner filled dot
    cv2.circle(frame, (x, y),  4, CFG["color_bounce"], -1, cv2.LINE_AA)

    # Crosshair lines
    for dx, dy in [(-22, 0), (13, 0), (0, -22), (0, 13)]:
        ex = x + (dx if dx > 0 else dx + 9) if dx != 0 else x
        ey = y + (dy if dy > 0 else dy + 9) if dy != 0 else y
    # Simpler explicit crosshair
    cv2.line(frame, (x - 22, y), (x - 13, y), CFG["color_bounce"], 1, cv2.LINE_AA)
    cv2.line(frame, (x + 13, y), (x + 22, y), CFG["color_bounce"], 1, cv2.LINE_AA)
    cv2.line(frame, (x, y - 22), (x, y - 13), CFG["color_bounce"], 1, cv2.LINE_AA)
    cv2.line(frame, (x, y + 13), (x, y + 22), CFG["color_bounce"], 1, cv2.LINE_AA)

    # "BOUNCE" label with dark background pill
    if label:
        txt = "BOUNCE"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        lx, ly = x + 24, y - 4
        cv2.rectangle(frame, (lx - 3, ly - th - 3), (lx + tw + 3, ly + 4),
                      (10, 10, 10), -1)
        cv2.putText(frame, txt, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, CFG["color_bounce"], 1, cv2.LINE_AA)


# ── HUD (top-left scoreboard) ────────────────────────────────────
def draw_hud(frame, results_so_far, current_ball_no, total_balls: int = 6):
    h, w = frame.shape[:2]
    rows  = max(len(results_so_far), 1)
    box_w = 290
    box_h = 50 + 32 * rows + 14
    pad   = 10

    # Semi-transparent dark background
    overlay = frame.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + box_w, pad + box_h),
                  (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Border
    cv2.rectangle(frame, (pad, pad), (pad + box_w, pad + box_h),
                  (75, 75, 75), 1, cv2.LINE_AA)

    # Title row
    cv2.putText(frame, "DELIVERY  ANALYSIS",
                (pad + 14, pad + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.54, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.line(frame, (pad + 8, pad + 32), (pad + box_w - 8, pad + 32),
             (65, 65, 65), 1, cv2.LINE_AA)

    for i, res in enumerate(results_so_far):
        y = pad + 54 + i * 32

        # Highlight most-recent row
        if res.ball_no == current_ball_no - 1:
            hl = frame.copy()
            cv2.rectangle(hl, (pad + 4, y - 16), (pad + box_w - 4, y + 8),
                          (40, 40, 40), -1)
            cv2.addWeighted(hl, 0.5, frame, 0.5, 0, frame)

        b_label = "BOUNCED"   if res.bounced else "FULL TOSS"
        b_color = CFG["color_bounce"] if res.bounced else CFG["color_no_bounce"]
        l_color = LENGTH_COLORS.get(res.length, (200, 200, 200))

        # Ball number
        cv2.putText(frame, f"B{res.ball_no}", (pad + 14, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (170, 170, 170), 1, cv2.LINE_AA)

        # Status dot + label
        dot_x = pad + 56
        cv2.circle(frame, (dot_x, y - 5), 5, b_color, -1, cv2.LINE_AA)
        cv2.putText(frame, b_label, (dot_x + 12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, b_color, 1, cv2.LINE_AA)

        # Length pill (right-aligned)
        l_text = res.length if res.length else "—"
        (tw, th), _ = cv2.getTextSize(l_text, cv2.FONT_HERSHEY_SIMPLEX, 0.46, 1)
        pill_x = pad + box_w - tw - 24
        cv2.rectangle(frame, (pill_x - 5, y - th - 3), (pill_x + tw + 5, y + 4),
                      l_color, -1, cv2.LINE_AA)
        cv2.putText(frame, l_text, (pill_x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, (10, 10, 10), 1, cv2.LINE_AA)


# ── Bottom length banner ─────────────────────────────────────────
def draw_length_banner(frame, result: DeliveryResult, frame_no: int,
                       banner_until: dict):
    """
    Draws a full-width banner at the bottom of the frame for BANNER_DURATION_FRAMES
    after each delivery ends.  banner_until[ball_no] = last frame to show it.
    """
    if result is None:
        return
    if frame_no > banner_until.get(result.ball_no, 0):
        return

    h, w  = frame.shape[:2]
    color = LENGTH_COLORS.get(result.length, (200, 200, 200))
    b_str = "BOUNCED" if result.bounced else "FULL TOSS"
    label = f"Ball {result.ball_no}   {b_str}   {result.length.upper()}"

    # Dark semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 64), (w, h), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    # Coloured accent bar at top of banner
    cv2.rectangle(frame, (0, h - 64), (w, h - 59), color, -1)

    # Centred text
    (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.76, 2)
    tx = max(10, (w - tw) // 2)
    cv2.putText(frame, label, (tx, h - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.76, color, 2, cv2.LINE_AA)


# ── Pitch zone overlay (side view) ──────────────────────────────
def draw_pitch_zones_side(frame):
    h, w = frame.shape[:2]
    overlay = frame.copy()

    for name, (lo, hi) in CFG["length_zones_side"].items():
        x1    = int(lo * w)
        x2    = int(hi * w)
        color = LENGTH_COLORS.get(name, (200, 200, 200))
        cv2.rectangle(overlay,
                      (x1, int(h * 0.28)), (x2, int(h * 0.84)),
                      color, -1)

    cv2.addWeighted(overlay, 0.07, frame, 0.93, 0, frame)

    # Zone boundary lines + name labels
    for name, (lo, hi) in CFG["length_zones_side"].items():
        x1    = int(lo * w)
        x2    = int(hi * w)
        mid_x = (x1 + x2) // 2
        color = LENGTH_COLORS.get(name, (200, 200, 200))

        cv2.line(frame,
                 (x1, int(h * 0.26)), (x1, int(h * 0.86)),
                 color, 1, cv2.LINE_AA)

        (tw, th), _ = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
        lx = mid_x - tw // 2
        ly = int(h * 0.25)
        cv2.rectangle(frame, (lx - 3, ly - th - 2), (lx + tw + 3, ly + 2),
                      (15, 15, 15), -1)
        cv2.putText(frame, name, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)