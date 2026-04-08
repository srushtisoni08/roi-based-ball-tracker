import cv2
import numpy as np
from collections import deque
from src.config import CFG
from src.models.delivery_result import DeliveryResult


# ── Ball trail with glowing path line ───────────────────────────
def draw_ball_trail(frame, trail, ball_no: int = 0):
    """
    Draws a smooth glowing trajectory line connecting all detected
    ball positions. Three layered passes:
      1. Wide soft glow (thick, low opacity blended)
      2. Bright core line (thin, full brightness)
      3. White hot centre line (1px, newest segment only)
    Plus a sharp dot at the current ball position.
    """
    pts = list(trail)
    if len(pts) < 2:
        return

    n = len(pts)

    # ── Pass 1: outer glow ────────────────────────────────────────
    overlay = frame.copy()
    for i in range(1, n):
        alpha = i / n
        # Glow colour — soft cyan-yellow tint
        glow_color = (
            int(80  * alpha),
            int(200 * alpha),
            int(255 * alpha),
        )
        thickness = max(2, int(6 * alpha))
        cv2.line(overlay, pts[i - 1], pts[i], glow_color, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # ── Pass 2: bright core line ──────────────────────────────────
    for i in range(1, n):
        alpha = i / n
        core_color = (
            int(CFG["color_trail"][0] * alpha),
            int(CFG["color_trail"][1] * alpha),
            int(CFG["color_trail"][2] * alpha),
        )
        thickness = max(1, int(2 * alpha))
        cv2.line(frame, pts[i - 1], pts[i], core_color, thickness, cv2.LINE_AA)

    # ── Pass 3: white-hot newest 10 points ───────────────────────
    recent = pts[max(0, n - 10):]
    for i in range(1, len(recent)):
        cv2.line(frame, recent[i - 1], recent[i], (255, 255, 255), 1, cv2.LINE_AA)

    # ── Tip dot ───────────────────────────────────────────────────
    tip = pts[-1]
    cv2.circle(frame, tip, 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, tip, 8, CFG["color_ball"],  1, cv2.LINE_AA)


# ── Bounce marker ────────────────────────────────────────────────
def draw_bounce_marker(frame, pt, label: str = ""):
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(frame, (x, y), 18, CFG["color_bounce"],  1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 10, CFG["color_bounce"],  2, cv2.LINE_AA)
    cv2.circle(frame, (x, y),  4, CFG["color_bounce"], -1, cv2.LINE_AA)
    cv2.line(frame, (x - 20, y), (x - 11, y), CFG["color_bounce"], 1, cv2.LINE_AA)
    cv2.line(frame, (x + 11, y), (x + 20, y), CFG["color_bounce"], 1, cv2.LINE_AA)
    cv2.line(frame, (x, y - 20), (x, y - 11), CFG["color_bounce"], 1, cv2.LINE_AA)
    cv2.line(frame, (x, y + 11), (x, y + 20), CFG["color_bounce"], 1, cv2.LINE_AA)


# ── Minimal bounce/toss indicator (top-right corner) ────────────
def draw_result_badge(frame, result: DeliveryResult, frame_no: int, show_until: dict):
    """
    Small badge in top-right showing only BOUNCED or FULL TOSS.
    No length labels, no zones, no banners.
    """
    if result is None:
        return
    if frame_no > show_until.get(result.ball_no, 0):
        return

    h, w = frame.shape[:2]

    if result.bounced:
        text  = "BOUNCED"
        color = CFG["color_bounce"]
    else:
        text  = "FULL TOSS"
        color = CFG["color_no_bounce"]

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad  = 10
    rx1  = w - tw - pad * 3
    rx2  = w - pad
    ry1  = pad
    ry2  = pad + th + pad

    # Dark semi-transparent pill
    overlay = frame.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)

    # Coloured left accent bar
    cv2.rectangle(frame, (rx1, ry1), (rx1 + 4, ry2), color, -1)

    # Text
    cv2.putText(frame, text, (rx1 + 12, ry2 - pad // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


# ── Stubbed-out functions kept for import compatibility ──────────
# processor.py still calls these — they now do nothing so no zones,
# no HUD table, and no bottom banner appear in the output video.

def draw_hud(frame, results_so_far, current_ball_no, total_balls: int = 6):
    """Removed — no HUD table drawn."""
    pass


def draw_length_banner(frame, result, frame_no: int, banner_until: dict):
    """Removed — no bottom length banner drawn."""
    pass


def draw_pitch_zones_side(frame):
    """Removed — no Full/Good/Yorker/Short zone lines drawn."""
    pass