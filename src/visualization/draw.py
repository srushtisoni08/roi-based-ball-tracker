import cv2
import numpy as np
from collections import deque
from src.config import CFG
from src.models.delivery_result import DeliveryResult


def draw_ball_trail(frame, trail, ball_no: int = 0):
    """
    Glowing motion trail with 3 layered passes for depth/glow effect.
    trail: deque or list of (x, y) tuples.
    """
    pts = list(trail)
    if len(pts) < 2:
        return

    n = len(pts)

    # Pass 1: wide soft glow layer
    overlay = frame.copy()
    for i in range(1, n):
        a = i / n
        thickness = max(4, int(14 * a))
        cv2.line(overlay, pts[i-1], pts[i],
                 (int(20*a), int(160*a), int(255*a)),
                 thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

    # Pass 2: bright core line
    for i in range(1, n):
        a = i / n
        cv2.line(frame, pts[i-1], pts[i],
                 (0, int(230*a), int(255*a)),
                 max(1, int(4*a)), cv2.LINE_AA)

    # Pass 3: white-hot newest 10 points
    tip = pts[max(0, n-10):]
    for i in range(1, len(tip)):
        cv2.line(frame, tip[i-1], tip[i], (255, 255, 255), 2, cv2.LINE_AA)

    # Ball tip dot — double ring for visibility on any background
    if pts:
        cx, cy = pts[-1]
        cv2.circle(frame, (cx, cy),  9, (0, 200, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy),  5, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 12, CFG["color_ball"], 1, cv2.LINE_AA)


def draw_bounce_marker(frame, pt, label: str = ""):
    """Crosshair marker at bounce location."""
    x, y = int(pt[0]), int(pt[1])
    color = CFG["color_bounce"]

    # Outer rings
    cv2.circle(frame, (x, y), 22, color, 1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 13, color, 2, cv2.LINE_AA)
    cv2.circle(frame, (x, y),  5, color, -1, cv2.LINE_AA)

    # Cross ticks
    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        x1 = x + dx * 25
        y1 = y + dy * 25
        x2 = x + dx * 14
        y2 = y + dy * 14
        cv2.line(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    if label:
        cv2.putText(frame, label, (x + 15, y - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_result_badge(frame, result: DeliveryResult, frame_no: int, show_until: dict):
    """Top-right badge: BOUNCED or FULL TOSS."""
    if result is None:
        return
    if frame_no > show_until.get(result.ball_no, 0):
        return

    h, w = frame.shape[:2]
    text  = "BOUNCED" if result.bounced else "FULL TOSS"
    color = CFG["color_bounce"] if result.bounced else CFG["color_no_bounce"]

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.72, 2)
    pad = 12
    rx1, rx2 = w - tw - pad*3, w - pad
    ry1, ry2 = pad, pad + th + pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    cv2.rectangle(frame, (rx1, ry1), (rx1+5, ry2), color, -1)
    cv2.putText(frame, text, (rx1+14, ry2 - pad//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.72, color, 2, cv2.LINE_AA)


def draw_delivery_counter(frame, ball_no: int, total: int):
    """Small top-left delivery counter overlay."""
    text = f"Ball {ball_no} / {total}"
    cv2.putText(frame, text, (14, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (220, 220, 220), 1, cv2.LINE_AA)


def draw_speed_label(frame, speed_kmh: float, pos: tuple):
    """Draw a small speed annotation near the ball."""
    if speed_kmh <= 0:
        return
    text = f"{speed_kmh:.0f} km/h"
    x, y = int(pos[0]) + 14, int(pos[1]) - 10
    cv2.putText(frame, text, (x, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 100), 1, cv2.LINE_AA)


# ── Stubs — kept for import compatibility ────────────────────────
def draw_hud(frame, results_so_far, current_ball_no, total_balls: int = 6):
    pass

def draw_length_banner(frame, result, frame_no: int, banner_until: dict):
    pass

def draw_pitch_zones_side(frame):
    pass