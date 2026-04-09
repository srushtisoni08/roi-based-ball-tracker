import cv2
import numpy as np
from collections import deque
from src.config import CFG
from src.models.delivery_result import DeliveryResult


def draw_ball_trail(frame, trail, ball_no: int = 0):
    """
    Glowing motion trail — 3 layered passes for depth effect.
    trail: deque or list of (x, y) tuples.
    """
    pts = list(trail)
    if len(pts) < 2:
        return

    n = len(pts)

    # Pass 1: wide soft glow
    overlay = frame.copy()
    for i in range(1, n):
        a = i / n
        cv2.line(overlay, pts[i-1], pts[i],
                 (int(20*a), int(160*a), int(255*a)),
                 max(4, int(12*a)), cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.38, frame, 0.62, 0, frame)

    # Pass 2: bright core line
    for i in range(1, n):
        a = i / n
        cv2.line(frame, pts[i-1], pts[i],
                 (0, int(230*a), int(255*a)),
                 max(1, int(3*a)), cv2.LINE_AA)

    # Pass 3: white-hot newest 8 points
    tip = pts[max(0, n-8):]
    for i in range(1, len(tip)):
        cv2.line(frame, tip[i-1], tip[i], (255, 255, 255), 2, cv2.LINE_AA)

    # Ball tip dot
    cv2.circle(frame, pts[-1], 5, (255, 255, 255), -1, cv2.LINE_AA)
    cv2.circle(frame, pts[-1], 8, CFG["color_ball"],  1, cv2.LINE_AA)


def draw_bounce_marker(frame, pt, label: str = ""):
    x, y = int(pt[0]), int(pt[1])
    cv2.circle(frame, (x, y), 18, CFG["color_bounce"],  1, cv2.LINE_AA)
    cv2.circle(frame, (x, y), 10, CFG["color_bounce"],  2, cv2.LINE_AA)
    cv2.circle(frame, (x, y),  4, CFG["color_bounce"], -1, cv2.LINE_AA)
    cv2.line(frame, (x-20, y), (x-11, y), CFG["color_bounce"], 1, cv2.LINE_AA)
    cv2.line(frame, (x+11, y), (x+20, y), CFG["color_bounce"], 1, cv2.LINE_AA)
    cv2.line(frame, (x, y-20), (x, y-11), CFG["color_bounce"], 1, cv2.LINE_AA)
    cv2.line(frame, (x, y+11), (x, y+20), CFG["color_bounce"], 1, cv2.LINE_AA)


def draw_result_badge(frame, result: DeliveryResult, frame_no: int, show_until: dict):
    """Small top-right badge: BOUNCED or FULL TOSS only. No length labels."""
    if result is None:
        return
    if frame_no > show_until.get(result.ball_no, 0):
        return

    h, w = frame.shape[:2]
    text  = "BOUNCED" if result.bounced else "FULL TOSS"
    color = CFG["color_bounce"] if result.bounced else CFG["color_no_bounce"]

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    pad = 10
    rx1, rx2 = w - tw - pad*3, w - pad
    ry1, ry2 = pad, pad + th + pad

    overlay = frame.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
    cv2.rectangle(frame, (rx1, ry1), (rx1+4, ry2), color, -1)
    cv2.putText(frame, text, (rx1+12, ry2 - pad//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


# Stubs — kept for import compatibility, intentionally do nothing
def draw_hud(frame, results_so_far, current_ball_no, total_balls: int = 6):
    pass

def draw_length_banner(frame, result, frame_no: int, banner_until: dict):
    pass

def draw_pitch_zones_side(frame):
    pass