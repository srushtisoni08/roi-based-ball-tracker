import numpy as np
from src.models.track_point import TrackPoint
from src.analysis.trajectory_utils import _smooth
from src.config import CFG

def analyse_front(track: list[TrackPoint], fps: float, frame_height: int) -> tuple:
    """
    Behind-bowler view: ball travels AWAY from camera toward batsman.
    - Y decreases as ball moves down the pitch
    - Radius shrinks as ball gets further away
    Bounce → sudden upward deflection (Y briefly increases after decreasing)
    Length → Y position at bounce point (lower Y = further down pitch)
    """
    ys       = [p.y for p in track]
    xs       = [p.x for p in track]
    radii    = [p.radius for p in track]
    smoothed_y = _smooth(ys, 5)
    smoothed_r = _smooth(radii, 3)

    # ── Bounce detection ─────────────────────────────────────────
    # Ball goes away (Y decreasing). Bounce = Y stops decreasing and
    # briefly increases (ball pops up off pitch), then decreases again.
    bounced  = False
    bounce_i = None
    w = CFG["front_bounce_window"]

    for i in range(w, len(smoothed_y) - w):
        pre_dy  = smoothed_y[i]     - smoothed_y[i - w]   # should be negative (going away)
        post_dy = smoothed_y[i + w] - smoothed_y[i]       # positive = ball popped up

        # Ball was going away (pre_dy < -5) then bounced up (post_dy > 5)
        if pre_dy < -5 and post_dy > 5:
            bounced  = True
            bounce_i = i
            break

    # Fallback: check radius shrink then grow (ball hits pitch and gets closer briefly)
    if not bounced:
        for i in range(w, len(smoothed_r) - w):
            pre_dr  = smoothed_r[i]     - smoothed_r[i - w]
            post_dr = smoothed_r[i + w] - smoothed_r[i]
            if pre_dr < -0.5 and post_dr > 0.5:
                bounced  = True
                bounce_i = i
                break

    bounce_pt = (track[bounce_i].x, track[bounce_i].y) if bounce_i else None

    # ── Length via Y position at bounce ──────────────────────────
    # In behind-bowler view:
    #   High Y (bottom of frame) = ball still close to bowler = Short
    #   Low Y (top of frame)     = ball far down pitch near batsman = Full/Yorker
    if bounce_i:
        ref_y = track[bounce_i].y
    else:
        # No bounce detected — use the lowest Y reached (furthest point)
        ref_y = min(ys)
    print(f"[LENGTH] ref_y={ref_y} frame_height={frame_height} y_frac={ref_y/frame_height:.2f} bounce_i={bounce_i}")
    y_frac = ref_y / frame_height
    length = _classify_length_front(y_frac)

    return bounced, bounce_pt, length


def _classify_length_front(y_frac: float) -> str:
    """
    Behind-bowler view — ball travels away so lower y_frac = further down pitch.
      0.00–0.30 → Yorker   (very far, near batsman feet)
      0.30–0.50 → Full
      0.50–0.65 → Good
      0.65–1.00 → Short    (close to bowler end)
    """
    if y_frac <= 0.30:   return "Yorker"
    if y_frac <= 0.50:   return "Full"
    if y_frac <= 0.65:   return "Good"
    return "Short"