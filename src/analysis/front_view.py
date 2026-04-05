import numpy as np
from src.models.track_point import TrackPoint
from src.analysis.trajectory_utils import _smooth
from src.config import CFG


def analyse_front(track: list[TrackPoint], fps: float, frame_height: int) -> tuple:
    """
    Front/behind-bowler view analysis for portrait phone videos.

    In this view the ball is bowled TOWARD the camera, so:
      - Y increases as ball travels down the pitch (toward camera/batsman)
      - After the bounce the ball rises, so Y decreases (or slows)
      - Ball also appears to grow in radius as it approaches

    Bounce detection uses two signals:
      1. Y-velocity reversal: Y was increasing fast, then slows/reverses
      2. Radius jump: ball radius increases suddenly at bounce point
    """
    if len(track) < 6:
        return False, None, "N/A"

    ys = np.array([p.y for p in track], dtype=float)
    rs = np.array([getattr(p, 'radius', 5.0) for p in track], dtype=float)

    smoothed_y = _smooth(list(ys), 5)
    smoothed_r = _smooth(list(rs), 5)

    bounced  = False
    bounce_i = None

    n = len(smoothed_y)
    w = max(3, min(CFG.get("front_bounce_window", 8), n // 4))

    # ── Signal 1: Y-velocity reversal ─────────────────────────────
    # Ball approaching camera → Y increases (positive vy).
    # After bounce it rises → Y decreases or decelerates sharply.
    # We look for a local MAXIMUM in Y (peak descent = bounce point).
    best_score = 0.0
    for i in range(w, n - w):
        pre_dy  = smoothed_y[i] - smoothed_y[i - w]   # should be positive (moving down)
        post_dy = smoothed_y[i + w] - smoothed_y[i]   # should be negative or small (rising)

        # Ball was moving toward camera (Y increasing) and then rises/slows
        if pre_dy > 5 and post_dy < pre_dy * 0.4:
            score = pre_dy - post_dy
            if score > best_score:
                best_score = score
                bounce_i   = i
                bounced    = True

    # ── Signal 2: Radius jump (ball suddenly appears larger at bounce) ──
    size_jump = CFG.get("front_bounce_size_jump", 1.18)
    if not bounced and len(smoothed_r) > w * 2:
        for i in range(w, len(smoothed_r) - w):
            pre_r  = float(np.mean(smoothed_r[max(0, i - w): i]))
            post_r = float(np.mean(smoothed_r[i: min(len(smoothed_r), i + w)]))
            if pre_r > 0 and post_r / pre_r >= size_jump:
                # Confirm Y is still generally increasing (ball moving toward camera)
                if smoothed_y[i] > smoothed_y[max(0, i - w)]:
                    bounced  = True
                    bounce_i = i
                    break

    bounce_pt = None
    if bounce_i is not None:
        bounce_pt = (track[min(bounce_i, len(track) - 1)].x,
                     track[min(bounce_i, len(track) - 1)].y)

    return bounced, bounce_pt, "N/A"