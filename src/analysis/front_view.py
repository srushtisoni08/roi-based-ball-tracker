import numpy as np
from src.models.track_point import TrackPoint
from src.analysis.trajectory_utils import _smooth
from src.config import CFG


def analyse_front(track: list[TrackPoint], fps: float, frame_height: int) -> tuple:
    """
    Front-view analysis.

    Bounce detection:
    ─────────────────
    In a front view the ball grows as it approaches the camera.
    At the bounce:
      1. The ball dips downward (Y increases) then rises (Y decreases).
      2. The apparent radius may jump slightly as the ball kicks up.

    We use a dual-signal approach:
      Signal A — Y reversal (same logic as side view, but on Y in front view).
      Signal B — radius growth burst (original approach).
    Either signal alone can confirm the bounce so recall is higher.

    Length classification:
    ─────────────────────
    Y position in frame (closer = lower in frame = larger y_frac).
    Zones calibrated for typical broadcast front-view angles.
    """
    if len(track) < 3:
        return False, None, "Unknown"

    radii = [p.radius for p in track]
    ys    = [p.y      for p in track]

    smoothed_r = _smooth(radii, window=5)
    smoothed_y = _smooth(ys,    window=5)

    bounced  = False
    bounce_i = None

    # ── Signal A: Y reversal (descent → ascent) ───────────────────
    min_desc   = CFG["min_descent_frames"]
    rev_px     = CFG["bounce_reversal_px"]
    desc_count = 0
    local_max_y = smoothed_y[0]

    for i in range(1, len(smoothed_y)):
        dy = smoothed_y[i] - smoothed_y[i - 1]
        if dy > rev_px:
            desc_count  += 1
            local_max_y  = max(local_max_y, smoothed_y[i])
        elif dy < -rev_px:
            if desc_count >= min_desc and (local_max_y - smoothed_y[i]) >= rev_px:
                bounced  = True
                bounce_i = i
                break
            desc_count  = 0
            local_max_y = smoothed_y[i]

    # ── Signal B: radius burst (fallback if Y signal fails) ───────
    if not bounced:
        w = CFG["front_bounce_window"]
        for i in range(len(smoothed_r) - w):
            if smoothed_r[i] < 0.5:
                continue
            ratio = smoothed_r[i + w] / (smoothed_r[i] + 0.01)
            if ratio >= CFG["front_bounce_size_jump"]:
                bounced  = True
                bounce_i = i + w // 2
                break

    bounce_pt = (track[bounce_i].x, track[bounce_i].y) if bounce_i is not None else None

    # ── Length ────────────────────────────────────────────────────
    if bounce_i is not None:
        ref_y = track[bounce_i].y
    else:
        # No bounce: use median Y of the last third (ball near batsman)
        last_ys   = ys[max(0, len(ys) * 2 // 3):]
        ref_y     = int(np.median(last_ys)) if last_ys else int(np.median(ys))

    y_frac = ref_y / frame_height
    length = _classify_length_front(y_frac)

    return bounced, bounce_pt, length


def _classify_length_front(y_frac: float) -> str:
    """
    Front view: ball appears lower in frame the closer it is to batsman.
      ≥ 0.58 → Yorker   (very close, very low)
      ≥ 0.42 → Full
      ≥ 0.26 → Good
         else → Short   (far, high in frame)
    """
    if y_frac >= 0.58:
        return "Yorker"
    if y_frac >= 0.42:
        return "Full"
    if y_frac >= 0.26:
        return "Good"
    return "Short"