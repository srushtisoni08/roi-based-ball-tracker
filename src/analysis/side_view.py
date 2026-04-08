import numpy as np
from src.models.track_point import TrackPoint
from src.analysis.trajectory_utils import _smooth
from src.config import CFG


def analyse_side(track: list[TrackPoint], fps: float, frame_width: int) -> tuple:
    """
    Side-view bounce and length analysis.

    Bounce detection:
    ─────────────────
    Ball descends (Y increases in image coords) then reverses after pitch contact.
    Requires sustained descent (min_descent_frames) before reversal, and the
    reversal must be at least bounce_reversal_px — now 15px instead of 2px so
    detection noise (3-5px jitter) cannot trigger a false bounce.
    """
    if len(track) < 3:
        return False, None, "Unknown"

    ys = [p.y for p in track]
    xs = [p.x for p in track]

    smoothed_y = _smooth(ys, window=7)

    min_desc = CFG["min_descent_frames"]   # 5
    rev_px   = CFG["bounce_reversal_px"]   # 15

    bounced     = False
    bounce_i    = None
    desc_count  = 0
    local_max_y = smoothed_y[0]

    for i in range(1, len(smoothed_y)):
        dy = smoothed_y[i] - smoothed_y[i - 1]

        if dy > 1.0:
            desc_count  += 1
            local_max_y  = max(local_max_y, smoothed_y[i])
        elif dy < -1.0:
            reversal_mag = local_max_y - smoothed_y[i]
            if desc_count >= min_desc and reversal_mag >= rev_px:
                bounced  = True
                bounce_i = i
                break
            desc_count  = 0
            local_max_y = smoothed_y[i]
        # flat: keep counting descent

    bounce_pt = (track[bounce_i].x, track[bounce_i].y) if bounce_i is not None else None

    # ── Length classification ──────────────────────────────────────
    if bounce_i is not None:
        ref_x = track[bounce_i].x
    else:
        last_third = xs[max(0, len(xs) * 2 // 3):]
        ref_x = int(np.median(last_third)) if last_third else int(np.median(xs))

    x_frac = ref_x / max(frame_width, 1)
    length  = _classify_length_side(x_frac)

    return bounced, bounce_pt, length


def _classify_length_side(x_frac: float) -> str:
    for name, (lo, hi) in CFG["length_zones_side"].items():
        if lo <= x_frac < hi:
            return name
    return "Unknown"