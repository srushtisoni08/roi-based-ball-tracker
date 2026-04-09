import numpy as np
from src.models.track_point import TrackPoint
from src.analysis.trajectory_utils import _smooth
from src.config import CFG


def analyse_side(track: list[TrackPoint], fps: float, frame_width: int) -> tuple:
    """
    Side-view bounce detection using peak-based method.

    In side view the ball descends (Y increases) then bounces upward (Y decreases).
    Peak-based detection: find the highest Y, check there is real descent before
    and real ascent after. More robust than state-machine for short tracks.

    Parameters from config:
      bounce_reversal_px  = 12  (min descent/ascent magnitude)
      min_descent_frames  = 3   (min frames of descent before peak)
    """
    if len(track) < 4:
        return False, None, "Unknown"

    ys = [p.y for p in track]
    xs = [p.x for p in track]

    smoothed_y = _smooth(ys, window=5)

    min_desc = CFG["min_descent_frames"]   # 3
    rev_px   = CFG["bounce_reversal_px"]   # 12

    bounced  = False
    bounce_i = None

    # ── Primary: peak-based ───────────────────────────────────────
    # Find the Y maximum — this is where the ball contacts the pitch.
    # Require: enough frames of descent before it AND real ascent after it.
    peak_i = int(np.argmax(smoothed_y))

    if 0 < peak_i < len(smoothed_y) - 1:
        descent = smoothed_y[peak_i] - smoothed_y[0]
        ascent  = smoothed_y[peak_i] - min(smoothed_y[peak_i:])

        frames_before_peak = peak_i  # number of frames before peak

        if (frames_before_peak >= min_desc and
                descent >= rev_px and
                ascent  >= rev_px):
            bounced  = True
            bounce_i = peak_i

    # ── Fallback: state-machine ───────────────────────────────────
    if not bounced:
        desc_count  = 0
        local_max_y = smoothed_y[0]

        for i in range(1, len(smoothed_y)):
            dy = smoothed_y[i] - smoothed_y[i - 1]
            if dy > 1.0:
                desc_count  += 1
                local_max_y  = max(local_max_y, smoothed_y[i])
            elif dy < -2.0:
                rev = local_max_y - smoothed_y[i]
                if desc_count >= min_desc and rev >= rev_px:
                    bounced  = True
                    bounce_i = i
                    break
                desc_count  = 0
                local_max_y = smoothed_y[i]

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