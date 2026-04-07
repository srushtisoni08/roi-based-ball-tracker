import numpy as np
from src.models.track_point import TrackPoint
from src.analysis.trajectory_utils import _smooth
from src.config import CFG


def analyse_side(track: list[TrackPoint], fps: float, frame_width: int) -> tuple:
    """
    Returns (bounced: bool, bounce_pt: (x,y)|None, length: str)

    Bounce detection strategy:
    ─────────────────────────
    A genuine bounce shows a clear descent (Y increasing in image coords)
    followed immediately by an ascent (Y decreasing).  We require BOTH
    windows to satisfy their condition to avoid single-frame spikes
    triggering a false bounce.

    The algorithm:
      1. Smooth Y with Gaussian window to suppress noise.
      2. Walk forward, counting consecutive descent frames.
      3. At the first ascent frame, check that the preceding descent run
         is long enough (min_descent_frames).  If so → bounce confirmed.
      4. Also require the reversal magnitude (max_y − y_at_bounce) to be
         at least `bounce_reversal_px` pixels so tiny tremors don't count.

    Length strategy:
    ─────────────────
    • If a bounce is found → use the X coordinate at the bounce point.
    • If no bounce (full toss / yorker) → use the median X of the last
      third of the track, where the ball is nearest the batsman.
      This avoids being biased by the far-away early frames.
    """
    if len(track) < 3:
        return False, None, "Unknown"

    ys = [p.y for p in track]
    xs = [p.x for p in track]

    # Use larger smoothing window for stability
    smoothed_y = _smooth(ys, window=7)

    min_desc = CFG["min_descent_frames"]
    rev_px   = CFG["bounce_reversal_px"]

    bounced  = False
    bounce_i = None
    desc_count = 0
    local_max_y = smoothed_y[0]

    for i in range(1, len(smoothed_y)):
        dy = smoothed_y[i] - smoothed_y[i - 1]

        if dy > rev_px:
            # Ball descending (Y growing in image coords = moving down)
            desc_count += 1
            local_max_y = max(local_max_y, smoothed_y[i])
        elif dy < -rev_px:
            # Ball ascending — check if we had enough descent before
            if desc_count >= min_desc:
                # Additional magnitude check: reversal must be real, not tiny
                reversal_magnitude = local_max_y - smoothed_y[i]
                if reversal_magnitude >= rev_px:
                    bounced  = True
                    bounce_i = i
                    break
            # Reset descent count if the pattern breaks
            desc_count  = 0
            local_max_y = smoothed_y[i]
        else:
            # Flat — don't reset desc_count (ball may pause briefly at bounce)
            pass

    bounce_pt = (track[bounce_i].x, track[bounce_i].y) if bounce_i is not None else None

    # ── Length classification ──────────────────────────────────────
    if bounce_i is not None:
        ref_x = track[bounce_i].x
    else:
        # Full toss / yorker: use median X from the last third of track
        last_third = xs[max(0, len(xs) * 2 // 3):]
        ref_x = int(np.median(last_third)) if last_third else int(np.median(xs))

    x_frac = ref_x / frame_width
    length  = _classify_length_side(x_frac)

    return bounced, bounce_pt, length


def _classify_length_side(x_frac: float) -> str:
    for name, (lo, hi) in CFG["length_zones_side"].items():
        if lo <= x_frac < hi:
            return name
    return "Unknown"