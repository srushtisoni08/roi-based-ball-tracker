import numpy as np
from src.models.track_point import TrackPoint
from src.analysis.trajectory_utils import _smooth
from src.config import CFG


def analyse_front(track: list[TrackPoint], fps: float, frame_height: int) -> tuple:
    """
    Behind-bowler view analysis.
    Detects bounce only — length classification disabled as ball is
    only reliably detected near bowler, not down the pitch.
    """
    ys        = [p.y for p in track]
    smoothed_y = _smooth(ys, 5)

    # ── Bounce detection ─────────────────────────────────────────
    bounced  = False
    bounce_i = None
    w = CFG["front_bounce_window"]

    for i in range(w, len(smoothed_y) - w):
        pre_dy  = smoothed_y[i]     - smoothed_y[i - w]
        post_dy = smoothed_y[i + w] - smoothed_y[i]
        if pre_dy < -5 and post_dy > 5:
            bounced  = True
            bounce_i = i
            break

    bounce_pt = (track[bounce_i].x, track[bounce_i].y) if bounce_i else None

    # Length not classifiable from behind-bowler view
    length = "N/A"

    return bounced, bounce_pt, length