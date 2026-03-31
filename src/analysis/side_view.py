import numpy as np
from src.models.track_point import TrackPoint
from src.analysis.trajectory_utils import _smooth
from src.config import CFG

def analyse_side(track: list[TrackPoint], fps: float, frame_width: int) -> tuple:
    """
    Returns (bounced: bool, bounce_pt: (x,y)|None, length: str)
 
    Bounce detection:  Y reversal (down → up)
    Length detection:  X position where bounce occurs (or mid-trajectory X)
    """
    ys = [p.y for p in track]
    xs = [p.x for p in track]
    smoothed_y = _smooth(ys, 3)
 
    # ── Bounce ──────────────────────────────
    descent  = 0
    bounced  = False
    bounce_i = None
 
    for i in range(1, len(smoothed_y)):
        dy = smoothed_y[i] - smoothed_y[i - 1]
        if dy > CFG["bounce_reversal_px"]:
            descent += 1
        elif dy < -CFG["bounce_reversal_px"] and descent >= CFG["min_descent_frames"]:
            bounced  = True
            bounce_i = i
            break
 
    bounce_pt = (track[bounce_i].x, track[bounce_i].y) if bounce_i else None
 
    # ── Length ──────────────────────────────
    # Use X position of bounce point (or median X if no bounce)
    ref_x  = track[bounce_i].x if bounce_i else int(np.median(xs))
    x_frac = ref_x / frame_width
    length = _classify_length_side(x_frac)
 
    return bounced, bounce_pt, length
 
 
def _classify_length_side(x_frac: float) -> str:
    for name, (lo, hi) in CFG["length_zones_side"].items():
        if lo <= x_frac < hi:
            return name
    return "Unknown"