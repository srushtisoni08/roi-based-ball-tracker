import numpy as np
from models.track_point import TrackPoint
from analysis.trajectory_utils import _smooth
from config import CFG

def analyse_front(track: list[TrackPoint], fps: float, frame_height: int) -> tuple:
    """
    Front view: ball grows as it approaches.
    Bounce → sudden radius jump followed by trajectory shift.
    Length  → Y position (height in frame) at bounce / mid point.
    """
    radii = [p.radius for p in track]
    ys    = [p.y      for p in track]
    smoothed_r = _smooth(radii, 3)
 
    # ── Bounce ──────────────────────────────
    w = CFG["front_bounce_window"]
    bounced  = False
    bounce_i = None
 
    for i in range(len(smoothed_r) - w):
        ratio = smoothed_r[i + w] / (smoothed_r[i] + 0.01)
        if ratio >= CFG["front_bounce_size_jump"]:
            bounced  = True
            bounce_i = i + w // 2
            break
 
    bounce_pt = (track[bounce_i].x, track[bounce_i].y) if bounce_i else None
 
    # ── Length via Y position ────────────────
    ref_y  = track[bounce_i].y if bounce_i else int(np.median(ys))
    y_frac = ref_y / frame_height
    length = _classify_length_front(y_frac)
 
    return bounced, bounce_pt, length
 
 
def _classify_length_front(y_frac: float) -> str:
    """
    Front view: ball appears lower in frame the closer it is to batsman.
    Approximate mapping (calibrate with actual footage):
      0.55–1.00 → Yorker   (very close, low in frame)
      0.40–0.55 → Full
      0.25–0.40 → Good
      0.00–0.25 → Short    (far away, high in frame)
    """
    if y_frac >= 0.55:   return "Yorker"
    if y_frac >= 0.40:   return "Full"
    if y_frac >= 0.25:   return "Good"
    return "Short"