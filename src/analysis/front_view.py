import numpy as np
from src.models.track_point import TrackPoint
from src.analysis.trajectory_utils import _smooth
from src.config import CFG


def analyse_front(track: list[TrackPoint], fps: float, frame_height: int) -> tuple:
    """
    Front/behind-bowler view analysis for portrait phone videos.

    In this view the ball is bowled TOWARD the camera, so:
      - Y increases as ball travels down the pitch (toward camera/batsman)
      - After the bounce the ball rises, so Y decreases
      - Ball also appears to grow in radius as it approaches

    Bounce detection uses two signals:
      1. Y-peak + drop: find the highest Y in the first 82% of the track,
         then check that Y genuinely drops by at least MIN_DROP afterwards.
         This replaces the old windowed-velocity approach which fired on
         any deceleration - including a full toss running out of track.
      2. Radius jump: ball radius increases suddenly at bounce point.
    """
    if len(track) < 6:
        return False, None, "N/A"

    ys = np.array([p.y for p in track], dtype=float)
    rs = np.array([getattr(p, 'radius', 5.0) for p in track], dtype=float)

    smoothed_y = _smooth(list(ys), 5)
    smoothed_r = _smooth(list(rs), 5)

    n = len(smoothed_y)

    # Signal 1: Y-peak then genuine drop
    # MIN_DROP: 2% of frame height (38px at 1920, 17px at 832).
    # Filters out noise wobbles and camera shake while catching real bounces.
    MIN_DROP = max(frame_height * 0.02, 10.0)

    search_start = max(1, int(n * 0.10))
    search_end   = max(search_start + 3, int(n * 0.82))

    bounced  = False
    bounce_i = None

    if search_end > search_start:
        window         = smoothed_y[search_start:search_end]
        peak_idx_local = int(np.argmax(window))
        peak_i         = search_start + peak_idx_local
        peak_val       = smoothed_y[peak_i]

        pre_start = max(0, peak_i - 8)
        pre_rise  = peak_val - smoothed_y[pre_start]

        if pre_rise >= 5.0:
            post_vals = smoothed_y[peak_i:]
            if len(post_vals) >= 3:
                post_min = float(np.min(post_vals))
                drop     = peak_val - post_min
                if drop >= MIN_DROP:
                    bounced  = True
                    bounce_i = peak_i

    # Signal 2: Radius jump - only runs if Signal 1 did not fire
    w = max(3, min(CFG.get("front_bounce_window", 8), n // 4))
    size_jump = CFG.get("front_bounce_size_jump", 1.18)

    if not bounced and len(smoothed_r) > w * 2:
        for i in range(w, len(smoothed_r) - w):
            if i > n * 0.82:
                break
            pre_r  = float(np.mean(smoothed_r[max(0, i - w): i]))
            post_r = float(np.mean(smoothed_r[i: min(len(smoothed_r), i + w)]))
            if pre_r > 0 and post_r / pre_r >= size_jump:
                if smoothed_y[i] > smoothed_y[max(0, i - w)]:
                    bounced  = True
                    bounce_i = i
                    break

    bounce_pt = None
    if bounce_i is not None:
        idx       = min(bounce_i, len(track) - 1)
        bounce_pt = (track[idx].x, track[idx].y)

    return bounced, bounce_pt, "N/A"