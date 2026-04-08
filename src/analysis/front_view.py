import numpy as np
from src.models.track_point import TrackPoint
from src.analysis.trajectory_utils import _smooth
from src.config import CFG


def analyse_front(track: list[TrackPoint], fps: float, frame_height: int) -> tuple:
    """
    Front-view bounce and length analysis.

    Camera setups handled:
    ──────────────────────
    A) Portrait / bowler-POV (e.g. phone behind bowler looking down pitch):
       - Ball travels DOWN the frame (Y increases) as it moves away
       - A real bounce: ball descends steeply then sharply reverses upward
       - Radius SHRINKS as ball moves away — do NOT use radius growth signal

    B) Landscape broadcast (camera at side-on facing batsman end):
       - Ball grows in radius as it approaches camera
       - Y dips then rises at bounce

    We detect the camera orientation from frame aspect ratio and use
    the correct signal for each case.

    Bounce detection (portrait — primary case):
    ───────────────────────────────────────────
    Signal A (Y reversal): ball descends >= min_descent_frames then
    reverses by >= bounce_reversal_px. Both thresholds are now in
    config at realistic values (15px, 5 frames) instead of noise-level
    values (2px, 2 frames) that caused every wobble to look like a bounce.

    Bounce detection (landscape — secondary case):
    ───────────────────────────────────────────────
    Signal A (Y reversal): same logic.
    Signal B (radius burst): ball grows significantly as it nears camera.
    Only used for landscape where ball approaches camera.
    """
    if len(track) < 3:
        return False, None, "Unknown"

    radii = [p.radius for p in track]
    ys    = [p.y      for p in track]
    xs    = [p.x      for p in track]

    smoothed_r = _smooth(radii, window=5)
    smoothed_y = _smooth(ys,    window=7)

    # Detect orientation: portrait = frame_height >> frame_width equivalent
    # We infer from the track's x-spread vs y-spread
    x_spread = max(xs) - min(xs) if xs else 1
    y_spread = max(ys) - min(ys) if ys else 1
    # In portrait bowler-POV: ball moves mostly in Y (down the pitch = down the frame)
    # In landscape: ball moves mostly in X
    is_portrait_pov = (y_spread > x_spread * 1.5) or (frame_height > 1000)

    bounced  = False
    bounce_i = None

    min_desc = CFG["min_descent_frames"]    # 5 — needs sustained descent
    rev_px   = CFG["bounce_reversal_px"]    # 15 — needs real reversal, not noise

    # ── Signal A: Y reversal (descent then ascent) ────────────────
    desc_count  = 0
    local_max_y = smoothed_y[0]

    for i in range(1, len(smoothed_y)):
        dy = smoothed_y[i] - smoothed_y[i - 1]
        if dy > 1.0:                        # descending (Y increasing in image)
            desc_count  += 1
            local_max_y  = max(local_max_y, smoothed_y[i])
        elif dy < -1.0:                     # ascending
            reversal_mag = local_max_y - smoothed_y[i]
            if desc_count >= min_desc and reversal_mag >= rev_px:
                bounced  = True
                bounce_i = i
                break
            desc_count  = 0
            local_max_y = smoothed_y[i]
        # flat: don't reset

    # ── Signal B: radius burst — ONLY for landscape (ball approaching) ──
    # In portrait/bowler-POV the ball moves AWAY so radius shrinks.
    # Using this signal on portrait video caused false positives.
    if not bounced and not is_portrait_pov:
        w = CFG["front_bounce_window"]
        for i in range(len(smoothed_r) - w):
            if smoothed_r[i] < 1.0:
                continue
            ratio = smoothed_r[i + w] / (smoothed_r[i] + 0.01)
            if ratio >= CFG["front_bounce_size_jump"]:
                bounced  = True
                bounce_i = i + w // 2
                break

    bounce_pt = (track[bounce_i].x, track[bounce_i].y) if bounce_i is not None else None

    # ── Length classification ─────────────────────────────────────
    if bounce_i is not None:
        ref_y = track[bounce_i].y
    else:
        last_ys = ys[max(0, len(ys) * 2 // 3):]
        ref_y   = int(np.median(last_ys)) if last_ys else int(np.median(ys))

    y_frac = ref_y / max(frame_height, 1)
    length = _classify_length_front(y_frac)

    return bounced, bounce_pt, length


def _classify_length_front(y_frac: float) -> str:
    """
    Front view: ball appears lower in frame the closer it is to batsman.
      >= 0.58 → Yorker
      >= 0.42 → Full
      >= 0.26 → Good
         else → Short
    """
    if y_frac >= 0.58:
        return "Yorker"
    if y_frac >= 0.42:
        return "Full"
    if y_frac >= 0.26:
        return "Good"
    return "Short"