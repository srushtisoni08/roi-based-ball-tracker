import numpy as np
from src.models.track_point import TrackPoint
from src.config import CFG


def analyse_side(delivery: list[TrackPoint],
                 fps: float,
                 frame_width: int) -> tuple[bool, tuple | None, str]:

    if len(delivery) < 3:
        return False, None, "Unknown"

    xs = np.array([p.x for p in delivery], dtype=float)
    ys = np.array([p.y for p in delivery], dtype=float)

    # ── Smooth Y trajectory ────────────────────────────────────────
    win = min(5, max(2, len(ys) // 4))
    ys_smooth = _moving_average(ys, win)
    xs_smooth = _moving_average(xs, win)

    # ── Bounce detection ───────────────────────────────────────────
    bounced, bounce_idx = _detect_bounce(ys_smooth)

    bounce_point = None
    if bounced and bounce_idx is not None:
        # Map smoothed index back to original array
        orig_idx = min(bounce_idx + win // 2, len(delivery) - 1)
        bx = float(xs[orig_idx])
        by = float(ys[orig_idx])
        bounce_point = (bx, by)

    # ── Length classification ──────────────────────────────────────
    length = _classify_length(xs_smooth, ys_smooth, bounce_point,
                               frame_width, bounced)

    return bounced, bounce_point, length


# ── Private helpers ──────────────────────────────────────────────────────────

def _moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    if window < 2 or len(arr) < window:
        return arr.copy()
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode='valid')


def _detect_bounce(ys: np.ndarray,
                   min_descent_frames: int = 2,
                   min_reversal_px: int = 8,
                   min_accel: float = 0.3) -> tuple[bool, int | None]:
    """
    Detect a Y-direction reversal that indicates a bounce.

    A genuine bounce shows:
    1. Ball descending (y increasing in image coords) for several frames
    2. Then ascending (y decreasing) after hitting the pitch
    3. The reversal is at least min_reversal_px large
    4. The acceleration (2nd derivative) at the reversal exceeds min_accel

    Parameters
    ----------
    ys                  : smoothed y-coordinate array
    min_descent_frames  : min frames of downward motion before bounce
    min_reversal_px     : min y change before+after reversal
    min_accel           : minimum |d²y/dt²| at reversal point
    """
    if len(ys) < 5:
        return False, None

    vy = np.diff(ys)         # velocity (dy per frame)
    ay = np.diff(vy)         # acceleration

    # Find local maxima in y (= lowest point = bounce candidate)
    # A local max means vy goes from positive to negative
    best_idx  = None
    best_score = 0.0

    for i in range(min_descent_frames, len(vy) - 1):
        # Velocity switches from positive (descending) to negative (ascending)
        if vy[i - 1] > 0 and vy[i] <= 0:
            # How much did it descend before?
            descent = ys[i] - ys[max(0, i - min_descent_frames)]
            # How much does it rise after?
            ascent_end = min(i + 6, len(ys) - 1)
            ascent = ys[i] - ys[ascent_end]

            if descent < min_reversal_px:
                continue
            if ascent < min_reversal_px * 0.3:   # at least some rise
                continue

            accel_mag = abs(ay[i - 1]) if i - 1 < len(ay) else 0.0
            score = descent + ascent + accel_mag * 5

            if score > best_score:
                best_score = score
                best_idx   = i

    if best_idx is None:
        return False, None

    return True, best_idx


def _classify_length(xs: np.ndarray,
                     ys: np.ndarray,
                     bounce_point: tuple | None,
                     frame_width: int,
                     bounced: bool) -> str:
    """
    Classify delivery length from bounce X position.

    Length zones (fraction of frame_width, measured from batsman end):
      Yorker : 0.70 – 1.00  (very close to batsman)
      Full   : 0.50 – 0.70
      Good   : 0.30 – 0.50
      Short  : 0.00 – 0.30  (pitched early / short-pitched)

    If no bounce was detected, classify by final X position as a proxy.
    """
    if not bounced or bounce_point is None:
        # Use final x position as a rough proxy
        ref_x = float(xs[-1])
    else:
        ref_x = bounce_point[0]

    # Normalise: x=0 is bowling end, x=frame_width is batsman end
    # But we don't know which end is which — use the direction of travel
    if len(xs) >= 2:
        direction = 1 if xs[-1] > xs[0] else -1
    else:
        direction = 1

    # If ball moves left→right, batsman is at right (high x)
    if direction > 0:
        norm_x = ref_x / frame_width
    else:
        norm_x = 1.0 - (ref_x / frame_width)

    # Try config zones first
    zones = CFG.get("length_zones_side", {})
    for zone_name, (lo, hi) in zones.items():
        # zones may be in pixels; convert to fraction if > 2
        if hi > 2:
            lo_f = lo / frame_width
            hi_f = hi / frame_width
        else:
            lo_f, hi_f = lo, hi
        if lo_f <= norm_x < hi_f:
            return _normalise_length_name(zone_name)

    # Fallback built-in zones (fractions)
    if norm_x >= 0.70:
        return "Yorker"
    if norm_x >= 0.50:
        return "Full"
    if norm_x >= 0.30:
        return "Good"
    return "Short"


def _normalise_length_name(raw: str) -> str:
    mapping = {
        "yorker": "Yorker", "full": "Full",
        "good": "Good", "short": "Short",
    }
    return mapping.get(raw.lower(), raw.capitalize())