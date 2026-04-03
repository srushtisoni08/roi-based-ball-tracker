import numpy as np
from src.models.track_point import TrackPoint


# ── Tunable constants ────────────────────────────────────────────────────────

MAX_INTRA_GAP = 4
MIN_POINTS = 20  # back to 20 — 25 is too strict for short videos
MIN_DISPLACEMENT = 20
MIN_X_STD_SIDE = 8
MIN_Y_RANGE_FRONT = 20
INTER_DELIVERY_GAP = 45


def segment_deliveries(detections: list,
                       view: str = "side") -> list[list]:
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.frame)

    # Adaptive MIN_POINTS — short videos need a lower bar
    total_frames = detections[-1].frame - detections[0].frame + 1
    adaptive_min = max(10, min(25, total_frames // 20))

    # ── Step 1: split by large frame gaps ────────────────────────
    raw_segments: list[list] = []
    current: list = [detections[0]]

    for det in detections[1:]:
        gap = det.frame - current[-1].frame
        if gap > MAX_INTRA_GAP:
            if len(current) >= adaptive_min:
                raw_segments.append(current)
            current = [det]
        else:
            current.append(det)

    if len(current) >= MIN_POINTS:
        raw_segments.append(current)

    # ── Step 2: direction-split within each raw segment ───────────
    direction_split: list[list] = []
    for seg in raw_segments:
        direction_split.extend(_split_on_direction_reversal(seg, view=view))

    # ── Step 2b: duration-split anything still too long ───────────
    duration_split: list[list] = []
    for seg in direction_split:
        duration_split.extend(_split_by_duration(seg, max_frames=90, min_pts=adaptive_min))

    direction_split = duration_split

    # ── Step 3: filter noise segments ─────────────────────────────
    valid: list[list] = []
    for seg in direction_split:
        if len(seg) < adaptive_min:
            continue
        if not _has_real_movement(seg, view):
            continue
        valid.append(seg)

    print(f"[INFO] Segmented into {len(valid)} valid deliveries "
          f"(from {len(raw_segments)} raw segments, "
          f"{len(duration_split)} after duration-split, "
          f"{len(valid)} after length filter)")

    return valid


# ── Private helpers ──────────────────────────────────────────────────────────

def _split_by_duration(seg: list, max_frames: int = 90, min_pts: int = 20) -> list[list]:
    """Split a segment that is too long to be a single delivery."""
    if len(seg) <= max_frames:
        return [seg]

    result = []
    current = [seg[0]]

    for i in range(1, len(seg)):
        frame_span = seg[i].frame - current[0].frame
        gap = seg[i].frame - seg[i - 1].frame

        # Split if we've exceeded max duration OR there's a gap in detections
        if frame_span > max_frames or gap > 6:
            if len(current) >= min_pts:
                result.append(current)
            current = [seg[i]]
        else:
            current.append(seg[i])

    if len(current) >= min_pts:
        result.append(current)

    return result if result else [seg]


def _split_on_direction_reversal(seg: list, view: str = "side") -> list[list]:
    if len(seg) < 10:
        return [seg]

    # Front view: ball moves on Y axis (toward camera)
    # Side view: ball moves on X axis (across frame)
    vals = [p.y for p in seg] if view == "front" else [p.x for p in seg]

    window = min(5, len(vals) // 3)
    if window < 2:
        return [seg]

    smoothed = np.convolve(vals, np.ones(window) / window, mode='valid')
    directions = np.sign(np.diff(smoothed))

    splits = [0]
    run_len = 1
    for i in range(1, len(directions)):
        if directions[i] != 0 and directions[i] != directions[i - 1]:
            run_len = 1
        else:
            run_len += 1
        if run_len == 1 and i > 12:
            pre_val  = float(np.mean(vals[max(0, i - 4):i]))
            post_val = float(np.mean(vals[i:min(len(vals), i + 4)]))
            if abs(post_val - pre_val) > 120:
                splits.append(i + window // 2)

    splits.append(len(seg))
    result = []
    for a, b in zip(splits, splits[1:]):
        chunk = seg[a:b]
        if len(chunk) >= MIN_POINTS:
            result.append(chunk)
    return result if result else [seg]


def _has_real_movement(seg: list, view: str) -> bool:
    xs = np.array([p.x for p in seg])
    ys = np.array([p.y for p in seg])

    x_std   = float(np.std(xs))
    y_std   = float(np.std(ys))
    y_range = float(np.max(ys) - np.min(ys))
    x_range = float(np.max(xs) - np.min(xs))

    disp = float(((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2) ** 0.5)

    if view == "front":
        ok = (y_range >= MIN_Y_RANGE_FRONT or
              y_std >= 15 or
              disp >= MIN_DISPLACEMENT)
        if not ok:
            print(f"[SKIP] Front-view segment rejected: "
                  f"y_range={y_range:.1f}px, y_std={y_std:.1f}px, disp={disp:.1f}px")
        return ok
    else:
        ok = (x_std >= MIN_X_STD_SIDE or
              x_range >= MIN_DISPLACEMENT or
              disp >= MIN_DISPLACEMENT)
        if not ok:
            print(f"[SKIP] Side-view segment rejected: "
                  f"x_std={x_std:.1f}px, x_range={x_range:.1f}px, disp={disp:.1f}px")
        return ok