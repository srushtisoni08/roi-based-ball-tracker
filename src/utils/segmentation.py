import numpy as np
from src.models.track_point import TrackPoint


# ── Tunable constants ────────────────────────────────────────────────────────

# Maximum frame gap inside one delivery (frames)
MAX_INTRA_GAP = 6

# Minimum tracked points to consider a sequence a real delivery
MIN_POINTS = 5

# Minimum total displacement (px) across the sequence
MIN_DISPLACEMENT = 20          # was implicitly ~30 via x_std check

# Minimum x standard deviation to accept as "real lateral movement"
# (side view: ball moves across frame)
MIN_X_STD_SIDE = 8             # was 30 — way too strict for phone videos

# Minimum y range to accept as "real forward movement"  
# (front view: ball grows in size / moves down)
MIN_Y_RANGE_FRONT = 20

# Maximum frame gap between deliveries — larger gap = new delivery
INTER_DELIVERY_GAP = 45


def segment_deliveries(detections: list,
                       view: str = "side") -> list[list]:
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.frame)

    # ── Step 1: split by large frame gaps ────────────────────────
    raw_segments: list[list] = []
    current: list = [detections[0]]

    for det in detections[1:]:
        gap = det.frame - current[-1].frame
        if gap > MAX_INTRA_GAP:
            if len(current) >= MIN_POINTS:
                raw_segments.append(current)
            current = [det]
        else:
            current.append(det)

    if len(current) >= MIN_POINTS:
        raw_segments.append(current)

    # ── Step 2: direction-split within each raw segment ───────────
    # If the ball clearly reverses direction mid-segment, split there.
    direction_split: list[list] = []
    for seg in raw_segments:
        direction_split.extend(_split_on_direction_reversal(seg))

    # ── Step 3: filter noise segments ─────────────────────────────
    valid: list[list] = []
    for seg in direction_split:
        if len(seg) < MIN_POINTS:
            continue
        if not _has_real_movement(seg, view):
            continue
        valid.append(seg)

    print(f"[INFO] Segmented into {len(valid)} valid deliveries "
          f"(from {len(raw_segments)} raw segments, "
          f"{len(direction_split)} after direction-split, "
          f"{len(valid)} after length filter)")

    return valid


# ── Private helpers ──────────────────────────────────────────────────────────

def _split_on_direction_reversal(seg: list) -> list[list]:
    """
    Split a segment if the ball clearly reverses its primary direction
    (e.g. two deliveries merged because camera didn't move between them).
    Only splits when there is a very strong reversal (>60 px) to avoid
    splitting a single delivery that has noisy detections.
    """
    if len(seg) < 10:
        return [seg]

    xs = [p.x for p in seg]
    # Smooth with a small window
    window = min(5, len(xs) // 3)
    if window < 2:
        return [seg]

    smoothed = np.convolve(xs, np.ones(window) / window, mode='valid')
    directions = np.sign(np.diff(smoothed))

    splits = [0]
    run_len = 1
    for i in range(1, len(directions)):
        if directions[i] != 0 and directions[i] != directions[i - 1]:
            run_len = 1
        else:
            run_len += 1
        # Only split after a sustained reversal of at least 4 frames
        if run_len == 1 and i > 8:
            # check magnitude of reversal
            pre_x  = np.mean(xs[max(0, i - 4):i])
            post_x = np.mean(xs[i:min(len(xs), i + 4)])
            if abs(post_x - pre_x) > 80:
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

    # Total Euclidean displacement (start → end)
    disp = float(((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2) ** 0.5)

    if view == "front":
        # Front view: ball moves toward camera — y grows, size grows
        # Accept if ANY of these show real movement
        ok = (y_range >= MIN_Y_RANGE_FRONT or
              y_std >= 15 or
              disp >= MIN_DISPLACEMENT)
        if not ok:
            print(f"[SKIP] Front-view segment rejected: "
                  f"y_range={y_range:.1f}px, y_std={y_std:.1f}px, disp={disp:.1f}px")
        return ok
    else:
        # Side view: ball moves across frame — check x movement
        ok = (x_std >= MIN_X_STD_SIDE or
              x_range >= MIN_DISPLACEMENT or
              disp >= MIN_DISPLACEMENT)
        if not ok:
            print(f"[SKIP] Side-view segment rejected: "
                  f"x_std={x_std:.1f}px, x_range={x_range:.1f}px, disp={disp:.1f}px")
        return ok