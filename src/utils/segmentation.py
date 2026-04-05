import numpy as np
from src.models.track_point import TrackPoint
from src.config import CFG


# ── Tunable constants ────────────────────────────────────────────────────────

# Within a single delivery: max frames the ball can go undetected.
# ~0.2s at 30fps, ~0.4s at 60fps (scaled at runtime).
MAX_INTRA_GAP_S  = 0.20   # seconds

# Two raw segments are the SAME delivery only if the gap is short AND
# the ball position is spatially continuous across the gap.
SAME_DELIVERY_GAP_S  = 0.40   # seconds — maximum gap to even consider merging
SAME_DELIVERY_JUMP_PX = 80    # max pixel jump between end of seg1 and start of seg2

# A single delivery lasts at most this long (full toss + pickup time).
MAX_DELIVERY_S = 3.0   # seconds

# Minimum detections to be a valid delivery.
MIN_POINTS = 12

MIN_DISPLACEMENT = CFG.get("min_total_displacement_px", 80)
MIN_X_STD_SIDE = CFG.get("min_x_spread_px", 30)
MIN_Y_RANGE_FRONT = max(30, int(CFG.get("min_total_displacement_px", 80) * 0.5))


def segment_deliveries(detections: list,
                       view: str = "side",
                       fps: float = 30.0) -> list[list]:
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d.frame)

    # Convert time-based thresholds to frames
    intra_gap  = max(3, int(MAX_INTRA_GAP_S  * fps))
    same_gap   = max(5, int(SAME_DELIVERY_GAP_S * fps))
    max_dur    = max(30, int(MAX_DELIVERY_S * fps))

    # Adaptive minimum detections
    total_frames  = detections[-1].frame - detections[0].frame + 1
    adaptive_min  = max(8, min(MIN_POINTS, total_frames // 30))

    # ── Step 1: split by frame gaps larger than intra_gap ─────────
    raw_segments: list[list] = []
    current: list = [detections[0]]

    for det in detections[1:]:
        gap = det.frame - current[-1].frame
        if gap > intra_gap:
            if len(current) >= adaptive_min:
                raw_segments.append(current)
            current = [det]
        else:
            current.append(det)

    if len(current) >= adaptive_min:
        raw_segments.append(current)

    # ── Step 2: conservative merge — only stitch if spatially continuous ──
    # Two segments belong to the same delivery ONLY IF:
    #   (a) the frame gap is short (≤ same_gap), AND
    #   (b) the ball position is spatially close (≤ SAME_DELIVERY_JUMP_PX)
    # This prevents swallowing the pause between two separate deliveries.
    merged_segments: list[list] = []
    if raw_segments:
        merged_segments.append(raw_segments[0])
        for seg in raw_segments[1:]:
            prev = merged_segments[-1]
            frame_gap = seg[0].frame - prev[-1].frame

            if frame_gap <= same_gap:
                # Check spatial continuity
                p1 = prev[-1]
                p2 = seg[0]
                spatial_jump = ((p1.x - p2.x)**2 + (p1.y - p2.y)**2) ** 0.5
                if spatial_jump <= SAME_DELIVERY_JUMP_PX:
                    # Same ball — stitch together
                    merged_segments[-1] = prev + seg
                    continue

            merged_segments.append(seg)

    # ── Step 3: duration-split — chop anything longer than one delivery ──
    # This is the primary guard against multi-delivery merged tracks.
    duration_split: list[list] = []
    for seg in merged_segments:
        duration_split.extend(_split_by_duration(seg, max_frames=max_dur, min_pts=adaptive_min))

    # ── Step 4: direction-split on very large reversals only ──────
    direction_split: list[list] = []
    for seg in duration_split:
        direction_split.extend(_split_on_direction_reversal(seg, view=view))

    # ── Step 5: filter noise segments ─────────────────────────────
    valid: list[list] = []
    for seg in direction_split:
        if len(seg) < adaptive_min:
            continue
        if not _has_real_movement(seg, view):
            continue
        valid.append(seg)

    print(f"[INFO] Segmented into {len(valid)} valid deliveries "
          f"(from {len(raw_segments)} raw → {len(merged_segments)} merged → "
          f"{len(duration_split)} after dur-split → "
          f"{len(direction_split)} after dir-split → "
          f"{len(valid)} after movement filter)")

    return valid


# ── Private helpers ──────────────────────────────────────────────────────────

def _split_by_duration(seg: list, max_frames: int, min_pts: int) -> list[list]:
    """
    Split a segment whose frame span exceeds max_frames.
    Splits at natural gaps (largest gap first), then by midpoint if needed.
    """
    frame_span = seg[-1].frame - seg[0].frame
    if frame_span <= max_frames:
        return [seg]

    # Find the largest internal gap — best natural split point
    best_gap  = -1
    best_idx  = len(seg) // 2
    for i in range(1, len(seg)):
        g = seg[i].frame - seg[i - 1].frame
        if g > best_gap:
            best_gap = g
            best_idx = i

    left  = seg[:best_idx]
    right = seg[best_idx:]

    result = []
    if len(left) >= min_pts:
        result.extend(_split_by_duration(left,  max_frames, min_pts))
    if len(right) >= min_pts:
        result.extend(_split_by_duration(right, max_frames, min_pts))
    return result if result else [seg]


def _split_on_direction_reversal(seg: list, view: str = "side") -> list[list]:
    """
    Split only on very large direction reversals (> 200px).
    Bounce arcs are much smaller and must NOT be split.
    """
    if len(seg) < 20:
        return [seg]

    vals = [p.y for p in seg] if view == "front" else [p.x for p in seg]

    window = min(7, len(vals) // 4)
    if window < 2:
        return [seg]

    smoothed   = np.convolve(vals, np.ones(window) / window, mode='valid')
    directions = np.sign(np.diff(smoothed))

    splits  = [0]
    run_len = 1
    for i in range(1, len(directions)):
        if directions[i] != 0 and directions[i] != directions[i - 1]:
            run_len = 1
        else:
            run_len += 1
        if run_len == 1 and i > 20:
            pre_val  = float(np.mean(vals[max(0, i - 5):i]))
            post_val = float(np.mean(vals[i:min(len(vals), i + 5)]))
            if abs(post_val - pre_val) > 200:
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
    disp    = float(((xs[-1] - xs[0])**2 + (ys[-1] - ys[0])**2) ** 0.5)

    if view == "front":
        # In a true end-on (front) view the ball travels TOWARD the camera,
        # so it barely moves in XY pixel space — it mainly grows in radius
        # and drifts slightly in X (swing/spin).  The old threshold
        # (y_range >= 40px OR disp >= 80px) incorrectly rejects these.
        #
        # Relaxed criteria for front view:
        #   • Any meaningful lateral drift (x_range >= 15px) — swing/angle
        #   • OR radius growth — check via the Detection.radius attribute
        #     if available (all_detections uses Detection objects, not pure
        #     TrackPoints, so .radius is always set).
        #   • OR the original Y/disp thresholds for non-end-on setups.
        rs = np.array([getattr(p, "radius", 0.0) for p in seg])
        r_growth = float(np.max(rs) - np.min(rs)) if len(rs) > 1 else 0.0
        r_max    = float(np.max(rs)) if len(rs) > 0 else 0.0

        ok = (
            y_range >= MIN_Y_RANGE_FRONT        # camera tilted, ball moves in Y
            or y_std >= MIN_Y_RANGE_FRONT / 2
            or disp >= MIN_DISPLACEMENT
            or x_range >= 15                     # lateral drift (swing/angle)
            or r_growth >= 2.0                   # ball getting bigger (toward camera)
            or r_max >= 8.0                      # large ball = close = valid detection
        )
        if not ok:
            print(f"[SKIP] Front-view segment rejected: "
                  f"y_range={y_range:.1f}px, y_std={y_std:.1f}px, disp={disp:.1f}px, "
                  f"x_range={x_range:.1f}px, r_growth={r_growth:.1f}px")
        return ok
    else:
        ok = (x_std >= MIN_X_STD_SIDE or x_range >= MIN_DISPLACEMENT or disp >= MIN_DISPLACEMENT)
        if not ok:
            print(f"[SKIP] Side-view segment rejected: "
                  f"x_std={x_std:.1f}px, x_range={x_range:.1f}px, disp={disp:.1f}px")
        return ok