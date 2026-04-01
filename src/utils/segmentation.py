import math
import numpy as np
from src.models.track_point import TrackPoint
from src.config import CFG


def _distance(p1: TrackPoint, p2: TrackPoint) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


# ──────────────────────────────────────────────────────────────────────────────
#  NEW: Movement validator
#  Kills the root cause of false deliveries: static blobs (batsman pad/foot,
#  stumps, net posts) that persist for many frames but never actually move.
# ──────────────────────────────────────────────────────────────────────────────
def _is_valid_delivery_track(segment: list[TrackPoint]) -> bool:
    """
    Returns True only if the segment looks like a real ball trajectory:

    1. Total displacement >= min_total_displacement_px
       A cricket ball travels 150–300 px across the frame minimum.
       A static pad/foot blob moves 0–15 px — always fails this.

    2. X-spread (std-dev) >= min_x_spread_px
       A real delivery moves consistently in one direction across X.
       Net mesh oscillations cluster at the same X position.

    3. Directional consistency: the track has a dominant direction.
       At least 60 % of consecutive pairs must share the same X-sign.
       Random blobs flicker left/right equally.
    """
    if len(segment) < 2:
        return False

    xs = [p.x for p in segment]
    ys = [p.y for p in segment]

    # ── Test 1: total displacement ─────────────────────────────────────
    total_disp = math.hypot(xs[-1] - xs[0], ys[-1] - ys[0])
    min_disp   = CFG.get("min_total_displacement_px", 80)
    if total_disp < min_disp:
        print(f"[SKIP] Static blob — displacement={total_disp:.1f}px < {min_disp}px")
        return False

    # ── Test 2: x-spread ───────────────────────────────────────────────
    x_spread  = float(np.std(xs))
    min_spread = CFG.get("min_x_spread_px", 30)
    if x_spread < min_spread:
        print(f"[SKIP] No lateral movement — x_std={x_spread:.1f}px < {min_spread}px")
        return False

    # ── Test 3: directional consistency ───────────────────────────────
    dx_signs = [np.sign(xs[i] - xs[i - 1]) for i in range(1, len(xs))
                if xs[i] != xs[i - 1]]
    if dx_signs:
        dominant = max(set(dx_signs), key=dx_signs.count)
        consistency = dx_signs.count(dominant) / len(dx_signs)
        if consistency < 0.55:          # less than 55 % consistent → noise
            print(f"[SKIP] Inconsistent direction — consistency={consistency:.2f}")
            return False

    return True


# ──────────────────────────────────────────────────────────────────────────────
def _filter_noisy_detections(detections: list[TrackPoint]) -> list[TrackPoint]:
    if not detections:
        return []

    max_jump       = CFG["max_interframe_jump_px"]
    max_interp_gap = CFG.get("max_interp_gap_frames", 4)

    filtered = [detections[0]]

    for i in range(1, len(detections)):
        curr      = detections[i]
        prev      = filtered[-1]
        frame_gap = curr.frame - prev.frame

        # Within a short gap allow proportional jump (ball briefly occluded)
        allowed_jump = max_jump * frame_gap if frame_gap <= max_interp_gap else max_jump

        if _distance(curr, prev) <= allowed_jump:
            filtered.append(curr)
        # else: silently discard positional teleport

    return filtered


# ──────────────────────────────────────────────────────────────────────────────
def _split_by_direction_reversal(segment: list[TrackPoint]) -> list[list[TrackPoint]]:
    """
    Splits a segment when the ball snaps back toward the bowler end,
    indicating a new delivery hidden inside one continuous track.
    """
    if len(segment) < 2:
        return [segment]

    reversal_px = CFG.get("direction_reversal_px", 250)
    min_travel  = CFG.get("min_travel_before_reversal_px", 150)

    splits   = [0]
    xs       = [p.x for p in segment]
    anchor_x = xs[0]

    for i in range(1, len(xs)):
        travel = xs[i] - anchor_x

        if abs(travel) >= min_travel:
            dx = xs[i] - xs[i - 1]
            if (travel > 0 and dx < -reversal_px) or (travel < 0 and dx > reversal_px):
                splits.append(i)
                anchor_x = xs[i]

    if len(splits) == 1:
        return [segment]

    parts = []
    for k in range(len(splits)):
        start = splits[k]
        end   = splits[k + 1] if k + 1 < len(splits) else len(segment)
        parts.append(segment[start:end])
    return parts


# ──────────────────────────────────────────────────────────────────────────────
def segment_deliveries(all_detections: list[TrackPoint]) -> list[list[TrackPoint]]:
    if not all_detections:
        return []

    # ── Step 1: Remove noisy single-frame jump detections ─────────────
    clean = _filter_noisy_detections(all_detections)
    print(f"[INFO] After noise filter: {len(clean)}/{len(all_detections)} detections kept")

    if not clean:
        return []

    # ── Step 2: Split into raw segments by frame gap ───────────────────
    raw_segments: list[list[TrackPoint]] = []
    current = [clean[0]]

    for i in range(1, len(clean)):
        gap = clean[i].frame - clean[i - 1].frame
        if gap > CFG["delivery_gap_frames"]:
            raw_segments.append(current)
            current = []
        current.append(clean[i])

    raw_segments.append(current)

    # ── Step 3: Direction-reversal split (catches merged deliveries) ───
    split_segments: list[list[TrackPoint]] = []
    for seg in raw_segments:
        split_segments.extend(_split_by_direction_reversal(seg))

    # ── Step 4: Minimum frame-count filter ────────────────────────────
    length_filtered: list[list[TrackPoint]] = []
    for seg in split_segments:
        if len(seg) >= CFG["min_track_frames"]:
            length_filtered.append(seg)
        else:
            print(f"[SKIP] Too short — {len(seg)} frames "
                  f"(min={CFG['min_track_frames']})")

    # ── Step 5: Movement / displacement filter (THE KEY FIX) ──────────
    # Eliminates static blobs that survived steps 1-4.
    deliveries: list[list[TrackPoint]] = []
    for seg in length_filtered:
        if _is_valid_delivery_track(seg):
            deliveries.append(seg)

    print(f"[INFO] Segmented into {len(deliveries)} valid deliveries "
          f"(from {len(raw_segments)} raw segments, "
          f"{len(split_segments)} after direction-split, "
          f"{len(length_filtered)} after length filter)")
    return deliveries


# ──────────────────────────────────────────────────────────────────────────────
def _smooth(values, window=3):
    half = window // 2
    return [np.mean(values[max(0, i - half): min(len(values), i + half + 1)])
            for i in range(len(values))]