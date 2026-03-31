import math
import numpy as np
from src.models.track_point import TrackPoint
from src.config import CFG


def _distance(p1: TrackPoint, p2: TrackPoint) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def _filter_noisy_detections(detections: list[TrackPoint]) -> list[TrackPoint]:
    if not detections:
        return []

    max_jump        = CFG["max_interframe_jump_px"]          # 80 px
    max_interp_gap  = CFG.get("max_interp_gap_frames", 4)    # new: default 4

    filtered = [detections[0]]

    for i in range(1, len(detections)):
        curr       = detections[i]
        prev       = filtered[-1]
        frame_gap  = curr.frame - prev.frame

        if frame_gap <= max_interp_gap:
            # Small gap: allow proportional jump (ball was briefly occluded)
            allowed_jump = max_jump * frame_gap
        else:
            # Large gap: this is almost certainly a dead zone between
            # deliveries — use strict single-frame limit so random blobs
            # (player, umpire, stumps) cannot connect two separate deliveries.
            allowed_jump = max_jump

        if _distance(curr, prev) <= allowed_jump:
            filtered.append(curr)
        # else: silently discard noise

    return filtered


def _split_by_direction_reversal(segment: list[TrackPoint]) -> list[list[TrackPoint]]:
    """
    Splits a segment when the ball teleports back to the bowler end,
    indicating a new delivery has started inside what looks like one track.

    Uses cumulative x-travel direction: once the ball has traveled consistently
    in one direction for at least `min_travel_px`, a reversal larger than
    `direction_reversal_px` is treated as a new delivery start.
    Normal ball deceleration / slight oscillation won't exceed this threshold.
    """
    if len(segment) < 2:
        return [segment]

    reversal_px  = CFG.get("direction_reversal_px", 300)
    min_travel   = CFG.get("min_travel_before_reversal_px", 200)  # new key

    splits = [0]
    xs     = [p.x for p in segment]

    # Track the x anchor — the x value at the start of the current "run"
    anchor_x = xs[0]

    for i in range(1, len(xs)):
        travel = xs[i] - anchor_x

        # Only consider reversal after meaningful travel in one direction
        if abs(travel) >= min_travel:
            dx = xs[i] - xs[i - 1]
            # Reversal: moving opposite to established direction by reversal_px
            if (travel > 0 and dx < -reversal_px) or (travel < 0 and dx > reversal_px):
                splits.append(i)
                anchor_x = xs[i]   # reset anchor for next delivery

    if len(splits) == 1:
        return [segment]

    parts = []
    for k in range(len(splits)):
        start = splits[k]
        end   = splits[k + 1] if k + 1 < len(splits) else len(segment)
        parts.append(segment[start:end])
    return parts


def segment_deliveries(all_detections: list[TrackPoint]) -> list[list[TrackPoint]]:
    if not all_detections:
        return []

    # ── Step 1: Remove noisy jump detections ──────────────────────────
    clean = _filter_noisy_detections(all_detections)
    print(f"[INFO] After noise filter: {len(clean)}/{len(all_detections)} detections kept")

    if not clean:
        return []

    # ── Step 2: Split into deliveries by frame gap ─────────────────────
    raw_segments: list[list[TrackPoint]] = []
    current = [clean[0]]

    for i in range(1, len(clean)):
        gap = clean[i].frame - clean[i - 1].frame

        if gap > CFG["delivery_gap_frames"]:
            raw_segments.append(current)
            current = []

        current.append(clean[i])

    raw_segments.append(current)   # always append last chunk

    # ── Step 3: Direction-reversal split (catches merged deliveries) ───
    split_segments: list[list[TrackPoint]] = []
    for seg in raw_segments:
        split_segments.extend(_split_by_direction_reversal(seg))

    # ── Step 4: Apply minimum-length filter ───────────────────────────
    deliveries: list[list[TrackPoint]] = []
    for seg in split_segments:
        if len(seg) >= CFG["min_track_frames"]:
            deliveries.append(seg)
        else:
            print(f"[SKIP] Segment discarded — only {len(seg)} points "
                  f"(min={CFG['min_track_frames']})")

    print(f"[INFO] Segmented into {len(deliveries)} deliveries")
    return deliveries


def _smooth(values, window=3):
    half = window // 2
    return [np.mean(values[max(0, i - half): min(len(values), i + half + 1)])
            for i in range(len(values))]