import math
import numpy as np
from src.models.track_point import TrackPoint
from src.config import CFG


def _distance(p1: TrackPoint, p2: TrackPoint) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def _filter_noisy_detections(detections: list[TrackPoint]) -> list[TrackPoint]:
    """
    Two-pass noise filter.

    Pass 1 — jump filter:
        Remove detections that jump too far from the previous accepted point,
        scaled by frame gap. A real cricket ball cannot teleport.

    Pass 2 — median spike filter:
        After pass 1, any remaining point that deviates greatly from its
        local median neighbourhood in X or Y is a stationary false positive
        (e.g. a stumps highlight, white line on pitch) that slipped through
        because it happened to land near the previous ball position.
    """
    if not detections:
        return []

    max_jump = CFG["max_interframe_jump_px"]

    # ── Pass 1: trajectory continuity ─────────────────────────────
    filtered = [detections[0]]
    for i in range(1, len(detections)):
        curr      = detections[i]
        prev      = filtered[-1]
        frame_gap = max(1, curr.frame - prev.frame)
        allowed   = max_jump * min(frame_gap, 3)   # cap so it doesn't become infinite

        if _distance(curr, prev) <= allowed:
            filtered.append(curr)

    if len(filtered) < 3:
        return filtered

    # ── Pass 2: median spike removal ──────────────────────────────
    xs = np.array([p.x for p in filtered], dtype=float)
    ys = np.array([p.y for p in filtered], dtype=float)

    window = 5
    half   = window // 2
    keep   = []
    for i, p in enumerate(filtered):
        lo    = max(0, i - half)
        hi    = min(len(filtered), i + half + 1)
        med_x = np.median(xs[lo:hi])
        med_y = np.median(ys[lo:hi])
        # Spike = deviates > 30px from local median in either axis
        if abs(p.x - med_x) > 30 or abs(p.y - med_y) > 30:
            continue
        keep.append(p)

    print(f"[INFO] Noise filter: {len(keep)}/{len(detections)} detections kept")
    return keep


def segment_deliveries(all_detections: list[TrackPoint]) -> list[list[TrackPoint]]:
    if not all_detections:
        return []

    clean = _filter_noisy_detections(all_detections)
    if not clean:
        return []

    deliveries: list[list[TrackPoint]] = []
    current: list[TrackPoint] = [clean[0]]

    for i in range(1, len(clean)):
        gap = clean[i].frame - clean[i - 1].frame
        if gap > CFG["delivery_gap_frames"]:
            if len(current) >= CFG["min_track_frames"]:
                deliveries.append(current)
            else:
                print(f"[SKIP] Segment dropped — only {len(current)} pts "
                      f"(min={CFG['min_track_frames']})")
            current = []
        current.append(clean[i])

    # Final segment
    if len(current) >= CFG["min_track_frames"]:
        deliveries.append(current)
    else:
        print(f"[SKIP] Last segment dropped — only {len(current)} pts")

    print(f"[INFO] Segmented into {len(deliveries)} deliveries")
    return deliveries