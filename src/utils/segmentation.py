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
        scaled by frame gap.

    Pass 2 — median spike filter:
        Remove points that deviate far from their local median neighbourhood.
        Uses spike_tolerance_px from config (was hardcoded 30px — too tight).
    """
    if not detections:
        return []

    max_jump      = CFG["max_interframe_jump_px"]
    spike_tol     = CFG.get("spike_tolerance_px", 55)

    # ── Pass 1: trajectory continuity ─────────────────────────────
    filtered = [detections[0]]
    for i in range(1, len(detections)):
        curr      = detections[i]
        prev      = filtered[-1]
        frame_gap = max(1, curr.frame - prev.frame)
        allowed   = max_jump * min(frame_gap, 4)

        if _distance(curr, prev) <= allowed:
            filtered.append(curr)

    if len(filtered) < 3:
        return filtered

    # ── Pass 2: median spike removal ──────────────────────────────
    xs = np.array([p.x for p in filtered], dtype=float)
    ys = np.array([p.y for p in filtered], dtype=float)

    window = 7       # wider window = smoother, less likely to flag real ball
    half   = window // 2
    keep   = []
    for i, p in enumerate(filtered):
        lo    = max(0, i - half)
        hi    = min(len(filtered), i + half + 1)
        med_x = np.median(xs[lo:hi])
        med_y = np.median(ys[lo:hi])
        if abs(p.x - med_x) > spike_tol or abs(p.y - med_y) > spike_tol:
            continue
        keep.append(p)

    print(f"[INFO] Noise filter: {len(keep)}/{len(detections)} detections kept")
    return keep if keep else filtered   # fallback: if filter kills everything, return pass-1 result


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

    if len(current) >= CFG["min_track_frames"]:
        deliveries.append(current)
    else:
        print(f"[SKIP] Last segment dropped — only {len(current)} pts")

    print(f"[INFO] Segmented into {len(deliveries)} deliveries")
    return deliveries