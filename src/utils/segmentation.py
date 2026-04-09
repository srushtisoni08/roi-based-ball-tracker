import math
import numpy as np
from src.models.track_point import TrackPoint
from src.config import CFG


def _distance(p1: TrackPoint, p2: TrackPoint) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)


def _remove_stationary(detections: list[TrackPoint]) -> list[TrackPoint]:
    """
    Remove detections that sit in the same pixel location for too many
    consecutive frames. These are pitch markings, logos, or white lines —
    not the ball. A real ball always moves.

    Uses config: stationary_frame_threshold and stationary_pixel_radius.
    """
    threshold = CFG.get("stationary_frame_threshold", 6)
    radius    = CFG.get("stationary_pixel_radius", 10)

    if len(detections) < threshold:
        return detections

    keep = []
    for i, pt in enumerate(detections):
        # Count how many nearby frames have a detection within radius pixels
        lo = max(0, i - threshold)
        hi = min(len(detections), i + threshold + 1)
        same_pos = sum(
            1 for j in range(lo, hi)
            if j != i and
            abs(detections[j].x - pt.x) <= radius and
            abs(detections[j].y - pt.y) <= radius
        )
        # If same position appears in >= threshold surrounding frames = stationary
        if same_pos < threshold:
            keep.append(pt)

    removed = len(detections) - len(keep)
    if removed > 0:
        print(f"[INFO] Stationarity filter: removed {removed} stationary detections")
    return keep if keep else detections


def _filter_noisy_detections(detections: list[TrackPoint]) -> list[TrackPoint]:
    """
    Three-pass filter:
      Pass 0 — stationarity: remove fixed-position blobs (logos, markings)
      Pass 1 — jump filter: remove teleporting detections
      Pass 2 — median spike: remove isolated outliers
    """
    if not detections:
        return []

    # Pass 0: stationarity
    detections = _remove_stationary(detections)
    if not detections:
        return []

    max_jump  = CFG["max_interframe_jump_px"]
    spike_tol = CFG.get("spike_tolerance_px", 55)

    # Pass 1: trajectory continuity
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

    # Pass 2: median spike removal
    xs = np.array([p.x for p in filtered], dtype=float)
    ys = np.array([p.y for p in filtered], dtype=float)

    window = 7
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
    return keep if keep else filtered


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