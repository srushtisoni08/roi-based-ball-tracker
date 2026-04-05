import math
from src.models.track_point import TrackPoint
from src.config import CFG

def _distance(p1: TrackPoint, p2: TrackPoint) -> float:
    return math.hypot(p1.x - p2.x, p1.y - p2.y)

def _filter_noisy_detections(detections: list[TrackPoint]) -> list[TrackPoint]:
    """
    Remove detections that jump too far from the previous point.
    A real ball moves smoothly — large jumps = false positive noise.
    """
    if not detections:
        return []

    max_jump = CFG["max_interframe_jump_px"]
    filtered = [detections[0]]

    for i in range(1, len(detections)):
        curr = detections[i]
        prev = filtered[-1]
        frame_gap = curr.frame - prev.frame

        # Allow proportionally larger jump if frames were skipped
        allowed_jump = max_jump * max(1, frame_gap * 0.5)

        if _distance(curr, prev) <= allowed_jump:
            filtered.append(curr)
        # else: silently discard — it's noise

    return filtered

def segment_deliveries(all_detections: list[TrackPoint]) -> list[list[TrackPoint]]:
    if not all_detections:
        return []

    # Step 1: Remove noisy jump detections
    clean = _filter_noisy_detections(all_detections)
    print(f"[INFO] After noise filter: {len(clean)}/{len(all_detections)} detections kept")

    if not clean:
        return []

    # Step 2: Split into deliveries by frame gap
    deliveries, current = [], [clean[0]]

    for i in range(1, len(clean)):
        gap = clean[i].frame - clean[i - 1].frame

        if gap > CFG["delivery_gap_frames"]:
            if len(current) >= CFG["min_track_frames"]:
                deliveries.append(current)
            else:
                print(f"[SKIP] Segment discarded — only {len(current)} points "
                      f"(min={CFG['min_track_frames']})")
            current = []

        current.append(clean[i])

    # Last segment
    if len(current) >= CFG["min_track_frames"]:
        deliveries.append(current)
    else:
        print(f"[SKIP] Last segment discarded — only {len(current)} points")

    print(f"[INFO] Segmented into {len(deliveries)} deliveries")
    return deliveries