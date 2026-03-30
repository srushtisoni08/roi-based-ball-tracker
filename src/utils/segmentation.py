from models.track_point import TrackPoint
from config import CFG

def segment_deliveries(all_detections: list[TrackPoint]) -> list[list[TrackPoint]]:
    if not all_detections:
        return []
    deliveries, current = [], [all_detections[0]]
    for i in range(1, len(all_detections)):
        gap = all_detections[i].frame - all_detections[i - 1].frame
        if gap > CFG["delivery_gap_frames"]:
            if len(current) >= CFG["min_track_frames"]:
                deliveries.append(current)
            current = []
        current.append(all_detections[i])
    if len(current) >= CFG["min_track_frames"]:
        deliveries.append(current)
    return deliveries