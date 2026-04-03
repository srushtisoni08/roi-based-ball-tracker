import cv2
import json
import os
import numpy as np
from collections import deque
from src.config import CFG
from src.models.track_point import TrackPoint
from src.models.delivery_result import DeliveryResult
from src.detection.ball_detector import BallDetector
from src.analysis.side_view import analyse_side
from src.analysis.front_view import analyse_front
from src.utils.camera import detect_camera_quality, detect_view
from src.utils.segmentation import segment_deliveries
from src.visualization.draw import (
    draw_hud,
    draw_ball_trail,
    draw_bounce_marker,
    draw_length_banner,
    draw_pitch_zones_side
)


def process_video(video_path, view="auto", output_path=None, show=False, debug=False):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open: {video_path}")
        return

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"[INFO] {width}x{height} @ {fps:.1f}fps  ({total} frames)")

    quality = detect_camera_quality(cap)
    if view == "auto":
        view = detect_view(cap, height, width)
    print(f"[INFO] View mode: {view}\n")

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    detector        = BallDetector(height, width, quality)
    all_detections  : list[TrackPoint] = []
    trail           = deque(maxlen=50)
    completed       : list[DeliveryResult] = []
    bounce_markers  : dict[int, tuple] = {}
    banner_until    : dict[int, int]   = {}

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        det = detector.detect(frame)
        if det:
            det.frame = frame_no
            all_detections.append(det)
            trail.append((int(det.x), int(det.y)))

        # ── Draw ──────────────────────────────────────────────────
        vis = frame.copy()

        if view == "side":
            draw_pitch_zones_side(vis)

        draw_ball_trail(vis, trail)

        if det:
            cv2.circle(vis, (int(det.x), int(det.y)), max(1, int(det.radius)) + 3,
                       CFG["color_ball"], 2, cv2.LINE_AA)

        for ball_no, pt in bounce_markers.items():
            draw_bounce_marker(vis, pt, f"Ball {ball_no}")

        draw_hud(vis, completed, len(completed) + 1)

        for res in completed:
            draw_length_banner(vis, res, frame_no, banner_until)

        if debug:
            cv2.putText(vis, f"frame:{frame_no}  dets:{len(all_detections)}  view:{view}",
                        (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.42, (160, 160, 160), 1, cv2.LINE_AA)

        if writer:
            writer.write(vis)
        if show:
            cv2.imshow("Cricket Analyzer", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_no += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    # ── Post-process all deliveries ──────────────────────────────
    deliveries = segment_deliveries(all_detections, view=view)
    results    = []

    print("=" * 58)
    print(f"  RESULTS  —  {len(deliveries)} deliveries detected  |  view: {view}")
    print("=" * 58)

    for i, delivery in enumerate(deliveries, 1):
        if view == "side":
            bounced, bounce_pt, length = analyse_side(delivery, fps, width)
        else:
            bounced, bounce_pt, length = analyse_front(delivery, fps, height)

        start_f = delivery[0].frame
        end_f   = delivery[-1].frame
        dur     = (end_f - start_f) / fps

        bounce_frame = None
        if bounce_pt and bounced:
            ref = bounce_pt[0] if view == "side" else bounce_pt[1]
            axis_vals = [p.x if view == "side" else p.y for p in delivery]
            closest_idx = int(np.argmin([abs(v - ref) for v in axis_vals]))
            bounce_frame = delivery[closest_idx].frame

        res = DeliveryResult(
            ball_no=i,
            bounced=bounced,
            length=length,
            bounce_frame=bounce_frame,
            bounce_point=bounce_pt,
            start_frame=start_f,
            end_frame=end_f,
            duration_s=round(dur, 2),
            tracked_points=len(delivery),
        )
        results.append(res)

        if bounce_pt:
            bounce_markers[i] = bounce_pt

        b_str = "BOUNCED  ↓" if bounced else "FULL TOSS →"
        print(f"  Ball {i:>2}:  {b_str:<14}  ({len(delivery)} pts, {dur:.2f}s)")

    print("=" * 58)
    bounced_count   = sum(1 for r in results if r.bounced)
    no_bounce_count = sum(1 for r in results if r.bounced is False)
    print(f"  Pitched deliveries : {bounced_count}")
    print(f"  Full tosses        : {no_bounce_count}")
    print("=" * 58)

    # ── Save JSON ────────────────────────────────────────────────
    report = {
        "video": video_path,
        "view": view,
        "camera_quality": quality,
        "fps": fps,
        "total_deliveries": len(results),
        "pitched": bounced_count,
        "full_toss": no_bounce_count,
        "deliveries": [
            {
                "ball": r.ball_no,
                "bounced": r.bounced,
                "bounce_point": r.bounce_point,
                "bounce_frame": r.bounce_frame,
                "start_frame": r.start_frame,
                "end_frame": r.end_frame,
                "duration_s": r.duration_s,
                "tracked_points": r.tracked_points,
            }
            for r in results
        ],
    }

    os.makedirs("data/reports", exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    json_out   = os.path.join("data", "reports", f"{video_name}_report.json")
    with open(json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Report saved : {json_out}")
    if output_path:
        print(f"[INFO] Video saved  : {output_path}")

    return report