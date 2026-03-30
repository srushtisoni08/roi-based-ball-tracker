import cv2
import json
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
 
    detector       = BallDetector(height, width, quality)
    all_detections: list[TrackPoint] = []
    trail          = deque(maxlen=50)
    completed      : list[DeliveryResult] = []
    bounce_markers : dict[int, tuple] = {}   # ball_no → (x,y)
    banner_until   : dict[int, int]   = {}   # ball_no → last frame to show banner
 
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
 
        det = detector.detect(frame)
        if det:
            det.frame = frame_no
            all_detections.append(det)
            trail.append((det.x, det.y))
 
        # ── Draw ──────────────────────────────────────────────────
        vis = frame.copy()
 
        if view == "side":
            draw_pitch_zones_side(vis)
 
        draw_ball_trail(vis, trail)
 
        if det:
            cv2.circle(vis, (det.x, det.y), int(det.radius) + 3,
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
    deliveries = segment_deliveries(all_detections)
    results    = []
 
    print("=" * 58)
    print(f"  RESULTS  —  {len(deliveries)} deliveries detected  |  view: {view}")
    print("=" * 58)
 
    for i, delivery in enumerate(deliveries, 1):
        if view == "side":
            bounced, bounce_pt, length = analyse_side(delivery, fps, width)
        else:
            bounced, bounce_pt, length = analyse_front(delivery, fps, height)
 
        start_f   = delivery[0].frame
        end_f     = delivery[-1].frame
        dur       = (end_f - start_f) / fps
 
        res = DeliveryResult(
            ball_no=i, bounced=bounced, length=length,
            bounce_frame=delivery[delivery.index(
                min(delivery, key=lambda p: abs(p.frame - (
                    (bounce_pt[0] if bounce_pt else 0)
                )))
            )].frame if bounce_pt else None,
            bounce_point=bounce_pt,
            start_frame=start_f, end_frame=end_f,
            duration_s=round(dur, 2),
            tracked_points=len(delivery),
        )
        results.append(res)
 
        b_str = "BOUNCED  ↓" if bounced else "FULL TOSS →"
        l_col = {"Yorker": "\033[92m", "Full": "\033[94m",
                  "Good": "\033[96m", "Short": "\033[91m"}.get(length, "")
        reset = "\033[0m"
        print(f"  Ball {i:>2}:  {b_str:<14}  Length: {l_col}{length:<8}{reset}"
              f"  ({len(delivery)} pts, {dur:.2f}s)")
 
    print("=" * 58)
    bounced_count   = sum(1 for r in results if r.bounced)
    no_bounce_count = sum(1 for r in results if r.bounced is False)
    print(f"  Pitched deliveries : {bounced_count}")
    print(f"  Full tosses        : {no_bounce_count}")
    print(f"  Length breakdown   : " + "  ".join(
        f"{k}={sum(1 for r in results if r.length == k)}" for k in ["Yorker","Full","Good","Short"]
    ))
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
        "length_summary": {k: sum(1 for r in results if r.length == k)
                           for k in ["Yorker","Full","Good","Short"]},
        "deliveries": [
            {
                "ball": r.ball_no,
                "bounced": r.bounced,
                "length": r.length,
                "bounce_point": r.bounce_point,
                "start_frame": r.start_frame,
                "end_frame": r.end_frame,
                "duration_s": r.duration_s,
                "tracked_points": r.tracked_points,
            }
            for r in results
        ],
    }
 
    json_out = video_path.rsplit(".", 1)[0] + "_report.json"
    with open(json_out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[INFO] Report saved : {json_out}")
    if output_path:
        print(f"[INFO] Video saved  : {output_path}")
 
    return report
 