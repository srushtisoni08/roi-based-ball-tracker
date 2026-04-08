import cv2
import json
import os
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
    draw_ball_trail,
    draw_bounce_marker,
    draw_result_badge,
    draw_hud,
    draw_length_banner,
    draw_pitch_zones_side,
)

BADGE_DURATION_FRAMES = 90


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
    all_detections : list[TrackPoint] = []
    trail          = deque(maxlen=80)   # longer trail = more path visible

    # Pass 1: detect only (no annotation yet — we annotate in pass 3 with correct results)
    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        det = detector.detect(frame)
        if det:
            det.frame = frame_no
            all_detections.append(det)
        frame_no += 1

    cap.release()
    if writer:
        writer.release()

    # ── Pass 2: segment & analyse ─────────────────────────────────
    deliveries = segment_deliveries(all_detections)
    results: list[DeliveryResult] = []

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
        if bounce_pt is not None:
            bounce_frame = min(
                delivery,
                key=lambda p: abs(p.x - bounce_pt[0]) + abs(p.y - bounce_pt[1])
            ).frame

        res = DeliveryResult(
            ball_no        = i,
            bounced        = bounced,
            length         = length,
            bounce_frame   = bounce_frame,
            bounce_point   = bounce_pt,
            start_frame    = start_f,
            end_frame      = end_f,
            duration_s     = round(dur, 2),
            tracked_points = len(delivery),
            track          = delivery,
        )
        results.append(res)

        b_str = "BOUNCED  ↓" if bounced else "FULL TOSS →"
        print(f"  Ball {i:>2}:  {b_str:<14}  ({len(delivery)} pts, {dur:.2f}s)")

    print("=" * 58)
    bounced_count   = sum(1 for r in results if r.bounced)
    no_bounce_count = sum(1 for r in results if not r.bounced)
    print(f"  Pitched deliveries : {bounced_count}")
    print(f"  Full tosses        : {no_bounce_count}")
    print("=" * 58)

    # ── Pass 3: render annotated video ────────────────────────────
    if output_path and results:
        _render_annotated_video(
            video_path, output_path, results, view, fps, width, height, debug
        )

    # ── Save JSON ─────────────────────────────────────────────────
    report = {
        "video"           : video_path,
        "view"            : view,
        "camera_quality"  : quality,
        "fps"             : fps,
        "total_deliveries": len(results),
        "pitched"         : bounced_count,
        "full_toss"       : no_bounce_count,
        "deliveries": [
            {
                "ball"          : r.ball_no,
                "bounced"       : r.bounced,
                "bounce_point"  : r.bounce_point,
                "start_frame"   : r.start_frame,
                "end_frame"     : r.end_frame,
                "duration_s"    : r.duration_s,
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


def _render_annotated_video(video_path, output_path, results, view,
                            fps, width, height, debug):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fourcc   = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_path = output_path + ".tmp.mp4"
    writer   = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    # Build bounce active-window: frame → [(ball_no, bounce_pt)]
    active_bounces: dict[int, list] = {}
    for res in results:
        if res.bounce_point is not None and res.bounce_frame is not None:
            for f in range(res.bounce_frame, res.end_frame + 90 + 1):
                active_bounces.setdefault(f, []).append((res.ball_no, res.bounce_point))

    # Badge show window: ball_no → last frame to show badge
    badge_until: dict[int, int] = {
        r.ball_no: r.end_frame + BADGE_DURATION_FRAMES for r in results
    }

    # All track points sorted by frame for trail replay
    all_track_pts: list[tuple] = []
    for res in results:
        for pt in res.track:
            all_track_pts.append((pt.frame, pt.x, pt.y, res.ball_no))
    all_track_pts.sort(key=lambda t: t[0])

    # Per-delivery full path (draw entire path once delivery ends)
    delivery_paths: dict[int, list] = {
        res.ball_no: [(pt.x, pt.y) for pt in res.track]
        for res in results
    }

    trail     = deque(maxlen=80)
    trail_ptr = 0
    frame_no  = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Advance trail
        while trail_ptr < len(all_track_pts) and all_track_pts[trail_ptr][0] <= frame_no:
            _, tx, ty, _ = all_track_pts[trail_ptr]
            trail.append((tx, ty))
            trail_ptr += 1

        vis = frame.copy()

        # Draw completed delivery full paths (persistent ghost trail)
        for res in results:
            if res.end_frame < frame_no and res.bounce_point is not None:
                pts = delivery_paths[res.ball_no]
                for i in range(1, len(pts)):
                    alpha = i / len(pts)
                    c = (int(40 * alpha), int(80 * alpha), int(120 * alpha))
                    cv2.line(vis, pts[i-1], pts[i], c, 1, cv2.LINE_AA)

        # Live trail for current ball in flight
        draw_ball_trail(vis, trail)

        # Ball circle at current detected position
        matched = next((t for t in all_track_pts if t[0] == frame_no), None)
        if matched:
            _, tx, ty, _ = matched
            cv2.circle(vis, (tx, ty), 10, CFG["color_ball"], 2, cv2.LINE_AA)
            # Inner dot
            cv2.circle(vis, (tx, ty),  4, (255, 255, 255), -1, cv2.LINE_AA)

        # Bounce marker
        for ball_no, pt in active_bounces.get(frame_no, []):
            draw_bounce_marker(vis, pt)

        # Bounced / Full Toss badge (top-right, no length labels)
        completed_now = [r for r in results if r.end_frame < frame_no]
        for res in completed_now:
            draw_result_badge(vis, res, frame_no, badge_until)

        if debug:
            cv2.putText(vis, f"frame:{frame_no}  view:{view}",
                        (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                        (160, 160, 160), 1, cv2.LINE_AA)

        writer.write(vis)
        frame_no += 1

    cap.release()
    writer.release()

    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp_path, output_path)
    print(f"[INFO] Annotated video written: {output_path}")