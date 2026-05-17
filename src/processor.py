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
    draw_ball_trail,
    draw_bounce_marker,
    draw_result_badge,
    draw_delivery_counter,
    draw_speed_label,
    draw_hud,
    draw_length_banner,
    draw_pitch_zones_side,
)

BADGE_DURATION_FRAMES = 90


def _interpolate_track(track: list[TrackPoint], max_gap: int = 6) -> list[TrackPoint]:
    """
    Fill small gaps in a delivery track with linearly interpolated points.
    This produces a continuous trail even when the ball is briefly occluded.
    max_gap: skip interpolation if frame gap is larger than this.
    """
    if len(track) < 2:
        return track

    filled = [track[0]]
    for i in range(1, len(track)):
        prev = track[i - 1]
        curr = track[i]
        gap  = curr.frame - prev.frame

        if 1 < gap <= max_gap:
            for step in range(1, gap):
                t  = step / gap
                ix = int(prev.x + t * (curr.x - prev.x))
                iy = int(prev.y + t * (curr.y - prev.y))
                ir = prev.radius + t * (curr.radius - prev.radius)
                interp = TrackPoint(frame=prev.frame + step, x=ix, y=iy, radius=ir)
                interp._coasted = True
                filled.append(interp)

        filled.append(curr)

    return filled


def _estimate_speed(track: list[TrackPoint], fps: float) -> float:
    """
    Rough speed estimate in km/h from the middle third of the track.
    Uses pixel-per-frame velocity with a heuristic scale factor.
    """
    if len(track) < 4:
        return 0.0

    lo = len(track) // 3
    hi = (2 * len(track)) // 3
    segment = track[lo:hi]
    if len(segment) < 2:
        return 0.0

    dists = []
    for i in range(1, len(segment)):
        dx = segment[i].x - segment[i-1].x
        dy = segment[i].y - segment[i-1].y
        df = max(1, segment[i].frame - segment[i-1].frame)
        dists.append(np.hypot(dx, dy) / df)

    px_per_frame = float(np.median(dists)) if dists else 0.0
    px_per_sec   = px_per_frame * fps
    # Heuristic: 1 px/s ≈ 0.15 km/h at typical cricket video resolution
    return round(px_per_sec * 0.15, 1)


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

    detector       = BallDetector(height, width, quality)
    all_detections : list[TrackPoint] = []

    # ── Pass 1: detect ────────────────────────────────────────────
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

    # ── Pass 2: segment & analyse ─────────────────────────────────
    deliveries = segment_deliveries(all_detections)

    # Interpolate each delivery track to fill small frame gaps
    max_gap = CFG.get("trail_max_interp_gap", 6)
    deliveries = [_interpolate_track(d, max_gap) for d in deliveries]

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

        speed = _estimate_speed(delivery, fps)

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
        res._speed_kmh = speed
        results.append(res)

        b_str   = "BOUNCED  ↓" if bounced else "FULL TOSS →"
        spd_str = f"  ~{speed:.0f} km/h" if speed > 0 else ""
        print(f"  Ball {i:>2}:  {b_str:<14}  ({len(delivery)} pts, {dur:.2f}s){spd_str}")

    print("=" * 58)
    bounced_count   = sum(1 for r in results if r.bounced)
    no_bounce_count = sum(1 for r in results if not r.bounced)
    print(f"  Pitched : {bounced_count}   Full tosses : {no_bounce_count}")
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
                "speed_kmh"     : getattr(r, "_speed_kmh", 0),
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

    # ── Pre-compute per-frame lookups ─────────────────────────────

    # Bounce marker: show from bounce_frame → end_frame+90
    active_bounces: dict[int, list] = {}
    for res in results:
        if res.bounce_point is not None and res.bounce_frame is not None:
            for f in range(res.bounce_frame, res.end_frame + 91):
                active_bounces.setdefault(f, []).append((res.ball_no, res.bounce_point))

    # Badge show-until map
    badge_until: dict[int, int] = {
        r.ball_no: r.end_frame + BADGE_DURATION_FRAMES for r in results
    }

    # All track points sorted for trail replay
    all_track_pts: list[tuple] = []
    for res in results:
        for pt in res.track:
            all_track_pts.append((pt.frame, pt.x, pt.y, res.ball_no))
    all_track_pts.sort(key=lambda t: t[0])

    # Fast frame → (x, y, ball_no) lookup for the ball circle
    frame_to_pt: dict[int, tuple] = {
        t[0]: (t[1], t[2], t[3]) for t in all_track_pts
    }

    # Ghost trail paths for ALL deliveries
    delivery_paths: dict[int, list] = {
        res.ball_no: [(pt.x, pt.y) for pt in res.track]
        for res in results
    }

    # Speed labels: show near bounce point for 45 frames
    speed_labels: dict[int, tuple] = {}
    for res in results:
        spd = getattr(res, "_speed_kmh", 0)
        if spd > 0 and res.bounce_frame is not None:
            bp = res.bounce_point
            for f in range(res.bounce_frame, res.bounce_frame + 46):
                speed_labels[f] = (bp[0], bp[1], spd)

    # Delivery counter range list
    delivery_frame_range: list[tuple] = [
        (res.start_frame, res.end_frame, res.ball_no) for res in results
    ]

    trail     = deque(maxlen=70)   # longer trail for more visible history
    trail_ptr = 0
    frame_no  = 0
    last_ball_no = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Find current delivery
        current_res = next(
            (r for r in results if r.start_frame <= frame_no <= r.end_frame), None
        )
        current_ball_no = current_res.ball_no if current_res else -1

        # Reset trail when a new delivery starts
        if current_ball_no != last_ball_no:
            trail.clear()
            last_ball_no = current_ball_no

        # Advance trail pointer
        while (trail_ptr < len(all_track_pts) and
               all_track_pts[trail_ptr][0] <= frame_no):
            _, tx, ty, _ = all_track_pts[trail_ptr]
            trail.append((tx, ty))
            trail_ptr += 1

        vis = frame.copy()

        # ── Ghost trails from completed deliveries ────────────────
        completed_now = [r for r in results if r.end_frame < frame_no]
        for res in completed_now:
            if res.ball_no in delivery_paths:
                pts = delivery_paths[res.ball_no]
                for i in range(1, len(pts)):
                    a = i / len(pts)
                    color = (int(30*a), int(70*a), int(100*a))
                    cv2.line(vis, pts[i-1], pts[i], color, 1, cv2.LINE_AA)

        # ── Live glowing trail ────────────────────────────────────
        draw_ball_trail(vis, trail)

        # ── Ball circle at detected position ──────────────────────
        matched = frame_to_pt.get(frame_no)
        if matched:
            tx, ty, _ = matched
            cv2.circle(vis, (tx, ty), 11, CFG["color_ball"], 2, cv2.LINE_AA)
            cv2.circle(vis, (tx, ty),  5, (255, 255, 255), -1, cv2.LINE_AA)

        # ── Bounce marker ─────────────────────────────────────────
        for ball_no, pt in active_bounces.get(frame_no, []):
            draw_bounce_marker(vis, pt)

        # ── Speed label ───────────────────────────────────────────
        if frame_no in speed_labels:
            sx, sy, spd = speed_labels[frame_no]
            draw_speed_label(vis, spd, (sx, sy))

        # ── Badges — iterate ALL results against their window ─────
        # (fixes bug where only completed deliveries got badges)
        for res in results:
            draw_result_badge(vis, res, frame_no, badge_until)

        # ── Delivery counter ──────────────────────────────────────
        for sf, ef, bn in delivery_frame_range:
            if sf <= frame_no <= ef:
                draw_delivery_counter(vis, bn, len(results))
                break

        if debug:
            cv2.putText(
                vis,
                f"frame:{frame_no}  view:{view}  detections:{len(all_track_pts)}",
                (10, height - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                (160, 160, 160), 1, cv2.LINE_AA,
            )

        writer.write(vis)
        frame_no += 1

    cap.release()
    writer.release()

    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp_path, output_path)
    print(f"[INFO] Annotated video written: {output_path}")