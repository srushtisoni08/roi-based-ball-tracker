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
    draw_hud,
    draw_ball_trail,
    draw_bounce_marker,
    draw_length_banner,
    draw_pitch_zones_side,
)

# How many frames to show the delivery banner after a delivery ends
BANNER_DURATION_FRAMES = 90


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
    all_detections: list[TrackPoint] = []
    trail           = deque(maxlen=60)
    completed       : list[DeliveryResult] = []
    bounce_markers  : dict[int, tuple] = {}   # ball_no → (x, y)
    banner_until    : dict[int, int]   = {}   # ball_no → last frame to show banner

    # ── Pass 1: Detect all frames and write annotated video ───────
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

        # ── Draw overlays ──────────────────────────────────────────
        vis = frame.copy()

        if view == "side":
            draw_pitch_zones_side(vis)

        draw_ball_trail(vis, trail)

        if det:
            cv2.circle(vis, (det.x, det.y), int(det.radius) + 3,
                       CFG["color_ball"], 2, cv2.LINE_AA)

        # Persist bounce markers on screen
        for ball_no, pt in bounce_markers.items():
            draw_bounce_marker(vis, pt, f"Ball {ball_no}")

        draw_hud(vis, completed, len(completed) + 1)

        for res in completed:
            draw_length_banner(vis, res, frame_no, banner_until)

        if debug:
            cv2.putText(
                vis,
                f"frame:{frame_no}  dets:{len(all_detections)}  view:{view}",
                (10, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (160, 160, 160), 1, cv2.LINE_AA,
            )

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

    # ── Pass 2: Segment & analyse deliveries ─────────────────────
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

        # Find the frame index closest to the bounce point
        if bounce_pt is not None:
            bounce_frame = min(
                delivery,
                key=lambda p: abs(p.x - bounce_pt[0]) + abs(p.y - bounce_pt[1])
            ).frame
        else:
            bounce_frame = None

        res = DeliveryResult(
            ball_no       = i,
            bounced       = bounced,
            length        = length,
            bounce_frame  = bounce_frame,
            bounce_point  = bounce_pt,
            start_frame   = start_f,
            end_frame     = end_f,
            duration_s    = round(dur, 2),
            tracked_points= len(delivery),
            track         = delivery,
        )
        results.append(res)

        b_str = "BOUNCED  ↓" if bounced else "FULL TOSS →"
        l_col = {
            "Yorker": "\033[92m", "Full": "\033[94m",
            "Good":   "\033[96m", "Short": "\033[91m",
        }.get(length, "")
        reset = "\033[0m"
        print(f"  Ball {i:>2}:  {b_str:<14}  Length: {l_col}{length:<8}{reset}"
              f"  ({len(delivery)} pts, {dur:.2f}s)")

    print("=" * 58)
    bounced_count   = sum(1 for r in results if r.bounced)
    no_bounce_count = sum(1 for r in results if not r.bounced)
    print(f"  Pitched deliveries : {bounced_count}")
    print(f"  Full tosses        : {no_bounce_count}")
    print("  Length breakdown   : " + "  ".join(
        f"{k}={sum(1 for r in results if r.length == k)}"
        for k in ["Yorker", "Full", "Good", "Short"]
    ))
    print("=" * 58)

    # ── Pass 3: Re-render video with correct annotations ──────────
    # Now that we know the analysis results, do a second pass to write
    # the final annotated video with correct labels on the right frames.
    if output_path and results:
        _render_annotated_video(
            video_path, output_path, results, view, fps, width, height, debug
        )

    # ── Save JSON ─────────────────────────────────────────────────
    report = {
        "video"          : video_path,
        "view"           : view,
        "camera_quality" : quality,
        "fps"            : fps,
        "total_deliveries": len(results),
        "pitched"        : bounced_count,
        "full_toss"      : no_bounce_count,
        "length_summary" : {
            k: sum(1 for r in results if r.length == k)
            for k in ["Yorker", "Full", "Good", "Short"]
        },
        "deliveries": [
            {
                "ball"          : r.ball_no,
                "bounced"       : r.bounced,
                "length"        : r.length,
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


# ── Second-pass renderer ──────────────────────────────────────────
def _render_annotated_video(video_path, output_path, results, view,
                            fps, width, height, debug):
    """
    Re-reads the source video and writes the final annotated output.
    Because we now have the full analysis results, every frame gets
    the correct bounce markers, length banners, and HUD entries.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Write to a temp file then rename so the first-pass file is replaced
    tmp_path = output_path + ".tmp.mp4"
    writer   = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    # Build per-frame lookup tables from results
    # bounce_markers_by_frame[f] = list of (ball_no, x, y)
    bounce_markers_by_frame: dict[int, list] = {}
    for res in results:
        if res.bounce_point is not None and res.bounce_frame is not None:
            bf = res.bounce_frame
            if bf not in bounce_markers_by_frame:
                bounce_markers_by_frame[bf] = []
            bounce_markers_by_frame[bf].append((res.ball_no, res.bounce_point))

    # Track: for each delivery, which frames does it span?
    # So we can draw the bounce marker from bounce_frame onwards
    # Build: frame → list of (ball_no, bounce_pt) that are "active"
    # A bounce marker is "active" from its bounce_frame until end of that delivery + 60 frames
    active_bounces: dict[int, list] = {}
    for res in results:
        if res.bounce_point is not None and res.bounce_frame is not None:
            start = res.bounce_frame
            end   = res.end_frame + 90
            for f in range(start, end + 1):
                if f not in active_bounces:
                    active_bounces[f] = []
                active_bounces[f].append((res.ball_no, res.bounce_point))

    # Completed results up to each frame
    # result is "completed" once its end_frame has passed
    banner_until: dict[int, int] = {
        r.ball_no: r.end_frame + BANNER_DURATION_FRAMES for r in results
    }

    # Rebuild per-ball track point lookup for trail drawing
    # trail_by_frame[f] = list of (x,y) up to frame f for the active delivery
    # Build a sorted list of all track points with their ball_no
    all_track_pts: list[tuple] = []   # (frame, x, y, ball_no)
    for res in results:
        for pt in res.track:
            all_track_pts.append((pt.frame, pt.x, pt.y, res.ball_no))
    all_track_pts.sort(key=lambda t: t[0])

    trail       = deque(maxlen=60)
    trail_ptr   = 0          # index into all_track_pts

    frame_no = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Advance trail pointer up to current frame
        while trail_ptr < len(all_track_pts) and all_track_pts[trail_ptr][0] <= frame_no:
            _, tx, ty, _ = all_track_pts[trail_ptr]
            trail.append((tx, ty))
            trail_ptr += 1

        # Determine completed results up to this frame
        completed_now = [r for r in results if r.end_frame < frame_no]
        current_ball  = next((r for r in results
                              if r.start_frame <= frame_no <= r.end_frame), None)
        current_ball_no = (current_ball.ball_no if current_ball
                           else (completed_now[-1].ball_no + 1 if completed_now else 1))

        vis = frame.copy()

        if view == "side":
            draw_pitch_zones_side(vis)

        draw_ball_trail(vis, trail)

        # Draw bounce markers for all active deliveries
        for ball_no, pt in active_bounces.get(frame_no, []):
            draw_bounce_marker(vis, pt, f"Ball {ball_no}")

        # Ball circle: find if any track point lands on this exact frame
        matched = next((t for t in all_track_pts if t[0] == frame_no), None)
        if matched:
            _, tx, ty, _ = matched
            cv2.circle(vis, (tx, ty), 10, CFG["color_ball"], 2, cv2.LINE_AA)

        draw_hud(vis, completed_now, current_ball_no)

        for res in completed_now:
            draw_length_banner(vis, res, frame_no, banner_until)

        if debug:
            cv2.putText(
                vis,
                f"frame:{frame_no}  view:{view}",
                (10, height - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (160, 160, 160), 1, cv2.LINE_AA,
            )

        writer.write(vis)
        frame_no += 1

    cap.release()
    writer.release()

    # Replace the first-pass file with the correctly annotated one
    if os.path.exists(output_path):
        os.remove(output_path)
    os.rename(tmp_path, output_path)
    print(f"[INFO] Annotated video written: {output_path}")