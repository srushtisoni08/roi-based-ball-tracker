import os
import glob
import argparse
from src.processor import process_video

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cricket ball bounce & length analyzer")
    ap.add_argument("--video",  help="Single input video path")
    ap.add_argument("--folder", default="data/input_videos", help="Folder containing multiple videos")
    ap.add_argument("--view",   default="auto", choices=["auto", "side", "front"], help="Camera view")
    ap.add_argument("--output", default="data/output_videos", help="Output video folder")
    ap.add_argument("--show",   action="store_true", help="Live preview")
    ap.add_argument("--debug",  action="store_true", help="Debug overlay")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # ── Collect videos to process ──────────────────────────────
    if args.video:
        videos = [args.video]
    else:
        videos = (glob.glob(os.path.join(args.folder, "*.mp4")) +
                  glob.glob(os.path.join(args.folder, "*.avi")) +
                  glob.glob(os.path.join(args.folder, "*.mov")))

        if not videos:
            print(f"[ERROR] No videos found in folder: {args.folder}")
            exit(1)

    print(f"[INFO] Found {len(videos)} video(s) to process\n")

    # ── Process each video ─────────────────────────────────────
    for video_path in videos:
        video_name  = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(args.output, f"{video_name}_result.mp4")

        print(f"\n{'='*58}")
        print(f"  Processing: {video_name}")
        print(f"{'='*58}")

        process_video(
            video_path  = video_path,
            view        = args.view,
            output_path = output_path,
            show        = args.show,
            debug       = args.debug,
        )