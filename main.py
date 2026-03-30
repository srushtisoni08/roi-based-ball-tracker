from src.processor import process_video
import argparse

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Cricket ball bounce & length analyzer")
    ap.add_argument("--video",  required=True,          help="Input video path")
    ap.add_argument("--view",   default="auto",
                    choices=["auto", "side", "front"],  help="Camera view")
    ap.add_argument("--output", default=None,           help="Annotated output video path")
    ap.add_argument("--show",   action="store_true",    help="Live preview")
    ap.add_argument("--debug",  action="store_true",    help="Debug overlay")
    args = ap.parse_args()
 
    process_video(
        video_path=args.video,
        view=args.view,
        output_path=args.output,
        show=args.show,
        debug=args.debug,
    )