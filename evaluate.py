import argparse
import json
import os
import sys
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.analysis.evaluator import Evaluator, EvalResult
from src.utils.metrics import bounce_metrics, print_report


def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate cricket analyzer against ground-truth labels.")
    p.add_argument("--input",  default="data/input",
                   help="Folder containing input videos (default: data/input)")
    p.add_argument("--gt",     default="data/ground_truth",
                   help="Folder containing ground-truth JSON files")
    p.add_argument("--output", default="data/output",
                   help="Folder where analyzer writes result JSON files")
    p.add_argument("--tolerance", type=int, default=5,
                   help="Frame tolerance for bounce hit (default: 5)")
    p.add_argument("--save",   default=None,
                   help="Optional path to save full results as JSON")
    return p.parse_args()


def load_predictions(output_dir: str) -> dict[str, dict]:
    """
    Load per-video prediction JSONs written by the analyzer.

    Expected filename pattern: <video_stem>_result.json
    Expected keys: video, bounce_detected, bounce_frame, length, confidence
    """
    preds = {}
    out_path = Path(output_dir)
    if not out_path.exists():
        return preds

    for jf in out_path.glob("*_result.json"):
        with open(jf) as f:
            data = json.load(f)
        name = os.path.basename(data.get("video", jf.stem))
        preds[name] = data
    return preds


def main():
    args = parse_args()

    evaluator = Evaluator(args.gt, frame_tolerance=args.tolerance)
    predictions = load_predictions(args.output)

    if not predictions:
        print(f"[evaluate] No prediction JSONs found in '{args.output}'.")
        print("           Run main.py first to generate predictions.")
        sys.exit(0)

    eval_results: list[EvalResult] = []

    for video_name, pred in predictions.items():
        if not evaluator.has_label(video_name):
            print(f"  [skip] No ground-truth label for: {video_name}")
            continue

        er = evaluator.evaluate_one(
            video_name=video_name,
            pred_bounce_frame=pred.get("bounce_frame"),
            pred_length=pred.get("length"),
            pred_confidence=pred.get("confidence"),
        )
        if er:
            eval_results.append(er)

    if not eval_results:
        print("[evaluate] No labeled videos matched predictions.")
        sys.exit(0)

    # ── Aggregate ────────────────────────────────────────────────────
    agg = Evaluator.aggregate(eval_results)

    # ── Print report ─────────────────────────────────────────────────
    bounce_m = {
        "n_videos": agg["n_videos"],
        "n_detected": agg["n_detected"],
        "detection_rate": agg["detection_rate"],
        "n_hits": int(agg["bounce_accuracy"] * agg["n_videos"]),
        "hit_rate": agg["bounce_accuracy"],
        "mean_frame_error": agg["mean_frame_error"],
        "median_frame_error": agg["median_frame_error"],
    }
    print_report(bounce_m,
                 length_acc=agg["length_accuracy"],
                 mean_conf=agg.get("mean_confidence"))

    # ── Per-video breakdown ──────────────────────────────────────────
    print("\n  Per-video breakdown:")
    print(f"  {'Video':<25} {'Bounce':>7} {'FrameErr':>9} {'Length':>8}")
    print("  " + "─" * 55)
    for r in eval_results:
        b  = "✓" if r.bounce_correct else "✗"
        fe = f"±{r.frame_err}" if r.frame_err is not None else "N/A"
        lc = "✓" if r.length_correct else "✗"
        print(f"  {r.video:<25} {b:>7} {fe:>9} {lc:>8}")

    # ── Optional JSON save ───────────────────────────────────────────
    if args.save:
        out = {
            "summary": agg,
            "per_video": [r.to_dict() for r in eval_results],
        }
        with open(args.save, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\n  Full results saved to: {args.save}")


if __name__ == "__main__":
    main()