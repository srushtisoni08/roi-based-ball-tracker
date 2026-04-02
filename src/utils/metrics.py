"""
metrics.py
──────────
Low-level metric helpers used by evaluator.py and evaluate.py.

All functions are pure (no I/O) so they are easy to unit-test.
"""

import numpy as np
from typing import Sequence


# ── Bounce detection ──────────────────────────────────────────────────────────

def frame_error(predicted: int | None, ground_truth: int) -> int | None:
    """
    Absolute frame difference between predicted and true bounce frame.

    Returns None when prediction is missing (detector didn't fire).
    """
    if predicted is None:
        return None
    return abs(predicted - ground_truth)


def bounce_hit(predicted: int | None,
               ground_truth: int,
               tolerance: int = 5) -> bool:
    """
    True when |predicted - ground_truth| ≤ tolerance frames.

    A tolerance of 5 frames is roughly ±0.2 s at 25 fps – fair for
    a non-AI tracker.
    """
    err = frame_error(predicted, ground_truth)
    return err is not None and err <= tolerance


# ── Length classification ────────────────────────────────────────────────────

def length_accuracy(predictions: Sequence[str | None],
                    ground_truths: Sequence[str]) -> float:
    """
    Fraction of videos where predicted length == ground_truth length.

    None predictions count as wrong.
    """
    if not ground_truths:
        return 0.0
    correct = sum(p == g for p, g in zip(predictions, ground_truths))
    return correct / len(ground_truths)


# ── Aggregate bounce metrics ─────────────────────────────────────────────────

def bounce_metrics(predicted_frames: Sequence[int | None],
                   true_frames: Sequence[int],
                   tolerance: int = 5) -> dict:
    """
    Compute a full set of bounce detection metrics.

    Parameters
    ----------
    predicted_frames : sequence of int or None
        One entry per test video.  None = detector didn't fire.
    true_frames : sequence of int
        Ground-truth bounce frames, same length.
    tolerance : int
        Frame window for a hit.

    Returns
    -------
    dict with keys:
        n_videos, n_detected, detection_rate,
        n_hits, hit_rate, mean_frame_error, median_frame_error
    """
    assert len(predicted_frames) == len(true_frames)

    n = len(true_frames)
    detected = [p for p in predicted_frames if p is not None]
    hits = [bounce_hit(p, g, tolerance)
            for p, g in zip(predicted_frames, true_frames)]
    errors = [frame_error(p, g)
              for p, g in zip(predicted_frames, true_frames)
              if p is not None]

    return {
        "n_videos": n,
        "n_detected": len(detected),
        "detection_rate": len(detected) / n if n else 0.0,
        "n_hits": sum(hits),
        "hit_rate": sum(hits) / n if n else 0.0,
        "mean_frame_error": float(np.mean(errors)) if errors else None,
        "median_frame_error": float(np.median(errors)) if errors else None,
    }


# ── Confidence calibration ───────────────────────────────────────────────────

def mean_confidence(confidences: Sequence[float]) -> float:
    return float(np.mean(confidences)) if confidences else 0.0


# ── Pretty-print helper ──────────────────────────────────────────────────────

def print_report(bounce_m: dict,
                 length_acc: float,
                 mean_conf: float | None = None) -> None:
    """Print a human-readable evaluation report to stdout."""
    sep = "─" * 45
    print(sep)
    print("  Cricket Analyzer – Evaluation Report")
    print(sep)
    print(f"  Videos tested        : {bounce_m['n_videos']}")
    print(f"  Bounce detected      : {bounce_m['n_detected']} "
          f"({100*bounce_m['detection_rate']:.0f}%)")
    print(f"  Bounce accuracy      : {100*bounce_m['hit_rate']:.0f}%")

    if bounce_m["mean_frame_error"] is not None:
        print(f"  Mean frame error     : ±{bounce_m['mean_frame_error']:.1f} frames")
        print(f"  Median frame error   : ±{bounce_m['median_frame_error']:.1f} frames")
    else:
        print("  Frame error          : N/A (no detections)")

    print(f"  Length accuracy      : {100*length_acc:.0f}%")

    if mean_conf is not None:
        print(f"  Mean confidence      : {mean_conf:.2f}")
    print(sep)