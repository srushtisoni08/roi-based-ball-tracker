"""
evaluator.py
────────────
Compares the analyzer's output for one video against its ground-truth
label and returns structured results that evaluate.py can aggregate.

Ground-truth format (JSON)
--------------------------
{
    "video": "sample.mp4",
    "bounce_frame": 120,        // frame index (0-based)
    "length": "good",           // yorker | full | good | short
    "notes": "optional string"  // ignored by evaluator
}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..utils.metrics import bounce_hit, frame_error, length_accuracy


@dataclass
class EvalResult:
    """Evaluation result for one video."""
    video: str
    gt_bounce_frame: int
    gt_length: str

    pred_bounce_frame: int | None = None
    pred_length: str | None = None
    pred_confidence: float | None = None

    # Computed fields (filled by evaluate())
    bounce_correct: bool = False
    frame_err: int | None = None
    length_correct: bool = False

    def to_dict(self) -> dict:
        return {
            "video": self.video,
            "gt_bounce_frame": self.gt_bounce_frame,
            "gt_length": self.gt_length,
            "pred_bounce_frame": self.pred_bounce_frame,
            "pred_length": self.pred_length,
            "pred_confidence": self.pred_confidence,
            "bounce_correct": self.bounce_correct,
            "frame_error": self.frame_err,
            "length_correct": self.length_correct,
        }


class Evaluator:
    """
    Loads ground-truth labels and evaluates predictions against them.

    Usage
    -----
        ev = Evaluator("data/ground_truth")
        result = ev.evaluate_one(
            video_name="sample.mp4",
            pred_bounce_frame=118,
            pred_length="good",
            pred_confidence=0.82,
        )
        print(result)
    """

    def __init__(self, ground_truth_dir: str | Path,
                 frame_tolerance: int = 5):
        self.gt_dir = Path(ground_truth_dir)
        self.tolerance = frame_tolerance
        self._labels: dict[str, dict] = {}
        self._load_labels()

    # ------------------------------------------------------------------

    def _load_labels(self) -> None:
        """Load all JSON label files from the ground_truth directory."""
        if not self.gt_dir.exists():
            return
        for path in self.gt_dir.glob("*.json"):
            with open(path) as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    name = os.path.basename(item.get("video", ""))
                    self._labels[name] = item
            else:
                name = os.path.basename(data.get("video", ""))
                self._labels[name] = data

    def has_label(self, video_name: str) -> bool:
        return os.path.basename(video_name) in self._labels

    def evaluate_one(self,
                     video_name: str,
                     pred_bounce_frame: int | None,
                     pred_length: str | None,
                     pred_confidence: float | None = None) -> EvalResult | None:
        """
        Compare one video's predictions to its ground-truth label.

        Returns None if no label is found for this video.
        """
        key = os.path.basename(video_name)
        label = self._labels.get(key)
        if label is None:
            return None

        gt_bf = int(label["bounce_frame"])
        gt_len = str(label["length"]).lower()

        result = EvalResult(
            video=key,
            gt_bounce_frame=gt_bf,
            gt_length=gt_len,
            pred_bounce_frame=pred_bounce_frame,
            pred_length=pred_length,
            pred_confidence=pred_confidence,
        )

        result.bounce_correct = bounce_hit(pred_bounce_frame, gt_bf,
                                           self.tolerance)
        result.frame_err = frame_error(pred_bounce_frame, gt_bf)
        result.length_correct = (pred_length is not None and
                                 pred_length.lower() == gt_len)

        return result

    def evaluate_batch(self, results: list[dict]) -> list[EvalResult]:
        """
        Evaluate a list of result dicts (one per video) against labels.

        Each dict must have keys:
            video, bounce_frame, length, confidence (optional)
        """
        eval_results = []
        for r in results:
            er = self.evaluate_one(
                video_name=r.get("video", ""),
                pred_bounce_frame=r.get("bounce_frame"),
                pred_length=r.get("length"),
                pred_confidence=r.get("confidence"),
            )
            if er is not None:
                eval_results.append(er)
        return eval_results

    @staticmethod
    def aggregate(eval_results: list[EvalResult]) -> dict:
        """
        Compute aggregate metrics across a list of EvalResult objects.

        Returns a dict suitable for JSON serialisation or pretty-printing.
        """
        n = len(eval_results)
        if n == 0:
            return {"n_videos": 0}

        n_detected = sum(1 for r in eval_results
                         if r.pred_bounce_frame is not None)
        n_bounce_correct = sum(r.bounce_correct for r in eval_results)
        n_length_correct = sum(r.length_correct for r in eval_results)

        errors = [r.frame_err for r in eval_results
                  if r.frame_err is not None]
        confs = [r.pred_confidence for r in eval_results
                 if r.pred_confidence is not None]

        import numpy as np
        return {
            "n_videos": n,
            "n_detected": n_detected,
            "detection_rate": n_detected / n,
            "bounce_accuracy": n_bounce_correct / n,
            "length_accuracy": n_length_correct / n,
            "mean_frame_error": float(np.mean(errors)) if errors else None,
            "median_frame_error": float(np.median(errors)) if errors else None,
            "mean_confidence": float(np.mean(confs)) if confs else None,
        }