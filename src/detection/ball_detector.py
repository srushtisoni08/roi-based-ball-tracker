"""
ball_detector.py
────────────────
Full three-signal detection pipeline:

    Frame → Motion mask (A)
          → Colour mask  (B)
          → Combined mask (A ∩ B)
          → Contour filter (C): circularity + area
          → Best candidate
          → Kalman tracker update

Constructor is backward-compatible with the old signature:
    BallDetector(height, width, quality)   ← processor.py calls this
as well as the new keyword form:
    BallDetector(ball_color="red", min_radius=5, ...)
"""

import cv2
import numpy as np
from dataclasses import dataclass

from .motion_detector import MotionDetector
from .color_detector import ColorDetector
from .tracker import BallTracker, TrackState


@dataclass
class Detection:
    """One frame's result from the full pipeline."""
    x: float
    y: float
    radius: float
    frame_idx: int
    confidence: float
    tracked: bool = False
    raw_cx: float | None = None
    raw_cy: float | None = None


class BallDetector:
    """
    Detects and tracks a single cricket ball across frames.

    Accepts two call styles so it works with the existing processor.py
    without any changes to that file:

    Old style (positional):
        BallDetector(height, width, quality)
        -- height / width used to derive sensible radius bounds
        -- quality string ("high"/"low"/etc.) is accepted but ignored

    New style (keyword):
        BallDetector(ball_color="auto", min_radius=5, max_radius=40, ...)
    """

    def __init__(self,
                 height_or_color=None,
                 width=None,
                 quality=None,
                 *,
                 ball_color: str = "auto",
                 min_radius: int | None = None,
                 max_radius: int | None = None,
                 min_circularity: float = 0.70,
                 use_motion: bool = True):

        # ── Resolve constructor style ─────────────────────────────
        if isinstance(height_or_color, int):
            # Old positional style: BallDetector(height, width, quality)
            frame_h = height_or_color
            frame_w = width if isinstance(width, int) else 1280
            # Derive radius bounds from frame size:
            # A cricket ball at broadcast distance is roughly 0.5-3% of height
            _min_r = max(4,  int(frame_h * 0.005))
            _max_r = max(30, int(frame_h * 0.030))
            _color = "auto"
        else:
            # New keyword style (height_or_color may be None or a color string)
            _color = height_or_color if isinstance(height_or_color, str) else ball_color
            frame_h = 720   # default, not used for radius if explicit values given
            _min_r  = None
            _max_r  = None

        # Explicit keyword args override derived values
        self.min_radius     = min_radius if min_radius is not None else _min_r or 5
        self.max_radius     = max_radius if max_radius is not None else _max_r or 40
        self.min_circularity = min_circularity
        self.use_motion     = use_motion

        self._motion  = MotionDetector(blur_ksize=5,
                                       min_area=self.min_radius ** 2)
        self._color   = ColorDetector(ball_color=_color, dilate=4)
        self._tracker = BallTracker(
            process_noise=1e-2,
            measurement_noise=5e-2,
            max_missed=8,
            confirm_hits=3,
        )

        self._frame_idx = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray):
        """
        Alias for process_frame() — keeps compatibility with the old
        detector interface used in processor.py:
            det = detector.detect(frame)
            det.x, det.y, det.radius, det.frame
        """
        return self.process_frame(frame)

    def process_frame(self, frame: np.ndarray):
        """Run the full pipeline on one frame. Returns Detection or None."""
        motion_mask = self._motion.update(frame) if self.use_motion else None
        color_mask  = self._color.detect(frame)

        if motion_mask is not None:
            combined = cv2.bitwise_and(color_mask, motion_mask)
        else:
            combined = color_mask

        candidate = self._best_candidate(combined)
        state: TrackState = self._tracker.update(candidate)
        self._frame_idx += 1

        if not self._tracker.is_active:
            return None

        det = Detection(
            x=int(state.x),
            y=int(state.y),
            radius=state.radius,
            frame_idx=self._frame_idx - 1,
            confidence=self._confidence(state, candidate),
            tracked=state.confirmed,
            raw_cx=candidate[0] if candidate else None,
            raw_cy=candidate[1] if candidate else None,
        )
        # processor.py reads det.frame (not det.frame_idx) — expose both
        det.frame = det.frame_idx
        return det

    def reset(self) -> None:
        self._motion.reset()
        self._tracker.reset()
        self._frame_idx = 0

    def get_trajectory(self) -> list[tuple]:
        return self._tracker.get_trajectory()

    @property
    def resolved_color(self) -> str | None:
        return self._color.resolved_color

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _best_candidate(self, mask: np.ndarray) -> tuple | None:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best_score = -1.0
        best = None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * self.min_radius ** 2:
                continue
            if area > np.pi * self.max_radius ** 2:
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter < 1:
                continue

            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity < self.min_circularity:
                continue

            (cx, cy), enc_r = cv2.minEnclosingCircle(cnt)
            if enc_r < self.min_radius or enc_r > self.max_radius:
                continue

            expected   = np.pi * enc_r ** 2
            area_score = min(area / expected, 1.0)
            score      = circularity * area_score

            if score > best_score:
                best_score = score
                best = (float(cx), float(cy), float(enc_r))

        return best

    @staticmethod
    def _confidence(state: TrackState, raw: tuple | None) -> float:
        base    = 0.5 if state.confirmed else 0.2
        bonus   = 0.5 if raw is not None else 0.0
        penalty = min(0.05 * state.missed, 0.4)
        return max(0.0, min(1.0, base + bonus - penalty))