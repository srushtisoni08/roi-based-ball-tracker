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
    confidence: float          # 0-1, based on circularity × signal coverage
    tracked: bool = False      # True if Kalman tracker is confirmed
    raw_cx: float | None = None   # raw contour centre before Kalman
    raw_cy: float | None = None


class BallDetector:
    """
    Detects and tracks a single cricket ball across frames.

    Parameters
    ----------
    ball_color : str
        "red" | "white" | "pink" | "auto"
    min_radius : int
        Minimum accepted ball radius in pixels.
    max_radius : int
        Maximum accepted ball radius in pixels.
    min_circularity : float
        0-1 score; 1 = perfect circle.  Cricket ball ≥ 0.70 is typical.
    use_motion : bool
        If False, skip motion masking (useful for very slow cameras).
    """

    def __init__(self,
                 ball_color: str = "auto",
                 min_radius: int = 5,
                 max_radius: int = 40,
                 min_circularity: float = 0.70,
                 use_motion: bool = True):

        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_circularity = min_circularity
        self.use_motion = use_motion

        self._motion = MotionDetector(blur_ksize=5, min_area=min_radius ** 2)
        self._color = ColorDetector(ball_color=ball_color, dilate=4)
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

    def process_frame(self, frame: np.ndarray) -> Detection | None:
        """
        Run the full pipeline on one frame.

        Returns a Detection (with Kalman-smoothed position) or None when
        the tracker has been inactive too long.
        """
        # ── Signal A: motion ──────────────────────────────────────
        motion_mask = self._motion.update(frame) if self.use_motion else None

        # ── Signal B: colour ──────────────────────────────────────
        color_mask = self._color.detect(frame)

        # ── Combine A ∩ B ─────────────────────────────────────────
        if motion_mask is not None:
            combined = cv2.bitwise_and(color_mask, motion_mask)
        else:
            combined = color_mask

        # ── Signal C: contour filter ──────────────────────────────
        candidate = self._best_candidate(combined, frame)

        # ── Kalman update ─────────────────────────────────────────
        state: TrackState = self._tracker.update(candidate)

        self._frame_idx += 1

        if not self._tracker.is_active:
            return None

        det = Detection(
            x=state.x,
            y=state.y,
            radius=state.radius,
            frame_idx=self._frame_idx - 1,
            confidence=self._confidence(state, candidate),
            tracked=state.confirmed,
            raw_cx=candidate[0] if candidate else None,
            raw_cy=candidate[1] if candidate else None,
        )
        return det

    def reset(self) -> None:
        """Reset between deliveries or videos."""
        self._motion.reset()
        self._tracker.reset()
        self._frame_idx = 0

    def get_trajectory(self) -> list[tuple]:
        """Full (x, y, r) history from the Kalman tracker."""
        return self._tracker.get_trajectory()

    @property
    def resolved_color(self) -> str | None:
        return self._color.resolved_color

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _best_candidate(self,
                        mask: np.ndarray,
                        frame: np.ndarray) -> tuple | None:
        """
        Find the single best circular blob in the combined mask.

        Scores each contour by:
            score = circularity × (area / expected_area)
        and returns the (cx, cy, radius) of the highest-scorer.
        """
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
            radius = enc_r

            if radius < self.min_radius or radius > self.max_radius:
                continue

            # Expected area for this radius
            expected = np.pi * radius ** 2
            area_score = min(area / expected, 1.0)

            score = circularity * area_score
            if score > best_score:
                best_score = score
                best = (float(cx), float(cy), float(radius))

        return best

    @staticmethod
    def _confidence(state: TrackState, raw: tuple | None) -> float:
        """Simple 0-1 confidence: confirmed + recent detection = high."""
        base = 0.5 if state.confirmed else 0.2
        bonus = 0.5 if raw is not None else 0.0
        penalty = min(0.05 * state.missed, 0.4)
        return max(0.0, min(1.0, base + bonus - penalty))