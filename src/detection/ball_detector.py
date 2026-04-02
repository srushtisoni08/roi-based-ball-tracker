import cv2
import numpy as np
from dataclasses import dataclass
from collections import deque

from .motion_detector import MotionDetector
from .color_detector import ColorDetector
from .tracker import BallTracker, TrackState


@dataclass
class Detection:
    x: float
    y: float
    radius: float
    frame_idx: int
    confidence: float
    tracked: bool = False
    raw_cx: float | None = None
    raw_cy: float | None = None


class BallDetector:

    def __init__(self,
                 height_or_color=None,
                 width=None,
                 quality=None,
                 *,
                 ball_color: str = "auto",
                 min_radius: int | None = None,
                 max_radius: int | None = None,
                 min_circularity: float = 0.65,
                 use_motion: bool = True):

        if isinstance(height_or_color, int):
            frame_h = height_or_color
            frame_w = width if isinstance(width, int) else 1280
            _min_r = max(4, int(frame_h * 0.005))
            _max_r = max(30, int(frame_h * 0.030))
            _color = "auto"
        else:
            _color = height_or_color if isinstance(height_or_color, str) else ball_color
            frame_h = 720
            _min_r = None
            _max_r = None

        self.min_radius      = min_radius if min_radius is not None else _min_r or 5
        self.max_radius      = max_radius if max_radius is not None else _max_r or 40
        self.min_circularity = min_circularity
        self.use_motion      = use_motion

        self._motion  = MotionDetector(blur_ksize=5, min_area=self.min_radius ** 2)
        self._color   = ColorDetector(ball_color=_color, dilate=4)
        self._tracker = BallTracker(
            process_noise=1e-2,
            measurement_noise=5e-2,
            max_missed=12,
            confirm_hits=2,
        )

        self._frame_idx = 0

        # ── Trajectory filter state ──────────────────────────────
        # Keep last N raw candidates to check consistency
        self._recent: deque = deque(maxlen=6)  # (frame_idx, x, y, r)
        self._confirmed_trajectory: bool = False

    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray):
        return self.process_frame(frame)

    def process_frame(self, frame: np.ndarray):
        from src.utils.roi_mask import get_roi_mask

        motion_mask = self._motion.update(frame) if self.use_motion else None
        color_mask  = self._color.detect(frame)

        if motion_mask is not None:
            combined = cv2.bitwise_and(color_mask, motion_mask)
        else:
            combined = color_mask

        # Apply ROI mask
        roi = get_roi_mask(frame)
        combined = cv2.bitwise_and(combined, roi)

        # Get ALL candidates this frame, not just best one
        candidates = self._all_candidates(combined)

        # Pick the candidate most consistent with recent trajectory
        candidate = self._trajectory_filter(candidates)

        state: TrackState = self._tracker.update(candidate)
        self._frame_idx += 1

        if not self._tracker.is_active:
            self._tracker.reset()
            self._recent.clear()
            self._confirmed_trajectory = False
            return None

        # Only return detection if we have trajectory confidence
        if not self._has_trajectory_confidence():
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
        det.frame = det.frame_idx
        return det

    def reset(self) -> None:
        self._motion.reset()
        self._tracker.reset()
        self._frame_idx = 0
        self._recent.clear()
        self._confirmed_trajectory = False

    def get_trajectory(self) -> list[tuple]:
        return self._tracker.get_trajectory()

    @property
    def resolved_color(self) -> str | None:
        return self._color.resolved_color

    # ------------------------------------------------------------------
    # Trajectory filter — the core fix
    # ------------------------------------------------------------------

    def _all_candidates(self, mask: np.ndarray) -> list[tuple]:
        """Return ALL circular candidates in mask, sorted by score."""
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        results = []
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
            results.append((float(cx), float(cy), float(enc_r), score))

        results.sort(key=lambda x: -x[3])
        return [(x, y, r) for x, y, r, _ in results]

    def _trajectory_filter(self, candidates: list[tuple]) -> tuple | None:
        """
        From all candidates, pick the one that best continues
        the recent trajectory. If no recent trajectory exists,
        accept the best scoring candidate and start building one.
        """
        if not candidates:
            return None

        if len(self._recent) < 2:
            # Not enough history — take best candidate but don't confirm yet
            best = candidates[0]
            self._recent.append((self._frame_idx, best[0], best[1], best[2]))
            return best

        # Predict where ball should be based on recent movement
        pred_x, pred_y, pred_vx, pred_vy = self._predict_next()

        # Find candidate closest to prediction within speed limit
        # A ball moves at most ~150px per frame at 30fps
        MAX_DIST = 150
        best_candidate = None
        best_dist = MAX_DIST

        for cx, cy, cr in candidates:
            dist = ((cx - pred_x)**2 + (cy - pred_y)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_candidate = (cx, cy, cr)

        if best_candidate is not None:
            self._recent.append((self._frame_idx,
                                  best_candidate[0],
                                  best_candidate[1],
                                  best_candidate[2]))
            return best_candidate

        # No candidate near prediction — ball lost this frame
        return None

    def _predict_next(self) -> tuple:
        """Predict next ball position from recent history."""
        if len(self._recent) < 2:
            last = self._recent[-1]
            return last[1], last[2], 0.0, 0.0

        # Use last 2 points for velocity
        recent_list = list(self._recent)
        x1, y1 = recent_list[-2][1], recent_list[-2][2]
        x2, y2 = recent_list[-1][1], recent_list[-1][2]
        vx = x2 - x1
        vy = y2 - y1
        return x2 + vx, y2 + vy, vx, vy

    def _has_trajectory_confidence(self) -> bool:
        """
        Return True only if recent detections form a consistent
        straight-ish path — ruling out stationary objects like
        elbows, shoes, stumps which stay in one place.
        """
        if len(self._recent) < 3:
            return False

        recent_list = list(self._recent)
        xs = [p[1] for p in recent_list]
        ys = [p[2] for p in recent_list]

        # Must have moved at least 20px total in last 3 detections
        total_disp = ((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2) ** 0.5
        if total_disp < 20:
            return False

        # Speed must be consistent — no teleporting
        dists = []
        for i in range(1, len(recent_list)):
            d = ((xs[i]-xs[i-1])**2 + (ys[i]-ys[i-1])**2) ** 0.5
            dists.append(d)

        if not dists:
            return False

        # Reject if speed varies wildly (ratio of max to min > 5)
        # A real ball has fairly consistent speed between frames
        min_d = max(min(dists), 0.1)
        max_d = max(dists)
        if max_d / min_d > 5:
            return False

        return True

    # ------------------------------------------------------------------

    def _best_candidate(self, mask: np.ndarray) -> tuple | None:
        candidates = self._all_candidates(mask)
        return candidates[0] if candidates else None

    @staticmethod
    def _confidence(state: TrackState, raw: tuple | None) -> float:
        base    = 0.5 if state.confirmed else 0.2
        bonus   = 0.5 if raw is not None else 0.0
        penalty = min(0.05 * state.missed, 0.4)
        return max(0.0, min(1.0, base + bonus - penalty))