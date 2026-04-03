import cv2
import numpy as np
from dataclasses import dataclass
from collections import deque

from .tracker import BallTracker, TrackState

try:
    from ultralytics import YOLO as _YOLO
    _YOLO_AVAILABLE = True
except ImportError:
    _YOLO_AVAILABLE = False

# YOLO classes that fire on a cricket ball
_BALL_CLASSES = {
    29,   # frisbee
    32,   # sports ball
    33,   # kite
    36,   # skateboard
    37,   # baseball bat (sometimes fires on ball)
    38,   # baseball glove
}
_MIN_CONF = 0.08


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
                 **kwargs):

        if not _YOLO_AVAILABLE:
            raise RuntimeError("ultralytics not installed. Run: pip install ultralytics")

        self._model = _YOLO("yolov8n.pt")

        self._tracker = BallTracker(
            process_noise=1e-2,
            measurement_noise=5e-2,
            max_missed=12,
            confirm_hits=2,
        )

        self._frame_idx = 0
        self._recent: deque = deque(maxlen=8)

    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray):
        return self.process_frame(frame)

    def process_frame(self, frame: np.ndarray):
        results  = self._model(frame, conf=_MIN_CONF, verbose=False)
        boxes    = results[0].boxes

        candidates = []
        for box in boxes:
            cls = int(box.cls[0])
            if cls not in _BALL_CLASSES:
                continue
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0]
            cx = float((x1 + x2) / 2)
            cy = float((y1 + y2) / 2)
            r  = max(float(min(x2 - x1, y2 - y1) / 4), 5.0)
            candidates.append((cx, cy, r, conf))

        candidates.sort(key=lambda c: -c[3])

        candidate = self._trajectory_filter(candidates)

        state: TrackState = self._tracker.update(
            (candidate[0], candidate[1], candidate[2]) if candidate else None
        )
        self._frame_idx += 1

        if not self._tracker.is_active:
            self._tracker.reset()
            self._recent.clear()
            return None

        if not self._has_trajectory_confidence():
            return None

        yolo_conf = candidate[3] if candidate else 0.0

        det = Detection(
            x=int(state.x),
            y=int(state.y),
            radius=state.radius,
            frame_idx=self._frame_idx - 1,
            confidence=yolo_conf,
            tracked=state.confirmed,
            raw_cx=candidate[0] if candidate else None,
            raw_cy=candidate[1] if candidate else None,
        )
        det.frame = det.frame_idx
        return det

    def reset(self) -> None:
        self._tracker.reset()
        self._frame_idx = 0
        self._recent.clear()

    def get_trajectory(self) -> list[tuple]:
        return self._tracker.get_trajectory()

    @property
    def resolved_color(self) -> str | None:
        return "yolo"

    # ------------------------------------------------------------------

    def _trajectory_filter(self, candidates: list[tuple]) -> tuple | None:
        if not candidates:
            return None

        if len(self._recent) < 2:
            best = candidates[0]
            self._recent.append((self._frame_idx, best[0], best[1], best[2]))
            return best

        pred_x, pred_y = self._predict_next()
        MAX_DIST = 120
        best_candidate = None
        best_dist = MAX_DIST

        for cx, cy, cr, conf in candidates:
            dist = ((cx - pred_x)**2 + (cy - pred_y)**2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best_candidate = (cx, cy, cr, conf)

        if best_candidate is not None:
            self._recent.append((self._frame_idx,
                                  best_candidate[0],
                                  best_candidate[1],
                                  best_candidate[2]))
            return best_candidate

        return None

    def _predict_next(self) -> tuple:
        if len(self._recent) < 2:
            last = self._recent[-1]
            return last[1], last[2]
        recent_list = list(self._recent)
        x1, y1 = recent_list[-2][1], recent_list[-2][2]
        x2, y2 = recent_list[-1][1], recent_list[-1][2]
        return x2 + (x2 - x1), y2 + (y2 - y1)

    def _has_trajectory_confidence(self) -> bool:
        if len(self._recent) < 3:
            return False
        recent_list = list(self._recent)
        xs = [p[1] for p in recent_list]
        ys = [p[2] for p in recent_list]
        total_disp = ((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2) ** 0.5
        return total_disp >= 10