import cv2
import numpy as np

_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
    "leather": [
        (np.array([5,  50, 60],  dtype=np.uint8),
         np.array([25, 210, 230], dtype=np.uint8)),
    ],
    "red": [
        (np.array([0,   80, 60],  dtype=np.uint8),
         np.array([12, 255, 255], dtype=np.uint8)),
        (np.array([165, 80, 60],  dtype=np.uint8),
         np.array([180, 255, 255], dtype=np.uint8)),
    ],
    "white": [
        (np.array([0,   0, 190], dtype=np.uint8),
         np.array([180, 40, 255], dtype=np.uint8)),
    ],
    "pink": [
        (np.array([140, 50, 80],  dtype=np.uint8),
         np.array([175, 255, 255], dtype=np.uint8)),
        (np.array([0,   50, 80],  dtype=np.uint8),
         np.array([10, 255, 255], dtype=np.uint8)),
    ],
}

_AUTO_MIN_COVERAGE = 0.0002


class ColorDetector:
    def __init__(self, ball_color: str = "auto", dilate: int = 3):
        if ball_color not in (*_RANGES.keys(), "auto"):
            raise ValueError(f"Unknown ball_color '{ball_color}'. "
                             f"Choose from: leather, red, white, pink, auto.")
        self.ball_color = ball_color
        self._dilate_px = dilate
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))
        self._resolved: str | None = None

    def detect(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        if self.ball_color == "auto":
            color = self._auto_resolve(hsv)
        else:
            color = self.ball_color
        mask = self._apply_ranges(hsv, _RANGES[color])
        if self._dilate_px > 0:
            mask = cv2.dilate(mask, self._kernel)
        return mask

    def _apply_ranges(self, hsv: np.ndarray, ranges: list[tuple]) -> np.ndarray:
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            combined = cv2.bitwise_or(combined, cv2.inRange(hsv, lo, hi))
        return combined

    def _auto_resolve(self, hsv: np.ndarray) -> str:
        if self._resolved is not None:
            return self._resolved

        total = hsv.shape[0] * hsv.shape[1]
        best_color, best_cov = "leather", -1.0

        for color, ranges in _RANGES.items():
            mask = self._apply_ranges(hsv, ranges)
            cov = np.count_nonzero(mask) / total
            if cov > best_cov:
                best_cov, best_color = cov, color

        if best_cov >= _AUTO_MIN_COVERAGE:
            self._resolved = best_color
        else:
            self._resolved = "leather"  # fallback for outdoor leather ball

        return self._resolved

    @property
    def resolved_color(self) -> str | None:
        return self._resolved if self.ball_color == "auto" else self.ball_color