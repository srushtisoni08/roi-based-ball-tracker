import cv2
import numpy as np

# ── HSV colour ranges ────────────────────────────────────────────────────────
# Each entry is a list of (lower, upper) pairs so we can union multiple bands
# (red wraps around hue 0/180 in OpenCV's 0-179 scale).

_RANGES: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {
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

# Minimum fraction of the mask that must be non-zero for auto-detection
_AUTO_MIN_COVERAGE = 0.0002   # 0.02 % of frame pixels


class ColorDetector:
    """
    Produces a binary mask for pixels matching a cricket-ball colour.

    Parameters
    ----------
    ball_color : str
        One of "red", "white", "pink", "auto".
    dilate : int
        How many pixels to dilate the resulting mask.  Helps the mask
        cover the full ball even when specular highlights desaturate the
        centre pixels.
    """

    def __init__(self, ball_color: str = "auto", dilate: int = 3):
        if ball_color not in (*_RANGES.keys(), "auto"):
            raise ValueError(f"Unknown ball_color '{ball_color}'. "
                             f"Choose from: red, white, pink, auto.")
        self.ball_color = ball_color
        self._dilate_px = dilate
        self._kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (dilate * 2 + 1, dilate * 2 + 1))

        # cache resolved color after first auto-detect call
        self._resolved: str | None = None

    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> np.ndarray:
        """
        Return a binary colour mask for *frame* (uint8, 0/255).

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from cv2.VideoCapture.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        if self.ball_color == "auto":
            color = self._auto_resolve(hsv)
        else:
            color = self.ball_color

        mask = self._apply_ranges(hsv, _RANGES[color])

        if self._dilate_px > 0:
            mask = cv2.dilate(mask, self._kernel)

        return mask

    # ------------------------------------------------------------------
    def _apply_ranges(self,
                      hsv: np.ndarray,
                      ranges: list[tuple]) -> np.ndarray:
        combined = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            combined = cv2.bitwise_or(combined,
                                      cv2.inRange(hsv, lo, hi))
        return combined

    def _auto_resolve(self, hsv: np.ndarray) -> str:
        """Pick the colour whose mask covers the most pixels."""
        if self._resolved is not None:
            return self._resolved

        total = hsv.shape[0] * hsv.shape[1]
        best_color, best_cov = "red", -1.0

        for color, ranges in _RANGES.items():
            mask = self._apply_ranges(hsv, ranges)
            cov = np.count_nonzero(mask) / total
            if cov > best_cov:
                best_cov, best_color = cov, color

        if best_cov >= _AUTO_MIN_COVERAGE:
            self._resolved = best_color
        else:
            # fallback – couldn't confidently detect, default to red
            self._resolved = "red"

        return self._resolved

    @property
    def resolved_color(self) -> str | None:
        """The ball colour that was auto-detected, or None if not yet run."""
        return self._resolved if self.ball_color == "auto" else self.ball_color