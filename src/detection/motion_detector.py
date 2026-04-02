import cv2
import numpy as np


class MotionDetector:
    """
    Produces a binary motion mask from consecutive frames.

    Usage
    -----
        md = MotionDetector(blur_ksize=5, min_area=30)
        mask = md.update(frame)   # call once per frame
    """

    def __init__(self, blur_ksize: int = 5, min_area: int = 30):
        """
        Parameters
        ----------
        blur_ksize : int
            Kernel size for Gaussian blur applied before differencing.
            Must be odd. Larger → less sensitive to pixel noise.
        min_area : int
            Contours smaller than this area (px²) are ignored when
            returning candidate regions.  Helps filter sensor noise.
        """
        if blur_ksize % 2 == 0:
            blur_ksize += 1          # enforce odd
        self.blur_ksize = blur_ksize
        self.min_area = min_area
        self._prev_gray: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, frame: np.ndarray) -> np.ndarray:
        """
        Process one frame and return a binary motion mask (uint8, 0/255).

        The very first call always returns an all-zero mask because there
        is no previous frame to difference against.

        Parameters
        ----------
        frame : np.ndarray
            BGR frame from cv2.VideoCapture.

        Returns
        -------
        np.ndarray
            Single-channel binary mask, same HxW as *frame*.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_ksize, self.blur_ksize), 0)

        if self._prev_gray is None:
            self._prev_gray = gray
            return np.zeros(gray.shape, dtype=np.uint8)

        diff = cv2.absdiff(self._prev_gray, gray)
        self._prev_gray = gray

        # Otsu threshold – adapts automatically to the scene brightness
        _, mask = cv2.threshold(diff, 0, 255,
                                cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        # Morphological close to fill small gaps inside a moving ball
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        return mask

    def candidate_regions(self, mask: np.ndarray) -> list[tuple]:
        """
        Return bounding boxes of significant motion regions.

        Parameters
        ----------
        mask : np.ndarray
            Binary mask from :meth:`update`.

        Returns
        -------
        list of (x, y, w, h) tuples sorted by area descending.
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        regions = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_area:
                regions.append(cv2.boundingRect(cnt))

        # largest regions first so callers can short-circuit early
        regions.sort(key=lambda r: r[2] * r[3], reverse=True)
        return regions

    def reset(self) -> None:
        """Clear state (call between deliveries / video resets)."""
        self._prev_gray = None