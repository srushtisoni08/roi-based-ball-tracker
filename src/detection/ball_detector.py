import cv2
import numpy as np
from typing import Optional
from src.models.track_point import TrackPoint
from src.config import CFG


class BallDetector:
    def __init__(self, frame_height, frame_width, quality):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=CFG["bg_history"],
            varThreshold=CFG["bg_var_threshold"],
            detectShadows=False,
        )

        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel3      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        self.min_r = max(3, int(CFG["ball_min_radius_frac"] * frame_height))
        self.max_r = int(CFG["ball_max_radius_frac"] * frame_height)
        self.circ_thresh = (CFG["circularity_pro"] if quality == "professional"
                            else CFG["circularity_mob"])

        self.roi_x1 = int(CFG["roi_x_min_frac"] * frame_width)
        self.roi_x2 = int(CFG["roi_x_max_frac"] * frame_width)
        self.roi_y1 = int(CFG["roi_y_min_frac"] * frame_height)
        self.roi_y2 = int(CFG["roi_y_max_frac"] * frame_height)

        self.frame_height = frame_height
        self.frame_width  = frame_width
        self.prev_gray    = None

        self.history: list[TrackPoint] = []
        self.max_history = 6

        self.hough_dp      = 1
        self.hough_minDist = max(10, int(self.min_r * 1.5))

        # Pre-build colour mask params from config
        self.hsv_ranges = [
            (np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            for lo, hi in CFG["ball_hsv_ranges"]
        ]

    # ── Colour mask ───────────────────────────────────────────────
    def _colour_mask(self, frame) -> np.ndarray:
        """
        Returns a binary mask that is 255 wherever the frame pixel
        matches any of the configured ball HSV colour ranges.
        Applied to BOTH the contour path and the Hough fallback so
        nothing bypasses the colour filter.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lo, hi in self.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        # Dilate slightly so motion-blurred edges still pass
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, k, iterations=1)
        return mask

    # ── Motion prediction ─────────────────────────────────────────
    def _predict_next(self) -> Optional[tuple]:
        if len(self.history) < 2:
            return None
        p1, p2 = self.history[-2], self.history[-1]
        return (p2.x + (p2.x - p1.x),
                p2.y + (p2.y - p1.y))

    def _in_roi(self, cx, cy) -> bool:
        return (self.roi_x1 <= cx <= self.roi_x2 and
                self.roi_y1 <= cy <= self.roi_y2)

    def _score(self, circularity, area, cx, cy, pred) -> float:
        score = circularity * np.sqrt(area)
        if pred is not None:
            dist = np.hypot(cx - pred[0], cy - pred[1])
            score *= (1.0 + max(0.0, 1.5 - dist / 80.0))
        return score

    # ── Main detection ────────────────────────────────────────────
    def detect(self, frame) -> Optional[TrackPoint]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Build colour mask ONCE — reused by both methods
        colour = self._colour_mask(frame)

        # Method 1: Background subtraction
        fg = self.bg_sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel_close)

        # Method 2: Adaptive frame differencing
        fd_mask = np.zeros_like(fg)
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            noise_level = np.std(diff[self.roi_y1:self.roi_y2,
                                      self.roi_x1:self.roi_x2])
            thresh_val  = max(8, int(noise_level * 1.3))
            _, fd_mask  = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
            fd_mask     = cv2.morphologyEx(fd_mask, cv2.MORPH_OPEN, self.kernel3)

        self.prev_gray = gray

        # Combine motion masks
        combined = cv2.bitwise_or(fg, fd_mask)

        # Apply ROI mask
        roi_mask = np.zeros_like(combined)
        roi_mask[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2] = 255
        combined = cv2.bitwise_and(combined, roi_mask)

        # ── Apply colour filter — this is the critical step ───────
        # Only pixels that are BOTH moving AND match ball colour survive
        combined = cv2.bitwise_and(combined, colour)

        # ── Contour-based candidates ──────────────────────────────
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple] = []
        pred = self._predict_next()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (CFG["ball_min_area_px"] <= area <= CFG["ball_max_area_px"]):
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx, cy = float(cx), float(cy)
            if not (self.min_r <= radius <= self.max_r):
                continue
            if not self._in_roi(cx, cy):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circularity = (4 * np.pi * area) / (perim ** 2)
            if circularity < self.circ_thresh:
                continue

            score = self._score(circularity, area, cx, cy, pred)
            candidates.append((score, cx, cy, radius))

        # ── Hough fallback — colour-gated ROI only ────────────────
        # IMPORTANT: run Hough on the colour-masked grayscale, NOT raw gray
        # This prevents Hough from finding circles in trees/background
        if not candidates:
            # Mask gray with colour filter before running Hough
            colour_roi = colour[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
            gray_roi   = gray[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
            # Apply: only keep gray values where colour mask is active
            masked_gray = cv2.bitwise_and(gray_roi, colour_roi)
            roi_blur = cv2.GaussianBlur(masked_gray, (5, 5), 1.5)

            circles = cv2.HoughCircles(
                roi_blur,
                cv2.HOUGH_GRADIENT,
                dp=self.hough_dp,
                minDist=self.hough_minDist,
                param1=CFG["hough_param1"],
                param2=CFG["hough_param2"],
                minRadius=self.min_r,
                maxRadius=self.max_r,
            )
            if circles is not None:
                for c in np.uint16(np.around(circles[0])):
                    hx = int(c[0]) + self.roi_x1
                    hy = int(c[1]) + self.roi_y1
                    hr = int(c[2])
                    if not self._in_roi(hx, hy):
                        continue
                    est_area = np.pi * hr * hr
                    score = self._score(0.85, est_area, hx, hy, pred)
                    candidates.append((score, hx, hy, hr))

        if not candidates:
            return None

        _, cx, cy, radius = max(candidates, key=lambda t: t[0])
        tp = TrackPoint(frame=0, x=int(cx), y=int(cy), radius=float(radius))

        self.history.append(tp)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return tp