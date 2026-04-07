import cv2
import numpy as np
from typing import Optional
from src.models.track_point import TrackPoint
from src.config import CFG


class BallDetector:
    def __init__(self, frame_height, frame_width, quality):
        # Background subtractor — moderate history, sensitive threshold
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=CFG["bg_history"],
            varThreshold=CFG["bg_var_threshold"],
            detectShadows=False,
        )

        # Two separate kernels:
        # - small open: removes tiny speckle noise without killing ball blobs
        # - larger close: fills holes in the ball silhouette caused by motion blur
        self.kernel_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        self.kernel3      = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        self.min_r = int(CFG["ball_min_radius_frac"] * frame_height)
        self.max_r = int(CFG["ball_max_radius_frac"] * frame_height)
        self.circ_thresh = (CFG["circularity_pro"] if quality == "professional"
                            else CFG["circularity_mob"])

        # ROI bounds
        self.roi_x1 = int(CFG["roi_x_min_frac"] * frame_width)
        self.roi_x2 = int(CFG["roi_x_max_frac"] * frame_width)
        self.roi_y1 = int(CFG["roi_y_min_frac"] * frame_height)
        self.roi_y2 = int(CFG["roi_y_max_frac"] * frame_height)

        self.frame_height = frame_height
        self.frame_width  = frame_width
        self.prev_gray    = None

        # Short history of confirmed detections for motion prediction
        self.history: list[TrackPoint] = []
        self.max_history = 6

        # Hough params
        self.hough_dp      = 1
        self.hough_minDist = max(10, int(self.min_r * 1.5))

    # ── Motion prediction ─────────────────────────────────────────
    def _predict_next(self) -> Optional[tuple]:
        """
        Linear extrapolation from last 2 confirmed detections.
        Used to bias scoring toward physically plausible positions.
        """
        if len(self.history) < 2:
            return None
        p1, p2 = self.history[-2], self.history[-1]
        return (p2.x + (p2.x - p1.x),
                p2.y + (p2.y - p1.y))

    def _in_roi(self, cx, cy) -> bool:
        return (self.roi_x1 <= cx <= self.roi_x2 and
                self.roi_y1 <= cy <= self.roi_y2)

    def _score(self, circularity, area, cx, cy, pred) -> float:
        """
        Multi-factor candidate score:
          - circularity * sqrt(area): rewards round, well-sized blobs
          - proximity bonus: rewards candidates near predicted position
        """
        score = circularity * np.sqrt(area)
        if pred is not None:
            dist = np.hypot(cx - pred[0], cy - pred[1])
            # Up to 2x bonus for candidates very close to prediction
            score *= (1.0 + max(0.0, 1.5 - dist / 80.0))
        return score

    # ── Main detection ────────────────────────────────────────────
    def detect(self, frame) -> Optional[TrackPoint]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Method 1: Background subtraction
        fg = self.bg_sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel_close)

        # Method 2: Adaptive frame differencing
        fd_mask = np.zeros_like(fg)
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            # Adaptive threshold relative to local noise level
            noise_level = np.std(diff[self.roi_y1:self.roi_y2,
                                      self.roi_x1:self.roi_x2])
            thresh_val  = max(10, int(noise_level * 1.4))
            _, fd_mask  = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
            fd_mask     = cv2.morphologyEx(fd_mask, cv2.MORPH_OPEN, self.kernel3)

        self.prev_gray = gray

        # Combine both motion masks
        combined = cv2.bitwise_or(fg, fd_mask)

        # Apply ROI mask
        roi_mask = np.zeros_like(combined)
        roi_mask[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2] = 255
        combined = cv2.bitwise_and(combined, roi_mask)

        # ── Contour-based candidates ──────────────────────────────
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        candidates: list[tuple] = []   # (score, cx, cy, radius)
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

        # ── Hough fallback — only when contours found nothing ─────
        # HoughCircles is expensive; only invoke it as a fallback so
        # accuracy is preserved without running it on every frame.
        if not candidates:
            roi_gray = gray[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
            roi_blur = cv2.GaussianBlur(roi_gray, (5, 5), 1.5)
            circles  = cv2.HoughCircles(
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
                    # Estimated area from radius
                    est_area = np.pi * hr * hr
                    score = self._score(0.85, est_area, hx, hy, pred)
                    candidates.append((score, hx, hy, hr))

        if not candidates:
            return None

        # Pick highest-scoring candidate
        _, cx, cy, radius = max(candidates, key=lambda t: t[0])
        tp = TrackPoint(frame=0, x=int(cx), y=int(cy), radius=float(radius))

        # Update motion history
        self.history.append(tp)
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return tp