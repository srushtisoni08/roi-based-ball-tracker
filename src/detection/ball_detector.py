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

        self.min_r = max(4, int(CFG["ball_min_radius_frac"] * frame_height))
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

        # Longer history = better velocity prediction
        self.history: list[TrackPoint] = []
        self.max_history = 10

        self.hough_dp      = 1
        self.hough_minDist = max(10, int(self.min_r * 1.5))

        self.hsv_ranges = [
            (np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
            for lo, hi in CFG["ball_hsv_ranges"]
        ]

    # ── Colour mask ───────────────────────────────────────────────
    def _colour_mask(self, frame) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for lo, hi in self.hsv_ranges:
            mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lo, hi))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.dilate(mask, k, iterations=1)
        return mask

    # ── Velocity-weighted prediction ─────────────────────────────
    def _predict_next(self) -> Optional[tuple]:
        """
        Weighted average of recent velocity vectors.
        More recent frames get higher weight — gives better prediction
        for curved ball trajectories than simple linear extrapolation.
        """
        if len(self.history) < 2:
            return None

        # Use up to last 4 points for velocity estimate
        pts = self.history[-4:]
        if len(pts) < 2:
            return None

        # Weighted velocities — most recent gets weight 4, oldest gets 1
        vx_sum, vy_sum, w_sum = 0.0, 0.0, 0.0
        for i in range(1, len(pts)):
            w = float(i)  # more recent = higher weight
            vx_sum += w * (pts[i].x - pts[i-1].x)
            vy_sum += w * (pts[i].y - pts[i-1].y)
            w_sum  += w

        vx = vx_sum / w_sum
        vy = vy_sum / w_sum
        last = self.history[-1]
        return (last.x + vx, last.y + vy)

    def _in_roi(self, cx, cy) -> bool:
        return (self.roi_x1 <= cx <= self.roi_x2 and
                self.roi_y1 <= cy <= self.roi_y2)

    def _score(self, circularity, area, cx, cy, pred) -> float:
        """
        Score = circularity × √area × trajectory_bonus
        Trajectory bonus: up to 3× for candidates near predicted position.
        Candidates far from prediction get penalised (0.2× if >200px away).
        """
        score = circularity * np.sqrt(area)
        if pred is not None:
            dist = np.hypot(cx - pred[0], cy - pred[1])
            if dist < 30:
                # Very close to prediction — strong bonus
                score *= 3.0
            elif dist < 80:
                score *= (1.0 + max(0.0, 2.0 - dist / 40.0))
            elif dist > 200:
                # Far from prediction — likely a false positive
                score *= 0.2
        return score

    # ── Main detection ────────────────────────────────────────────
    def detect(self, frame) -> Optional[TrackPoint]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        colour = self._colour_mask(frame)

        # Method 1: Background subtraction
        fg = self.bg_sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel_open)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel_close)

        # Method 2: Frame differencing
        fd_mask = np.zeros_like(fg)
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            noise_level = np.std(diff[self.roi_y1:self.roi_y2,
                                      self.roi_x1:self.roi_x2])
            thresh_val  = max(8, int(noise_level * 1.3))
            _, fd_mask  = cv2.threshold(diff, thresh_val, 255, cv2.THRESH_BINARY)
            fd_mask     = cv2.morphologyEx(fd_mask, cv2.MORPH_OPEN, self.kernel3)

        self.prev_gray = gray

        combined = cv2.bitwise_or(fg, fd_mask)

        roi_mask = np.zeros_like(combined)
        roi_mask[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2] = 255
        combined = cv2.bitwise_and(combined, roi_mask)

        # Colour gate — must match ball colour AND be moving
        combined = cv2.bitwise_and(combined, colour)

        # ── Contour candidates ────────────────────────────────────
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

        # ── Hough fallback — colour-gated ─────────────────────────
        if not candidates:
            colour_roi  = colour[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
            gray_roi    = gray[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
            masked_gray = cv2.bitwise_and(gray_roi, colour_roi)
            roi_blur    = cv2.GaussianBlur(masked_gray, (5, 5), 1.5)

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