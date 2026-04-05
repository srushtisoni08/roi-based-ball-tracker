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
        self.kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.min_r   = int(CFG["ball_min_radius_frac"] * frame_height)
        self.max_r   = int(CFG["ball_max_radius_frac"] * frame_height)
        self.circ_thresh = (CFG["circularity_pro"] if quality == "professional"
                            else CFG["circularity_mob"])

        # ROI
        self.roi_x1 = int(CFG["roi_x_min_frac"] * frame_width)
        self.roi_x2 = int(CFG["roi_x_max_frac"] * frame_width)
        self.roi_y1 = int(CFG["roi_y_min_frac"] * frame_height)
        self.roi_y2 = int(CFG["roi_y_max_frac"] * frame_height)

        self.frame_height = frame_height
        self.frame_width  = frame_width
        self.prev_gray    = None

    def detect(self, frame) -> Optional[TrackPoint]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Method 1: Background subtraction ──────────────────────
        fg = self.bg_sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel)

        # ── Method 2: Frame difference (catches fast-moving ball) ──
        fd_mask = np.zeros_like(fg)
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, fd_mask = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            fd_mask = cv2.morphologyEx(fd_mask, cv2.MORPH_OPEN, self.kernel3)

        self.prev_gray = gray

        # Combine both masks
        combined = cv2.bitwise_or(fg, fd_mask)

        # ── Apply ROI mask ─────────────────────────────────────────
        mask = np.zeros_like(combined)
        mask[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2] = 255
        combined = cv2.bitwise_and(combined, mask)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        best, best_score = None, -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Strict size filter — player blob is thousands of px²,
            # ball blob is 50–800 px²
            if not (CFG["ball_min_area_px"] <= area <= CFG["ball_max_area_px"]):
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if not (self.min_r <= radius <= self.max_r):
                continue
            perim = cv2.arcLength(cnt, True)
            if perim == 0:
                continue
            circularity = (4 * np.pi * area) / (perim ** 2)
            if circularity < self.circ_thresh:
                continue
            score = circularity * area
            if score > best_score:
                best_score = score
                best = TrackPoint(frame=0, x=int(cx), y=int(cy), radius=radius)

        return best