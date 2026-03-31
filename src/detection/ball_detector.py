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
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.min_r  = int(CFG["ball_min_radius_frac"] * frame_height)
        self.max_r  = int(CFG["ball_max_radius_frac"] * frame_height)
        self.circ_thresh = (CFG["circularity_pro"] if quality == "professional"
                            else CFG["circularity_mob"])
 
    def detect(self, frame) -> Optional[TrackPoint]:
        fg = self.bg_sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel)
 
        contours, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
        best, best_score = None, -1
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 8:
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