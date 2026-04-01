import cv2
import numpy as np
from typing import Optional
from src.models.track_point import TrackPoint
from src.config import CFG


class BallDetector:
    def __init__(self, frame_height, frame_width, quality):
        self.frame_height = frame_height
        self.frame_width  = frame_width
        self.quality      = quality

        self.kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.min_r   = int(CFG["ball_min_radius_frac"] * frame_height)
        self.max_r   = int(CFG["ball_max_radius_frac"] * frame_height)
        self.circ_thresh = (CFG["circularity_pro"] if quality == "professional"
                            else CFG["circularity_mob"])

        # ── ROI ───────────────────────────────────────────────────────
        self.roi_x1 = int(CFG["roi_x_min_frac"] * frame_width)
        self.roi_x2 = int(CFG["roi_x_max_frac"] * frame_width)
        self.roi_y1 = int(CFG["roi_y_min_frac"] * frame_height)
        self.roi_y2 = int(CFG["roi_y_max_frac"] * frame_height)

        self.prev_gray = None

        # ── Idle-frame counter for auto bg-subtractor reset ───────────
        self._frames_since_detection = 0
        self._reset_threshold        = CFG["delivery_gap_frames"]

        # ── NEW: Short-window position history for movement scoring ───
        # We keep the last N detected positions so that candidates that
        # appear at the same spot frame after frame score lower than
        # candidates that have been moving consistently.
        self._recent_positions: list[tuple[int, int]] = []   # (x, y)
        self._history_len = 6   # rolling window

        self._build_bg_sub()

    # ── helpers ───────────────────────────────────────────────────────
    def _build_bg_sub(self):
        self.bg_sub = cv2.createBackgroundSubtractorMOG2(
            history=CFG["bg_history"],
            varThreshold=CFG["bg_var_threshold"],
            detectShadows=False,
        )

    def reset_for_next_delivery(self):
        """Explicit reset between deliveries (optional)."""
        self._build_bg_sub()
        self.prev_gray             = None
        self._frames_since_detection = 0
        self._recent_positions     = []

    # ── NEW: movement-consistency score ──────────────────────────────
    def _movement_score(self, cx: int, cy: int) -> float:
        """
        Returns a score in [0, 1] representing how consistently this
        candidate is moving relative to the recent position history.

        Logic:
        - If there's no history yet → return 0.5 (neutral, let it pass).
        - Compute the distance from the candidate to the MEAN of recent
          positions. A static blob will always be close to its own mean,
          so distance ≈ 0 → score near 0.
        - A moving ball will drift away from its rolling mean → score > 0.
        - Normalise against max_interframe_jump_px so the scale is [0,1].
        """
        if len(self._recent_positions) < 2:
            return 0.5

        mean_x = np.mean([p[0] for p in self._recent_positions])
        mean_y = np.mean([p[1] for p in self._recent_positions])

        dist_from_mean = np.hypot(cx - mean_x, cy - mean_y)
        norm = CFG["max_interframe_jump_px"]          # normalisation factor
        return min(dist_from_mean / norm, 1.0)

    # ── main detect method ────────────────────────────────────────────
    def detect(self, frame) -> Optional[TrackPoint]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # ── Auto-reset bg subtractor after long idle gap ──────────────
        if self._frames_since_detection >= self._reset_threshold:
            self._build_bg_sub()
            self.prev_gray         = None
            self._frames_since_detection = 0
            self._recent_positions = []

        # ── Method 1: Background subtraction ─────────────────────────
        fg = self.bg_sub.apply(frame)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  self.kernel)
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, self.kernel)

        # ── Method 2: Frame difference (catches fast-moving ball) ─────
        fd_mask = np.zeros_like(fg)
        if self.prev_gray is not None:
            diff = cv2.absdiff(gray, self.prev_gray)
            _, fd_mask = cv2.threshold(diff, 18, 255, cv2.THRESH_BINARY)
            fd_mask = cv2.morphologyEx(fd_mask, cv2.MORPH_OPEN, self.kernel3)

        self.prev_gray = gray

        # ── Combine + apply ROI ───────────────────────────────────────
        combined = cv2.bitwise_or(fg, fd_mask)
        roi_mask = np.zeros_like(combined)
        roi_mask[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2] = 255
        combined = cv2.bitwise_and(combined, roi_mask)

        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # ── Score every candidate contour ─────────────────────────────
        best, best_score = None, -1

        for cnt in contours:
            area = cv2.contourArea(cnt)
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

            # ── NEW: incorporate movement score into final ranking ────
            # score = circularity × area × movement_bonus
            # A static blob with circularity=0.9, area=400 but movement=0.0
            # scores 0, while a moving ball with circ=0.6, area=300,
            # movement=0.8 scores 144 and wins.
            movement = self._movement_score(int(cx), int(cy))
            score    = circularity * area * (1.0 + movement)  # movement is a bonus

            if score > best_score:
                best_score = score
                best = TrackPoint(frame=0, x=int(cx), y=int(cy), radius=radius)

        # ── Update position history and idle counter ──────────────────
        if best is None:
            self._frames_since_detection += 1
        else:
            self._frames_since_detection = 0
            self._recent_positions.append((best.x, best.y))
            if len(self._recent_positions) > self._history_len:
                self._recent_positions.pop(0)

        return best