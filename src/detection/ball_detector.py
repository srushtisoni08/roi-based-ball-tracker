import cv2
import numpy as np
from dataclasses import dataclass
from collections import deque

from .tracker import BallTracker, TrackState
from .color_detector import ColorDetector
from .motion_detector import MotionDetector
from src.config import CFG


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
    """
    Non-AI ball detector using traditional computer vision.
    
    Combines:
    - Background subtraction (MOG2)
    - Contour analysis + circularity filtering
    - Hough Circle Transform for circular object detection
    - Motion detection for added robustness
    """

    def __init__(self,
                 height_or_color=None,
                 width=None,
                 quality=None,
                 **kwargs):
        
        self.height = height_or_color if isinstance(height_or_color, int) else 480
        self.width = width if width is not None else 640
        self.quality = quality or "standard"
        
        # Initialize background subtractor (MOG2)
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            history=CFG.get("bg_history", 200),
            varThreshold=CFG.get("bg_var_threshold", 50)
        )
        
        # Initialize color detector
        self._color_detector = ColorDetector(ball_color="auto", dilate=2)
        
        # Initialize motion detector
        self._motion_detector = MotionDetector(blur_ksize=5, min_area=20)
        
        # Tracker
        self._tracker = BallTracker(
            process_noise=1e-2,
            measurement_noise=5e-2,
            max_missed=12,
            confirm_hits=2,
        )
        
        self._frame_idx = 0
        self._recent: deque = deque(maxlen=8)
        self._roi_mask = None  # Will be created on first frame

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, frame: np.ndarray):
        return self.process_frame(frame)

    def process_frame(self, frame: np.ndarray):
        """
        Detect ball in frame using multi-modal approach:
        1. Background subtraction
        2. Color detection
        3. Motion detection
        4. Contour analysis + circularity filtering
        5. Hough circle detection as fallback
        """
        
        # Create ROI mask on first frame
        if self._roi_mask is None:
            self._roi_mask = self._create_roi_mask(frame)
        
        # Get candidate detections from multiple sources
        candidates = []
        
        # Method 1: Background subtraction + contour analysis
        bg_candidates = self._detect_via_background_subtraction(frame)
        candidates.extend([(c[0], c[1], c[2], 0.7) for c in bg_candidates])
        
        # Method 2: Color-based detection
        color_candidates = self._detect_via_color(frame)
        candidates.extend([(c[0], c[1], c[2], 0.6) for c in color_candidates])
        
        # Method 3: Hough circles (as fallback for edge cases)
        hough_candidates = self._detect_via_hough_circles(frame)
        candidates.extend([(c[0], c[1], c[2], 0.5) for c in hough_candidates])
        
        # Deduplicate and merge nearby detections
        candidates = self._merge_nearby_detections(candidates, merge_distance=15)
        candidates.sort(key=lambda c: -c[3])  # Sort by confidence
        
        # Apply trajectory filter
        candidate = self._trajectory_filter(candidates)
        
        # Update tracker
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
        
        conf = candidate[3] if candidate else 0.0
        
        det = Detection(
            x=int(state.x),
            y=int(state.y),
            radius=state.radius,
            frame_idx=self._frame_idx - 1,
            confidence=conf,
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
        self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=False,
            history=CFG.get("bg_history", 200),
            varThreshold=CFG.get("bg_var_threshold", 50)
        )

    def get_trajectory(self) -> list[tuple]:
        return self._tracker.get_trajectory()

    @property
    def resolved_color(self) -> str | None:
        return self._color_detector.resolved_color

    # ------------------------------------------------------------------
    # ROI & Mask Creation
    # ------------------------------------------------------------------

    def _create_roi_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Create ROI mask from config parameters.
        Returns a binary mask with WHITE (255) in valid region.
        """
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        x_min = int(w * CFG.get("roi_x_min_frac", 0.15))
        x_max = int(w * CFG.get("roi_x_max_frac", 0.85))
        y_min = int(h * CFG.get("roi_y_min_frac", 0.30))
        y_max = int(h * CFG.get("roi_y_max_frac", 0.82))
        
        mask[y_min:y_max, x_min:x_max] = 255
        return mask

    # ------------------------------------------------------------------
    # Detection Methods
    # ------------------------------------------------------------------

    def _detect_via_background_subtraction(self, frame: np.ndarray) -> list[tuple]:
        """
        Detect ball using MOG2 background subtraction.
        Returns: [(cx, cy, radius), ...]
        """
        fg_mask = self._bg_subtractor.apply(frame, learningRate=0.002)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # Apply ROI mask
        if self._roi_mask is not None:
            fg_mask = cv2.bitwise_and(fg_mask, fg_mask, mask=self._roi_mask)
        
        candidates = self._extract_circular_contours(fg_mask)
        return candidates

    def _detect_via_color(self, frame: np.ndarray) -> list[tuple]:
        """
        Detect ball using color information.
        Returns: [(cx, cy, radius), ...]
        """
        color_mask = self._color_detector.detect(frame)
        
        # Apply ROI mask
        if self._roi_mask is not None:
            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=self._roi_mask)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        
        candidates = self._extract_circular_contours(color_mask)
        return candidates

    def _detect_via_hough_circles(self, frame: np.ndarray) -> list[tuple]:
        """
        Detect ball using Hough Circle Transform.
        Returns: [(cx, cy, radius), ...]
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply preprocessing
        blurred = cv2.medianBlur(gray, 5)
        
        # Hough circle detection
        min_r = int(self.height * CFG.get("ball_min_radius_frac", 0.005))
        max_r = int(self.height * CFG.get("ball_max_radius_frac", 0.022))
        
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=50,
            param1=50,
            param2=30,
            minRadius=max(1, min_r),
            maxRadius=max_r
        )
        
        candidates = []
        if circles is not None:
            circles = np.uint16(np.around(circles))
            for i in circles[0, :]:
                cx, cy, radius = int(i[0]), int(i[1]), int(i[2])
                
                # Check if within ROI
                if self._roi_mask is not None and self._roi_mask[cy, cx] == 0:
                    continue
                
                # Verify circularity by checking area consistency
                candidates.append((cx, cy, radius))
        
        return candidates

    def _extract_circular_contours(self, mask: np.ndarray) -> list[tuple]:
        """
        Extract circular objects from binary mask using contour analysis.
        Returns: [(cx, cy, radius), ...]
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        candidates = []
        min_area = CFG.get("ball_min_area_px", 25)
        max_area = CFG.get("ball_max_area_px", 1800)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < min_area or area > max_area:
                continue
            
            # Fit circle to contour
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            
            if radius < 2:
                continue
            
            # Calculate circularity
            circularity = self._calculate_circularity(area, radius)
            
            # Adjust circularity threshold based on quality
            if self.quality == "professional":
                circ_threshold = CFG.get("circularity_pro", 0.55)
            else:
                circ_threshold = CFG.get("circularity_mob", 0.40)
            
            if circularity >= circ_threshold:
                candidates.append((cx, cy, radius))
        
        return candidates

    @staticmethod
    def _calculate_circularity(area: float, radius: float) -> float:
        """
        Calculate circularity of a shape.
        Perfect circle = 1.0
        """
        if radius < 1:
            return 0.0
        theoretical_area = np.pi * radius ** 2
        return min(1.0, area / theoretical_area) if theoretical_area > 0 else 0.0

    def _merge_nearby_detections(self, candidates: list[tuple], 
                                 merge_distance: float = 15) -> list[tuple]:
        """
        Merge detections that are close to each other.
        Returns merged detections with averaged position and radius.
        """
        if not candidates:
            return []
        
        merged = []
        used = set()
        
        for i, (cx1, cy1, r1, conf1) in enumerate(candidates):
            if i in used:
                continue
            
            group = [(cx1, cy1, r1, conf1)]
            used.add(i)
            
            # Find nearby detections to merge
            for j, (cx2, cy2, r2, conf2) in enumerate(candidates[i+1:], start=i+1):
                if j in used:
                    continue
                
                dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
                if dist < merge_distance:
                    group.append((cx2, cy2, r2, conf2))
                    used.add(j)
            
            # Average the group
            avg_cx = np.mean([g[0] for g in group])
            avg_cy = np.mean([g[1] for g in group])
            avg_r = np.mean([g[2] for g in group])
            avg_conf = np.mean([g[3] for g in group])
            
            merged.append((avg_cx, avg_cy, avg_r, avg_conf))
        
        return merged

    # ------------------------------------------------------------------
    # Filtering & Tracking
    # ------------------------------------------------------------------

    def _trajectory_filter(self, candidates: list[tuple]) -> tuple | None:
        """
        Filter candidates based on trajectory consistency.
        """
        if not candidates:
            return None
        
        if len(self._recent) < 2:
            best = candidates[0]
            self._recent.append((self._frame_idx, best[0], best[1], best[2]))
            return best
        
        pred_x, pred_y = self._predict_next()
        MAX_DIST = CFG.get("max_interframe_jump_px", 120)
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
        """Predict next ball position based on trajectory."""
        if len(self._recent) < 2:
            last = self._recent[-1]
            return last[1], last[2]
        recent_list = list(self._recent)
        x1, y1 = recent_list[-2][1], recent_list[-2][2]
        x2, y2 = recent_list[-1][1], recent_list[-1][2]
        return x2 + (x2 - x1), y2 + (y2 - y1)

    def _has_trajectory_confidence(self) -> bool:
        """Check if trajectory is confident enough."""
        if len(self._recent) < 3:
            return False
        recent_list = list(self._recent)
        xs = [p[1] for p in recent_list]
        ys = [p[2] for p in recent_list]
        total_disp = ((xs[-1]-xs[0])**2 + (ys[-1]-ys[0])**2) ** 0.5
        return total_disp >= 10