import cv2
import numpy as np


def get_roi_mask(frame: np.ndarray) -> np.ndarray:
    """
    Returns a binary mask that is WHITE (255) only in the region where
    the ball is expected to travel — blocking out the net and edges.

    Tuned for behind-bowler portrait video (478x850):
      - Block top 15% (sky/trees, no ball there early)
      - Block right 40% (net mesh on right side)
      - Block left 5% (edge noise)
      - Keep the central corridor where the ball travels
    """
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)

    # Define the corridor where the ball actually travels
    x1 = int(w * 0.05)   # 5% from left
    x2 = int(w * 0.60)   # 60% from left (cut out right-side net)
    y1 = int(h * 0.10)   # 10% from top
    y2 = int(h * 0.90)   # 90% from bottom

    mask[y1:y2, x1:x2] = 255
    return mask