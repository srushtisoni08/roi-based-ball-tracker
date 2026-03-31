CFG = {
    # Background subtractor
    # FIX 1: Increased history from 100→500 so the bg model doesn't "learn"
    # the ball's path and turn subsequent balls invisible.
    "bg_history":       500,
    "bg_var_threshold": 40,   # raised from 16 → fewer false-negatives on ball

    # Ball size (fraction of frame height)
    "ball_min_radius_frac": 0.004,
    "ball_max_radius_frac": 0.030,

    # Area filter in pixels²
    # FIX 2: Raised max area 1200→2500 to handle balls that appear larger
    # when the bowler is closer to the camera, or slight blur enlarges blobs.
    "ball_min_area_px": 20,
    "ball_max_area_px": 2500,

    # Circularity
    "circularity_pro":  0.45,
    "circularity_mob":  0.30,

    # Delivery segmentation
    # Raised 5 → 25: real delivery at 30fps lasts ~1s minimum (~30 frames).
    # Noise blobs typically get 5-15 hits — this kills them all.
    "min_track_frames":     25,
    "delivery_gap_frames":  90,

    # Trajectory noise filter
    "max_interframe_jump_px":  80,
    "max_interp_gap_frames":   4,
    # Raised 120 → 300: ball travels 150-200px/frame at pace, so 120px was
    # firing on normal ball motion. 300px only triggers on a true positional
    # teleport (i.e. a completely new delivery starting from bowler end).
    "direction_reversal_px":   300,
    # Must have traveled this far in one direction before a reversal counts.
    # Prevents mid-delivery jitter from being mistaken for a new delivery.
    "min_travel_before_reversal_px": 200,

    # Bounce detection
    "bounce_reversal_px":  3,
    "min_descent_frames":  3,

    # Front view
    "front_bounce_size_jump": 1.18,
    "front_bounce_window":    8,

    # ROI — pitch area only
    "roi_x_min_frac": 0.08,
    "roi_x_max_frac": 0.92,
    "roi_y_min_frac": 0.25,
    "roi_y_max_frac": 0.80,

    # Length zones (side view)
    "length_zones_side": {
        "Yorker": (0.80, 1.00),
        "Full":   (0.60, 0.80),
        "Good":   (0.38, 0.60),
        "Short":  (0.00, 0.38),
    },

    # Annotation colours (BGR)
    "color_bounce":    (0,  220, 80),
    "color_no_bounce": (0,  160, 255),
    "color_yorker":    (0,  220, 80),
    "color_full":      (50, 180, 255),
    "color_good":      (0,  200, 255),
    "color_short":     (0,  80,  255),
    "color_trail":     (255, 200, 0),
    "color_ball":      (0,  255, 140),
    "color_hud_bg":    (20,  20,  20),
}

LENGTH_COLORS = {
    "Yorker": CFG["color_yorker"],
    "Full":   CFG["color_full"],
    "Good":   CFG["color_good"],
    "Short":  CFG["color_short"],
}

import numpy as np

def _smooth(values, window=3):
    half = window // 2
    return [np.mean(values[max(0, i - half): min(len(values), i + half + 1)])
            for i in range(len(values))]