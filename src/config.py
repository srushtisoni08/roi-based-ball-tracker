import numpy as np

CFG = {
    # ── Background subtractor ──────────────────────────────────────────
    # Lower history: bg model adapts faster, so a new ball registers as
    # foreground immediately instead of being absorbed into the background.
    "bg_history":       200,
    "bg_var_threshold": 50,

    # ── Ball size (fraction of frame height) ───────────────────────────
    "ball_min_radius_frac": 0.005,
    "ball_max_radius_frac": 0.022,   # tighter max: rules out stumps/pads

    # ── Area filter in pixels² ─────────────────────────────────────────
    "ball_min_area_px":  25,
    "ball_max_area_px": 1800,        # tighter than 2500: pads/nets are larger

    # ── Circularity ────────────────────────────────────────────────────
    "circularity_pro":  0.55,        # raised: net mesh blobs are rarely circular
    "circularity_mob":  0.40,

    # ── Delivery segmentation ──────────────────────────────────────────
    # A real delivery at 30fps takes 15–25 frames (fast ball ~0.6s).
    # min_track_frames at 25 was dropping real deliveries. Set to 8.
    # Static blobs are killed separately by the displacement filter below.
    "min_track_frames":    8,
    # 45 frames = 1.5 s gap between deliveries at 30 fps.
    # 90 frames meant consecutive deliveries merged into one track.
    "delivery_gap_frames": 20,

    # ── Trajectory / noise filter ──────────────────────────────────────
    "max_interframe_jump_px": 120,   # ball moves fast; 80 was too strict
    "max_interp_gap_frames":  4,
    "direction_reversal_px":  250,
    "min_travel_before_reversal_px": 150,

    # ── Movement filter (NEW) ──────────────────────────────────────────
    # A valid delivery track must displace at least this many px total.
    # Stationary blobs (pad, foot, stump) score 0–15 px and get dropped.
    "min_total_displacement_px": 80,

    # Minimum standard deviation of x positions across the track.
    # Pure noise / static blobs have near-zero x-spread.
    "min_x_spread_px": 30,

    # ── Bounce detection ───────────────────────────────────────────────
    "bounce_reversal_px":  2,        # more sensitive (was 3)
    "min_descent_frames":  2,        # more sensitive (was 3)

    # ── Front view ─────────────────────────────────────────────────────
    "front_bounce_size_jump": 1.18,
    "front_bounce_window":    8,

    # ── ROI — tighter to exclude bowler run-up and boundary areas ──────
    # Raised y_min: excludes top of frame where stumps at bowler end live.
    # Lowered x margins: excludes wide sides where fielders/umpires walk.
    "roi_x_min_frac": 0.15,
    "roi_x_max_frac": 0.85,
    "roi_y_min_frac": 0.30,
    "roi_y_max_frac": 0.82,

    # ── Length zones (side view, x-fraction of frame width) ───────────
    "length_zones_side": {
        "Yorker": (0.80, 1.00),
        "Full":   (0.60, 0.80),
        "Good":   (0.38, 0.60),
        "Short":  (0.00, 0.38),
    },

    # ── Annotation colours (BGR) ───────────────────────────────────────
    "color_bounce":    (0,  220,  80),
    "color_no_bounce": (0,  160, 255),
    "color_yorker":    (0,  220,  80),
    "color_full":      (50, 180, 255),
    "color_good":      (0,  200, 255),
    "color_short":     (0,   80, 255),
    "color_trail":     (255, 200,   0),
    "color_ball":      (0,  255, 140),
    "color_hud_bg":    (20,  20,  20),
}

LENGTH_COLORS = {
    "Yorker": CFG["color_yorker"],
    "Full":   CFG["color_full"],
    "Good":   CFG["color_good"],
    "Short":  CFG["color_short"],
}


def _smooth(values, window=3):
    half = window // 2
    return [np.mean(values[max(0, i - half): min(len(values), i + half + 1)])
            for i in range(len(values))]