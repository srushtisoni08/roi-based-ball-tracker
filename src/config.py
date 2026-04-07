CFG = {
    # ── Background subtractor ───────────────────────────────────────
    # Shorter history = faster warmup; lower varThreshold = more sensitive
    "bg_history":       150,
    "bg_var_threshold": 25,
    # ── Ball colour filter (yellow tennis/soft ball) ─────────────────
    "ball_color": "yellow",
    "ball_hsv_ranges": [
        ([18, 100, 100], [40, 255, 255]),   # pure yellow
    ],

    # ── Ball size (fraction of frame height) ────────────────────────
    "ball_min_radius_frac": 0.004,
    "ball_max_radius_frac": 0.035,   # slightly wider to catch larger apparent size

    # ── Area filter in pixels² ──────────────────────────────────────
    "ball_min_area_px":  15,
    "ball_max_area_px": 1400,

    # ── Circularity thresholds ──────────────────────────────────────
    # Lower thresholds because motion blur makes ball slightly non-circular
    "circularity_pro":  0.40,
    "circularity_mob":  0.52,

    # ── Hough fallback parameters ───────────────────────────────────
    "hough_param1": 50,
    "hough_param2": 13,   # lower = more circles detected (better recall)

    # ── Delivery segmentation ───────────────────────────────────────
    "min_track_frames":    4,    # was 5 — catch shorter visible deliveries
    "delivery_gap_frames": 70,

    # ── Trajectory noise filter ──────────────────────────────────────
    "max_interframe_jump_px": 60,

    # ── Bounce detection (side view) ────────────────────────────────
    "bounce_reversal_px":  2,
    "min_descent_frames":  2,

    # ── Front view bounce ────────────────────────────────────────────
    "front_bounce_size_jump": 1.14,
    "front_bounce_window":    6,

    # ── ROI ──────────────────────────────────────────────────────────
    # Wider vertical range to catch full tosses and short-pitched balls
    "roi_x_min_frac": 0.20,
    "roi_x_max_frac": 0.80,
    "roi_y_min_frac": 0.18,
    "roi_y_max_frac": 0.88,

    # ── Length zones (side view, x-fraction of frame width) ──────────
    "length_zones_side": {
        "Yorker": (0.78, 1.00),
        "Full":   (0.58, 0.78),
        "Good":   (0.36, 0.58),
        "Short":  (0.00, 0.36),
    },

    # ── Annotation colours (BGR) ─────────────────────────────────────
    "color_bounce":    (0,   220,  80),
    "color_no_bounce": (0,   160, 255),
    "color_yorker":    (0,   220,  80),
    "color_full":      (50,  180, 255),
    "color_good":      (0,   200, 255),
    "color_short":     (0,    80, 255),
    "color_trail":     (255, 200,   0),
    "color_ball":      (0,   255, 140),
    "color_hud_bg":    (20,   20,  20),
}

LENGTH_COLORS = {
    "Yorker": CFG["color_yorker"],
    "Full":   CFG["color_full"],
    "Good":   CFG["color_good"],
    "Short":  CFG["color_short"],
}