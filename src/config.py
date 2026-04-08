CFG = {
    # ── Background subtractor ───────────────────────────────────────
    "bg_history":       200,
    "bg_var_threshold": 30,

    # ── Ball colour filter ──────────────────────────────────────────
    # CRITICAL: S > 150 and V > 140 required to separate ball from pitch dirt.
    # The pitch surface is H~30, S~80, V~85 — ball is same hue but S>200, V>200.
    # Old range [18,80,80]-[38,255,255] matched the ENTIRE PITCH causing snake trails.
    "ball_color": "yellow",
    "ball_hsv_ranges": [
        ([20, 150, 140], [32, 255, 255]),   # bright saturated yellow ball
    ],

    # ── Ball size ────────────────────────────────────────────────────
    "ball_min_radius_frac": 0.005,
    "ball_max_radius_frac": 0.06,

    # ── Area filter ──────────────────────────────────────────────────
    "ball_min_area_px":  200,    # raise min — pitch noise blobs are tiny
    "ball_max_area_px": 6000,

    # ── Circularity ──────────────────────────────────────────────────
    "circularity_pro":  0.50,
    "circularity_mob":  0.45,    # ball is consistently 0.60+ so 0.45 is safe

    # ── Hough fallback ───────────────────────────────────────────────
    "hough_param1": 40,
    "hough_param2": 12,

    # ── Delivery segmentation ────────────────────────────────────────
    "min_track_frames":    3,
    "delivery_gap_frames": 80,

    # ── Trajectory noise filter ──────────────────────────────────────
    "max_interframe_jump_px": 90,
    "spike_tolerance_px":     55,

    # ── Bounce detection ─────────────────────────────────────────────
    "bounce_reversal_px":  15,
    "min_descent_frames":   5,

    # ── Front view bounce ────────────────────────────────────────────
    "front_bounce_size_jump": 1.25,
    "front_bounce_window":    8,

    # ── ROI ──────────────────────────────────────────────────────────
    "roi_x_min_frac": 0.10,
    "roi_x_max_frac": 0.90,
    "roi_y_min_frac": 0.15,
    "roi_y_max_frac": 0.92,

    # ── Length zones (side view) ──────────────────────────────────────
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
    "color_trail":     (0,   220, 255),
    "color_ball":      (0,   255, 140),
    "color_hud_bg":    (20,   20,  20),
}

LENGTH_COLORS = {
    "Yorker": CFG["color_yorker"],
    "Full":   CFG["color_full"],
    "Good":   CFG["color_good"],
    "Short":  CFG["color_short"],
}