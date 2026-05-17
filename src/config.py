CFG = {
    # ── Background subtractor ───────────────────────────────────────
    "bg_history":       200,
    "bg_var_threshold": 22,

    # ── Ball colour filter ──────────────────────────────────────────
    # Three HSV ranges covering:
    #   1) Yellow/lime ball (clear frames) — wide for all motion states
    #   2) Orange-yellow for shade / indoor / underexposed lighting
    #   3) White/light ball and overexposed highlights
    "ball_color": "yellow",
    "ball_hsv_ranges": [
        ([18, 60,  70],  [45, 255, 255]),   # yellow — main range
        ([10, 40,  60],  [22, 220, 200]),   # orange-yellow — shade/indoor
        ([0,  0,  200],  [180, 40, 255]),   # white/overexposed highlight
    ],

    # ── Ball size ────────────────────────────────────────────────────
    "ball_min_radius_frac": 0.004,
    "ball_max_radius_frac": 0.065,

    # ── Area filter ──────────────────────────────────────────────────
    "ball_min_area_px":   25,
    "ball_max_area_px": 8000,

    # ── Circularity ──────────────────────────────────────────────────
    "circularity_pro":  0.38,    # slightly relaxed — real ball at speed is elliptical
    "circularity_mob":  0.30,

    # ── Hough fallback ───────────────────────────────────────────────
    "hough_param1": 38,
    "hough_param2": 10,          # lower = more sensitive in fallback

    # ── Delivery segmentation ────────────────────────────────────────
    "min_track_frames":    12,
    "delivery_gap_frames": 200,

    # ── Trajectory noise filter ──────────────────────────────────────
    "max_interframe_jump_px": 80,
    "spike_tolerance_px":     40,

    # ── Stationarity filter ──────────────────────────────────────────
    "stationary_frame_threshold": 4,
    "stationary_pixel_radius":    12,

    # ── Bounce detection ─────────────────────────────────────────────
    "bounce_reversal_px":  25,
    "min_descent_frames":   6,

    # ── Front view bounce ────────────────────────────────────────────
    "front_bounce_size_jump": 1.25,
    "front_bounce_window":    8,

    # ── ROI ──────────────────────────────────────────────────────────
    "roi_x_min_frac": 0.08,     # slightly wider — don't clip edge deliveries
    "roi_x_max_frac": 0.92,
    "roi_y_min_frac": 0.12,
    "roi_y_max_frac": 0.94,

    # ── Trail interpolation ──────────────────────────────────────────
    # Fill gaps in detected track up to this many frames with interpolated pts
    "trail_max_interp_gap": 15,

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