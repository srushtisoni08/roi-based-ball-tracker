CFG = {
    # ── Background subtractor ───────────────────────────────────────
    "bg_history":       200,
    "bg_var_threshold": 30,

    # ── Ball colour filter ──────────────────────────────────────────
    "ball_color": "yellow",
    "ball_hsv_ranges": [
        ([18, 80, 80],  [38, 255, 255]),
        ([15, 60, 120], [42, 255, 255]),
    ],

    # ── Ball size (fraction of frame height) ────────────────────────
    "ball_min_radius_frac": 0.003,
    "ball_max_radius_frac": 0.06,

    # ── Area filter in pixels² ──────────────────────────────────────
    "ball_min_area_px":   20,
    "ball_max_area_px": 8000,

    # ── Circularity thresholds ──────────────────────────────────────
    "circularity_pro":  0.45,
    "circularity_mob":  0.35,

    # ── Hough fallback parameters ───────────────────────────────────
    "hough_param1": 40,
    "hough_param2": 12,

    # ── Delivery segmentation ───────────────────────────────────────
    "min_track_frames":    3,
    "delivery_gap_frames": 80,

    # ── Trajectory noise filter ──────────────────────────────────────
    "max_interframe_jump_px": 90,
    "spike_tolerance_px":     55,

    # ── Bounce detection ─────────────────────────────────────────────
    # These were 2px / 2 frames — far too sensitive, any noise = false bounce
    # A real bounce reversal is at minimum 15px at pitch level
    "bounce_reversal_px":  15,   # was 2 — noise-level, caused false bounces
    "min_descent_frames":   5,   # was 2 — need sustained descent, not 2-frame dip

    # ── Front view bounce ─────────────────────────────────────────────
    # Signal B (radius burst): removed — ball shrinks when moving away from camera
    # (bowler POV portrait video). Radius growth signal is wrong for this setup.
    # Kept for landscape broadcast cameras where ball approaches camera.
    "front_bounce_size_jump": 1.25,   # raised — only trigger on very clear growth
    "front_bounce_window":    8,

    # ── ROI ──────────────────────────────────────────────────────────
    "roi_x_min_frac": 0.10,
    "roi_x_max_frac": 0.90,
    "roi_y_min_frac": 0.15,
    "roi_y_max_frac": 0.92,

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