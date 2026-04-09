CFG = {
    # ── Background subtractor ───────────────────────────────────────
    "bg_history":       200,
    "bg_var_threshold": 22,

    # ── Ball colour filter ──────────────────────────────────────────
    # Pixel-confirmed ball HSV: H=27-30, S=120-210, V=140-251 (clear frames)
    # Motion-blurred frames shift to: H=14-45, S=80-110, V=80-130
    # Two ranges: tight for clear frames, wider for blurred/fast frames
    "ball_color": "yellow",
    "ball_hsv_ranges": [
        ([22, 80, 80],  [45, 255, 255]),   # wider — catches ball in all motion states
    ],

    # ── Ball size ────────────────────────────────────────────────────
    "ball_min_radius_frac": 0.004,   # lowered — catches ball when small/far
    "ball_max_radius_frac": 0.065,

    # ── Area filter ──────────────────────────────────────────────────
    "ball_min_area_px":   30,    # lowered — ball can be small when far away
    "ball_max_area_px": 8000,

    # ── Circularity ──────────────────────────────────────────────────
    "circularity_pro":  0.40,
    "circularity_mob":  0.35,    # motion blur makes ball elliptical

    # ── Hough fallback ───────────────────────────────────────────────
    "hough_param1": 40,
    "hough_param2": 12,

    # ── Delivery segmentation ────────────────────────────────────────
    "min_track_frames":    3,
    "delivery_gap_frames": 80,

    # ── Trajectory noise filter ──────────────────────────────────────
    "max_interframe_jump_px": 90,
    "spike_tolerance_px":     55,

    # ── Stationarity filter ──────────────────────────────────────────
    "stationary_frame_threshold": 6,
    "stationary_pixel_radius":    10,

    # ── Bounce detection ─────────────────────────────────────────────
    # At 30fps a real ball bounce takes 3-4 frames of descent, not 5
    "bounce_reversal_px":  12,   # was 15 — ball only dips 24px, 15 was too close to margin
    "min_descent_frames":   3,   # was 5 — at 30fps bounce happens in 3-4 frames

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