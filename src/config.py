CFG = {
    # Background subtractor
    "bg_history":       100,
    "bg_var_threshold": 16,

    # Ball size (fraction of frame height)
    "ball_min_radius_frac": 0.004,
    "ball_max_radius_frac": 0.030,

    # Area filter in pixels²
    # Ball at 478x850 is ~5-20px radius = ~80-1200 px² area
    # A person/arm blob is 5000-50000 px² — this kills those
    "ball_min_area_px": 20,
    "ball_max_area_px": 1200,

    # Circularity
    "circularity_pro":  0.45,
    "circularity_mob":  0.30,

    # Delivery segmentation
    "min_track_frames":     5,
    "delivery_gap_frames":  45,

    # Trajectory noise filter
    "max_interframe_jump_px": 80,

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