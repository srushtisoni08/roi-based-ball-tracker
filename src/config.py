CFG = {
    # Background subtractor
    "bg_history":       300,
    "bg_var_threshold": 40,
 
    # Ball size range (radius in px). Script auto-scales to resolution.
    "ball_min_radius_frac": 0.003,   # fraction of frame height
    "ball_max_radius_frac": 0.04,
 
    # Circularity threshold (0–1). Lower = accept less circular blobs.
    # Mobile video is blurrier so we go lower.
    "circularity_pro":  0.50,
    "circularity_mob":  0.30,
 
    # Minimum tracked points to count as a delivery
    "min_track_frames": 12,
 
    # Frame gap that signals end of one delivery / start of next
    "delivery_gap_frames": 35,
 
    # Bounce: how many px Y must reverse to call it a bounce
    "bounce_reversal_px": 4,
 
    # Minimum descent frames before we accept a reversal as bounce
    "min_descent_frames": 5,
 
    # Front view: ball size jump ratio to detect bounce
    # e.g. 1.25 means radius must jump 25% in a few frames
    "front_bounce_size_jump": 1.22,
    "front_bounce_window":    6,   # frames to look ahead for size jump
 
    # Length zones — defined as fraction of frame width (side view)
    # or relative trajectory position (front view).
    # Side view X zones (0.0 = bowler end, 1.0 = batsman end)
    "length_zones_side": {
        "Yorker":   (0.80, 1.00),
        "Full":     (0.60, 0.80),
        "Good":     (0.38, 0.60),
        "Short":    (0.00, 0.38),
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