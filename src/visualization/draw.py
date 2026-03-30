import cv2
from config import CFG, LENGTH_COLORS
from models.delivery_result import DeliveryResult

def draw_hud(frame, results_so_far, current_ball_no, total_balls=6):
    """Draw scoreboard HUD in top-left corner."""
    h, w = frame.shape[:2]
    box_w, box_h = 260, 30 + 28 * total_balls + 20
    overlay = frame.copy()
    cv2.rectangle(overlay, (8, 8), (8 + box_w, 8 + box_h),
                  CFG["color_hud_bg"], -1)
    cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
 
    cv2.putText(frame, "DELIVERY ANALYSIS", (16, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
 
    for i, res in enumerate(results_so_far):
        y = 52 + i * 28
        b_label = "BOUNCE" if res.bounced else "FULL TOSS" if res.bounced is False else "?"
        b_color = CFG["color_bounce"] if res.bounced else CFG["color_no_bounce"]
        l_color = LENGTH_COLORS.get(res.length, (200, 200, 200))
 
        cv2.putText(frame, f"Ball {res.ball_no}:", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, b_label, (80, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, b_color, 1, cv2.LINE_AA)
        cv2.putText(frame, res.length, (170, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, l_color, 1, cv2.LINE_AA)
 
 
def draw_ball_trail(frame, trail):
    pts = list(trail)
    for i in range(1, len(pts)):
        alpha = i / len(pts)
        r = int(CFG["color_trail"][2] * alpha)
        g = int(CFG["color_trail"][1] * alpha)
        b = int(CFG["color_trail"][0] * alpha)
        cv2.line(frame, pts[i - 1], pts[i], (b, g, r), 2, cv2.LINE_AA)
 
 
def draw_bounce_marker(frame, pt, label):
    cv2.circle(frame, pt, 12, CFG["color_bounce"], 2, cv2.LINE_AA)
    cv2.circle(frame, pt, 3,  CFG["color_bounce"], -1)
    cv2.putText(frame, "BOUNCE", (pt[0] + 14, pt[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, CFG["color_bounce"], 1, cv2.LINE_AA)
 
 
def draw_length_banner(frame, result: DeliveryResult, frame_no: int, banner_until: dict):
    """Show a length banner for ~60 frames after delivery ends."""
    if result is None:
        return
    # Show banner for 60 frames from end of delivery
    if frame_no > banner_until.get(result.ball_no, 0):
        return
    h, w = frame.shape[:2]
    label = f"Ball {result.ball_no}:  {'BOUNCED' if result.bounced else 'FULL TOSS'}  |  {result.length.upper()}"
    color = LENGTH_COLORS.get(result.length, (200, 200, 200))
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 55), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, label, (w // 2 - len(label) * 7, h - 20),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, color, 2, cv2.LINE_AA)
 
 
def draw_pitch_zones_side(frame):
    """Draw semi-transparent length zone lines on side view."""
    h, w = frame.shape[:2]
    zone_y = int(h * 0.85)
    overlay = frame.copy()
    labels_drawn = set()
    for name, (lo, hi) in CFG["length_zones_side"].items():
        x1 = int(lo * w)
        x2 = int(hi * w)
        color = LENGTH_COLORS.get(name, (200, 200, 200))
        cv2.rectangle(overlay, (x1, zone_y - 8), (x2, zone_y + 8), color, -1)
        mid_x = (x1 + x2) // 2
        cv2.putText(frame, name, (mid_x - 18, zone_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)
 