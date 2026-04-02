# save as track_ball.py
import cv2

cap = cv2.VideoCapture("data/input_videos/long.mp4")
prev = None
bg = cv2.createBackgroundSubtractorMOG2(history=30, varThreshold=25)

# Prime with first 80 frames
for i in range(80):
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, f = cap.read()
    if ret:
        bg.apply(f)

# Now process frames 85-200 and save every frame as image
import os
os.makedirs("data/debug_frames", exist_ok=True)

for fi in range(85, 200):
    cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
    ret, frame = cap.read()
    if not ret:
        continue

    fg = bg.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel)

    vis = frame.copy()
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        a = cv2.contourArea(c)
        if 15 < a < 400:
            (cx, cy), r = cv2.minEnclosingCircle(c)
            p = cv2.arcLength(c, True)
            circ = 4*3.14159*a/(p**2) if p > 0 else 0
            if circ > 0.55 and r < 18:
                cv2.circle(vis, (int(cx), int(cy)), int(r)+3, (0,255,0), 2)

    cv2.imwrite(f"data/debug_frames/frame_{fi:04d}.jpg", vis)

cap.release()
print("Saved frames 85-200 to data/debug_frames/")