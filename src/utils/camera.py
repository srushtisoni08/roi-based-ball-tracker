import cv2
import numpy as np


def detect_camera_quality(cap):
    blurs = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurs.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    avg_blur = np.mean(blurs) if blurs else 0
    quality  = "professional" if avg_blur > 180 else "mobile"
    print(f"[INFO] Camera quality: {quality} (blur score: {avg_blur:.1f})")
    return quality


def detect_view(cap, height, width):
    """
    Determines whether the video is side-view or front-view.

    Fixed logic for portrait phone videos:
    - Portrait (height > width): phone held vertically pointing down pitch
      → ball travels left-right in X = SIDE view
    - Landscape (width > height): broadcast camera or phone landscape
      → use edge heuristic to distinguish side vs front

    The old edge-ratio heuristic was wrong for portrait videos —
    it returned 'front' for portrait phone footage where the ball
    actually moves laterally (side view).
    """
    # Portrait phone video: height > width → bowler-end side view
    if height > width:
        print(f"[INFO] Portrait video ({width}x{height}) → side view")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return "side"

    # Landscape: use edge heuristic
    samples = []
    for _ in range(8):
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        lower = edges[height // 2:, :]
        horiz = np.sum(lower[:-1] & lower[1:])
        vert  = np.sum(lower[:, :-1] & lower[:, 1:])
        samples.append(horiz / (vert + 1))

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ratio = np.mean(samples) if samples else 1.0
    view  = "side" if ratio > 1.2 else "front"
    print(f"[INFO] View detected: {view} (h/v edge ratio: {ratio:.2f})")
    return view