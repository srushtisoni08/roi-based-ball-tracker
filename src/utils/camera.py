import cv2
import numpy as np

def detect_camera_quality(cap):
    """
    Reads a few frames and measures blur (Laplacian variance).
    Low variance = mobile/blurry. High = professional.
    """
    blurs = []
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurs.append(cv2.Laplacian(gray, cv2.CV_64F).var())
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)   # rewind
    avg_blur = np.mean(blurs) if blurs else 0
    quality  = "professional" if avg_blur > 180 else "mobile"
    print(f"[INFO] Camera quality detected: {quality} (blur score: {avg_blur:.1f})")
    return quality
 

def detect_view(cap, height, width):
    """
    Heuristic: in a side view the pitch is horizontal — there will be a
    prominent horizontal edge (the pitch strip) in the lower third.
    In a front view the pitch recedes → strong vertical perspective lines.
    """
    samples = []
    for _ in range(8):
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        # Count horizontal vs vertical edges in the lower half
        lower = edges[height // 2:, :]
        horiz = np.sum(lower[:-1] & lower[1:])      # vertically adjacent
        vert  = np.sum(lower[:, :-1] & lower[:, 1:]) # horizontally adjacent
        samples.append(horiz / (vert + 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ratio = np.mean(samples)
    view  = "side" if ratio > 1.2 else "front"
    print(f"[INFO] View detected: {view} (h/v edge ratio: {ratio:.2f})")
    return view