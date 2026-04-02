import cv2
import numpy as np


def detect_camera_quality(cap: cv2.VideoCapture) -> str:
    """
    Estimate camera quality from Laplacian variance (sharpness proxy).

    Returns one of: "professional", "broadcast", "amateur", "low"
    """
    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    scores = []
    sample_frames = [int(total * p) for p in (0.2, 0.4, 0.6, 0.8)]

    for fi in sample_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        scores.append(score)

    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    if not scores:
        return "unknown"

    avg = float(np.mean(scores))
    print(f"[INFO] Camera quality detected: "
          f"{'professional' if avg > 1000 else 'broadcast' if avg > 300 else 'amateur' if avg > 100 else 'low'}"
          f" (blur score: {avg:.1f})")

    if avg > 1000:
        return "professional"
    if avg > 300:
        return "broadcast"
    if avg > 100:
        return "amateur"
    return "low"


def detect_view(cap: cv2.VideoCapture,
                height: int,
                width: int) -> str:
    """
    Detect whether the video is a side-view or front-view delivery.

    Strategy (in priority order):
    1. If the frame is portrait (height > width) it is almost certainly a
       phone side-view recording → return "side" immediately.
    2. Sample a few frames and compute horizontal vs vertical Sobel energy.
       A side-on pitch has strong horizontal lines (crease, pitch seam);
       a front-on (end-on) view has stronger vertical lines.
    3. Fall back to "side" when uncertain.

    Parameters
    ----------
    cap    : open VideoCapture (position is restored after sampling)
    height : frame height in pixels
    width  : frame width in pixels
    """
    original_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Rule 1: portrait aspect ratio ───────────────────────────────
    # A phone held in portrait while filming cricket from the side gives
    # a tall narrow frame.  This is NEVER a genuine end-on (front) view.
    if height > width * 1.2:
        cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)
        print(f"[INFO] View detected: front (portrait frame {width}x{height})")
        return "front"

    # ── Rule 2: Sobel edge orientation ──────────────────────────────
    h_energies, v_energies = [], []
    sample_positions = [int(total * p) for p in (0.25, 0.45, 0.65)]

    for fi in sample_positions:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        sx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   # vertical edges
        sy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   # horizontal edges
        h_energies.append(float(np.mean(np.abs(sy))))
        v_energies.append(float(np.mean(np.abs(sx))))

    cap.set(cv2.CAP_PROP_POS_FRAMES, original_pos)

    if not h_energies:
        return "side"

    h_mean = float(np.mean(h_energies))
    v_mean = float(np.mean(v_energies))
    ratio  = h_mean / v_mean if v_mean > 0 else 1.0

    # Side view has more horizontal structure (pitch, crease lines)
    # Front view has more vertical structure (stumps, sight-screen edges)
    # Threshold tuned conservatively: only call "front" when strongly vertical
    if ratio < 0.70:
        view = "front"
    else:
        view = "side"

    print(f"[INFO] View detected: {view} (h/v edge ratio: {ratio:.2f})")
    return view