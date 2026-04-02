"""
dashboard.py
────────────
Terminal-friendly ASCII dashboard + optional OpenCV overlay panel.

Shows real-time tracking stats when --debug is active, and generates a
summary dashboard image that can be embedded in the output video or
saved as a standalone PNG.
"""

import cv2
import numpy as np
from typing import Sequence


# ── Colour palette (BGR) ─────────────────────────────────────────────────────
_BG       = (25, 25, 25)
_ACCENT   = (0, 200, 100)       # green
_WARNING  = (0, 150, 255)       # orange
_ERROR    = (50, 50, 220)       # red
_TEXT     = (230, 230, 230)     # near-white
_SUBTEXT  = (140, 140, 140)
_BOUNCE   = (50, 50, 255)       # red dot for bounce
_TRACK    = (0, 200, 255)       # yellow-ish trajectory


class Dashboard:
    """
    Builds an info panel image that summarises the current delivery.

    Parameters
    ----------
    width, height : int
        Dimensions of the panel in pixels.
    font_scale : float
        cv2 font scale for body text.
    """

    def __init__(self, width: int = 320, height: int = 480,
                 font_scale: float = 0.55):
        self.w = width
        self.h = height
        self.fs = font_scale
        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self._lh = int(height / 18)   # line height

    # ------------------------------------------------------------------

    def render(self,
               frame_idx: int,
               bounce_detected: bool,
               bounce_frame: int | None,
               length: str | None,
               confidence: float | None,
               tracker_active: bool,
               tracker_confirmed: bool,
               missed_frames: int,
               trajectory: list[tuple] | None = None) -> np.ndarray:
        """
        Draw and return the dashboard panel as a BGR numpy array.
        """
        panel = np.full((self.h, self.w, 3), _BG, dtype=np.uint8)

        y = self._lh
        self._header(panel, "CRICKET ANALYZER", y)
        y += self._lh

        self._divider(panel, y)
        y += self._lh // 2 + 4

        # Frame counter
        self._row(panel, "Frame", str(frame_idx), y)
        y += self._lh

        # Tracker status
        t_status = "CONFIRMED" if tracker_confirmed else \
                   ("ACTIVE" if tracker_active else "LOST")
        t_color = _ACCENT if tracker_confirmed else \
                  (_WARNING if tracker_active else _ERROR)
        self._row(panel, "Tracker", t_status, y, val_color=t_color)
        y += self._lh

        self._row(panel, "Missed frm", str(missed_frames), y)
        y += self._lh

        # Bounce
        b_text = f"YES  (fr {bounce_frame})" if bounce_detected else "NO"
        b_color = _BOUNCE if bounce_detected else _SUBTEXT
        self._row(panel, "Bounce", b_text, y, val_color=b_color)
        y += self._lh

        # Length
        l_text = (length.upper() if length else "—")
        l_color = _ACCENT if length else _SUBTEXT
        self._row(panel, "Length", l_text, y, val_color=l_color)
        y += self._lh

        # Confidence
        if confidence is not None:
            conf_pct = f"{confidence*100:.0f}%"
            conf_color = (_ACCENT if confidence > 0.65 else
                          _WARNING if confidence > 0.35 else _ERROR)
            self._row(panel, "Confidence", conf_pct, y, val_color=conf_color)
        y += self._lh

        self._divider(panel, y)
        y += self._lh // 2 + 4

        # Mini trajectory graph
        if trajectory and len(trajectory) > 2:
            self._mini_trajectory(panel, trajectory, y)

        return panel

    # ------------------------------------------------------------------

    def _header(self, img, text, y):
        cv2.putText(img, text, (8, y), self._font,
                    self.fs * 0.9, _ACCENT, 1, cv2.LINE_AA)

    def _row(self, img, label, value, y,
             label_color=_SUBTEXT, val_color=_TEXT):
        cv2.putText(img, f"{label}:", (8, y), self._font,
                    self.fs * 0.8, label_color, 1, cv2.LINE_AA)
        cv2.putText(img, value, (130, y), self._font,
                    self.fs * 0.9, val_color, 1, cv2.LINE_AA)

    def _divider(self, img, y):
        cv2.line(img, (8, y), (self.w - 8, y), _SUBTEXT, 1)

    def _mini_trajectory(self, img, trajectory: list[tuple], y_start: int):
        """Draw a miniature X-Y trajectory graph in the lower panel."""
        max_h = self.h - y_start - 10
        if max_h < 30:
            return

        xs = [p[0] for p in trajectory]
        ys = [p[1] for p in trajectory]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        def norm(val, lo, hi, out_size):
            if hi == lo:
                return out_size // 2
            return int((val - lo) / (hi - lo) * out_size)

        pts = []
        for px, py, _ in trajectory:
            sx = 8 + norm(px, x_min, x_max, self.w - 20)
            sy = y_start + norm(py, y_min, y_max, max_h)
            pts.append((sx, sy))

        for i in range(1, len(pts)):
            cv2.line(img, pts[i - 1], pts[i], _TRACK, 1, cv2.LINE_AA)

        # Mark the latest position
        if pts:
            cv2.circle(img, pts[-1], 3, _ACCENT, -1)

        cv2.putText(img, "Trajectory", (8, y_start - 4),
                    self._font, self.fs * 0.7, _SUBTEXT, 1, cv2.LINE_AA)


# ── ASCII terminal summary ───────────────────────────────────────────────────

def print_summary(video_name: str,
                  bounce_detected: bool,
                  bounce_frame: int | None,
                  length: str | None,
                  confidence: float | None) -> None:
    """Print a compact delivery summary to stdout."""
    sep = "─" * 40
    print(sep)
    print(f"  Video     : {video_name}")
    print(f"  Bounce    : {'YES (frame ' + str(bounce_frame) + ')' if bounce_detected else 'NOT DETECTED'}")
    print(f"  Length    : {length.upper() if length else '—'}")
    if confidence is not None:
        print(f"  Confidence: {confidence:.2f}")
    print(sep)