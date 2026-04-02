import cv2
import numpy as np
from dataclasses import dataclass, field

@dataclass
class TrackState:
    """What the tracker reports after each frame."""
    x: float
    y: float
    radius: float
    vx: float = 0.0
    vy: float = 0.0
    confirmed: bool = False      # True once the track has been seen N times
    age: int = 0                 # frames since track was initialised
    missed: int = 0              # consecutive frames with no detection
    history: list = field(default_factory=list)   # [(x, y, r), ...]

class BallTracker:

    def __init__(self,
                 process_noise: float = 1e-2,
                 measurement_noise: float = 1e-1,
                 max_missed: int = 8,
                 confirm_hits: int = 3,
                 gate_sigma: float = 4.0):

        self.max_missed = max_missed
        self.confirm_hits = confirm_hits
        self.gate_sigma = gate_sigma

        # ── Build the Kalman filter ──────────────────────────────────
        # State:       [x, y, vx, vy, r, vr]   (6 dims)
        # Measurement: [x, y, r]                (3 dims)
        self._kf = cv2.KalmanFilter(6, 3)

        dt = 1.0  # one frame

        # Transition matrix (constant velocity)
        self._kf.transitionMatrix = np.array([
            [1, 0, dt, 0,  0, 0],
            [0, 1, 0,  dt, 0, 0],
            [0, 0, 1,  0,  0, 0],
            [0, 0, 0,  1,  0, 0],
            [0, 0, 0,  0,  1, dt],
            [0, 0, 0,  0,  0, 1],
        ], dtype=np.float32)

        # Measurement matrix (we observe x, y, r directly)
        self._kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ], dtype=np.float32)

        self._kf.processNoiseCov = np.eye(6, dtype=np.float32) * process_noise
        self._kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        self._kf.errorCovPost = np.eye(6, dtype=np.float32)

        self._initialized = False
        self._hits = 0
        self._state = TrackState(x=0, y=0, radius=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detection: tuple | None) -> TrackState:
        if not self._initialized:
            if detection is None:
                return self._state
            self._initialize(detection)
            return self._state

        # ── Predict ────────────────────────────────────────────────
        predicted = self._kf.predict()
        px, py = float(predicted[0]), float(predicted[1])
        pr = float(predicted[4])

        # ── Associate / correct ────────────────────────────────────
        if detection is not None and self._in_gate(detection, px, py, pr):
            meas = np.array([[detection[0]],
                             [detection[1]],
                             [detection[2]]], dtype=np.float32)
            corrected = self._kf.correct(meas)
            cx = float(corrected[0])
            cy = float(corrected[1])
            cvx = float(corrected[2])
            cvy = float(corrected[3])
            cr = max(1.0, float(corrected[4]))

            self._hits += 1
            self._state.missed = 0
        else:
            # coast on prediction
            cx, cy, cr = px, py, max(1.0, pr)
            cvx = float(predicted[2])
            cvy = float(predicted[3])
            self._hits = max(0, self._hits - 1)
            self._state.missed += 1

        # ── Update public state ────────────────────────────────────
        self._state.x = cx
        self._state.y = cy
        self._state.vx = cvx
        self._state.vy = cvy
        self._state.radius = cr
        self._state.confirmed = self._hits >= self.confirm_hits
        self._state.age += 1
        self._state.history.append((cx, cy, cr))

        return self._state

    @property
    def is_active(self) -> bool:
        """True while the tracker hasn't exceeded max_missed."""
        return self._state.missed <= self.max_missed

    @property
    def is_confirmed(self) -> bool:
        return self._state.confirmed

    def reset(self) -> None:
        """Full reset – call between deliveries."""
        self._initialized = False
        self._hits = 0
        self._state = TrackState(x=0, y=0, radius=0)
        self._kf.errorCovPost = np.eye(6, dtype=np.float32)

    def get_trajectory(self) -> list[tuple]:
        """Return list of (x, y, r) for the full track history."""
        return list(self._state.history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialize(self, detection: tuple) -> None:
        x, y, r = detection
        state = np.array([x, y, 0, 0, r, 0], dtype=np.float32).reshape(6, 1)
        self._kf.statePost = state
        self._initialized = True
        self._hits = 1
        self._state = TrackState(x=x, y=y, radius=r, confirmed=False)
        self._state.history.append((x, y, r))

    def _in_gate(self,
                 detection: tuple,
                 px: float, py: float, pr: float) -> bool:
        """Simple Euclidean gate (fast, no full Mahalanobis needed here)."""
        dx = detection[0] - px
        dy = detection[1] - py
        dist = (dx**2 + dy**2) ** 0.5
        # gate radius scales with predicted ball radius and sigma
        gate_radius = pr * self.gate_sigma + 20   # +20 px floor
        return dist < gate_radius