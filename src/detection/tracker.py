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
    confirmed: bool = False
    age: int = 0
    missed: int = 0
    history: list = field(default_factory=list)

class BallTracker:
    """
    Single-object Kalman tracker for the cricket ball.

    Parameters
    ----------
    process_noise     : Q diagonal scale
    measurement_noise : R diagonal scale
    max_missed        : frames without detection before track is dropped
    confirm_hits      : consecutive detections needed to confirm track
    gate_sigma        : Euclidean gate multiplier on predicted radius
    """

    def __init__(self,
                 process_noise: float = 1e-2,
                 measurement_noise: float = 1e-1,
                 max_missed: int = 8,
                 confirm_hits: int = 3,
                 gate_sigma: float = 4.0):

        self.max_missed    = max_missed
        self.confirm_hits  = confirm_hits
        self.gate_sigma    = gate_sigma

        # ── Build cv2 Kalman filter ──────────────────────────────
        self._kf = cv2.KalmanFilter(6, 3)
        dt = 1.0

        self._kf.transitionMatrix = np.array([
            [1, 0, dt, 0,  0,  0],
            [0, 1, 0,  dt, 0,  0],
            [0, 0, 1,  0,  0,  0],
            [0, 0, 0,  1,  0,  0],
            [0, 0, 0,  0,  1, dt],
            [0, 0, 0,  0,  0,  1],
        ], dtype=np.float32)

        self._kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
        ], dtype=np.float32)

        self._kf.processNoiseCov     = np.eye(6, dtype=np.float32) * process_noise
        self._kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * measurement_noise
        self._kf.errorCovPost        = np.eye(6, dtype=np.float32)

        self._initialized = False
        self._hits = 0
        self._state = TrackState(x=0, y=0, radius=0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(self, detection: tuple | None) -> TrackState:
        """
        Advance tracker by one frame.

        Parameters
        ----------
        detection : (x, y, radius) or None
        """
        if not self._initialized:
            if detection is None:
                return self._state
            self._initialize(detection)
            return self._state

        # ── Predict ────────────────────────────────────────────
        # predict() returns shape (6, 1) — flatten to (6,) for safe indexing
        p = self._kf.predict().flatten()
        px, py, pr = float(p[0]), float(p[1]), float(p[4])

        # ── Associate / correct ────────────────────────────────
        if detection is not None and self._in_gate(detection, px, py, pr):
            meas = np.array([[float(detection[0])],
                             [float(detection[1])],
                             [float(detection[2])]], dtype=np.float32)
            # correct() also returns shape (6, 1)
            c = self._kf.correct(meas).flatten()
            cx  = float(c[0])
            cy  = float(c[1])
            cvx = float(c[2])
            cvy = float(c[3])
            cr  = max(1.0, float(c[4]))

            self._hits += 1
            self._state.missed = 0
        else:
            # Coast on the prediction
            cx, cy, cr = px, py, max(1.0, pr)
            cvx = float(p[2])
            cvy = float(p[3])
            self._hits = max(0, self._hits - 1)
            self._state.missed += 1

        # ── Update public state ────────────────────────────────
        self._state.x         = cx
        self._state.y         = cy
        self._state.vx        = cvx
        self._state.vy        = cvy
        self._state.radius    = cr
        self._state.confirmed = self._hits >= self.confirm_hits
        self._state.age      += 1
        self._state.history.append((cx, cy, cr))

        return self._state

    @property
    def is_active(self) -> bool:
        return self._state.missed <= self.max_missed

    @property
    def is_confirmed(self) -> bool:
        return self._state.confirmed

    def reset(self) -> None:
        """Full reset — call between deliveries."""
        self._initialized = False
        self._hits        = 0
        self._state       = TrackState(x=0, y=0, radius=0)
        self._kf.errorCovPost = np.eye(6, dtype=np.float32)

    def get_trajectory(self) -> list[tuple]:
        return list(self._state.history)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _initialize(self, detection: tuple) -> None:
        x, y, r = float(detection[0]), float(detection[1]), float(detection[2])
        self._kf.statePost = np.array(
            [x, y, 0.0, 0.0, r, 0.0], dtype=np.float32
        ).reshape(6, 1)
        self._initialized = True
        self._hits        = 1
        self._state       = TrackState(x=x, y=y, radius=r, confirmed=False)
        self._state.history.append((x, y, r))

    def _in_gate(self, detection: tuple,
                 px: float, py: float, pr: float) -> bool:
        dx   = float(detection[0]) - px
        dy   = float(detection[1]) - py
        dist = (dx**2 + dy**2) ** 0.5
        # Base gate: sigma * radius with a minimum floor.
        # Cap at 80px so that a large predicted radius (e.g. after many coasted
        # frames) doesn't open the gate wide enough to grab distant noise blobs.
        gate = min(pr * self.gate_sigma + 20, 80.0)
        return dist < gate