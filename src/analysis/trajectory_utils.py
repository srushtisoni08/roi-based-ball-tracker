import numpy as np


def _smooth(values: list, window: int = 5) -> list:
    """
    Gaussian-weighted moving average.
    Gaussian weights give more influence to central points,
    producing smoother curves than a flat mean — important for
    reliable bounce detection from noisy Y-coordinate sequences.
    """
    if len(values) < 2:
        return list(values)

    half = window // 2
    # Build Gaussian weights centred at zero
    x       = np.arange(-half, half + 1, dtype=float)
    weights = np.exp(-0.5 * (x / (half / 2.0 + 1e-6)) ** 2)
    weights /= weights.sum()

    result = []
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        # Slice weights to match available window
        w_lo = half - (i - lo)
        w_hi = w_lo + (hi - lo)
        w    = weights[w_lo:w_hi]
        w    = w / w.sum()   # renormalize at edges
        result.append(float(np.dot(values[lo:hi], w)))

    return result