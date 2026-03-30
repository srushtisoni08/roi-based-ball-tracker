import numpy as np

def _smooth(values, window=3):
    half = window // 2
    return [np.mean(values[max(0, i - half): min(len(values), i + half + 1)])
            for i in range(len(values))]