import numpy as np


class LowPassFilter:
    def __init__(self):
        self.y = None

    def __call__(self, x, alpha):
        if self.y is None:
            self.y = x
        else:
            self.y = alpha * x + (1.0 - alpha) * self.y
        return self.y

class OneEuroFilter:
    def __init__(self, freq=30.0, mincutoff=1.2, beta=0.02, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_filter = LowPassFilter()
        self.dx_filter = LowPassFilter()
        self.last_x = None
        self.last_time = None

    @staticmethod
    def _alpha(cutoff, freq):
        te = 1.0 / freq
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t):
        if self.last_time is None or t <= self.last_time:
            dt = 1.0 / self.freq
        else:
            dt = t - self.last_time
        self.last_time = t
        freq = 1.0 / dt

        if self.last_x is None:
            dx = 0.0
        else:
            dx = (x - self.last_x) * freq

        edx = self.dx_filter(dx, self._alpha(self.dcutoff, freq))
        cutoff = self.mincutoff + self.beta * abs(edx)
        result = self.x_filter(x, self._alpha(cutoff, freq))

        self.last_x = x
        return result
