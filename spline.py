import numpy as np
from scipy import interpolate
import random


class SplineRandomizer:
    def __init__(self, min=-100, max=100, spline_ratio=30):
        self.min = min
        self.max = max
        self.spline_ratio = spline_ratio
        self.x = np.array([0, 10, 20, 30])
        self.y = [random.uniform(self.min, self.max) for _ in range(4)]
        self.tck = interpolate.splrep(self.x, self.y, s=0)

        self.values = interpolate.splev(np.linspace(0, self.spline_ratio, self.spline_ratio), self.tck, der=0).tolist()



    def get_next(self):
        if len(self.values) == 0:
            self.y = [self.y[-1]] + [random.uniform(self.min, self.max) for _ in range(3)]
            self.tck = interpolate.splrep(self.x, self.y, s=0)
            self.values = interpolate.splev(np.linspace(0, self.spline_ratio, self.spline_ratio), self.tck, der=0).tolist()
        return self.values.pop(0)
