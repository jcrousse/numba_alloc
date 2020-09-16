from numba import vectorize, njit
from numba import cuda
import numpy as np
import math


# example from tutorial https://github.com/ContinuumIO/gtc2017-numba

@cuda.jit(device=True)
def polar_to_cartesian(rho, theta):
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    return x, y  # This is Python, so let's return a tuple


@vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
def polar_distance(rho1, theta1, rho2, theta2):
    x1, y1 = polar_to_cartesian(rho1, theta1)
    x2, y2 = polar_to_cartesian(rho2, theta2)

    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


if __name__ == '__main__':
    n = 1000000
    rho1 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
    theta1 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    rho2 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
    theta2 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)

    polar_distance(rho1, theta1, rho2, theta2)
