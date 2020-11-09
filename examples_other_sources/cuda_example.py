from numba import vectorize, guvectorize, njit
from numba import cuda, prange
import numpy as np
import math


# example from tutorial https://github.com/ContinuumIO/gtc2017-numba

@cuda.jit(device=True)
def polar_to_cartesian(rho, theta):
    x = rho * math.cos(theta)
    y = rho * math.sin(theta)
    return x, y


@vectorize(['float32(float32, float32, float32, float32)'], target='cuda')
def polar_distance(rho1, theta1, rho2, theta2):
    x1, y1 = polar_to_cartesian(rho1, theta1)
    x2, y2 = polar_to_cartesian(rho2, theta2)

    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


@guvectorize(["void(float64[:],float64[:])"], "(n)->()")
def vector_func(a, b):
    b[0] = a.sum()


@cuda.jit
def silly_operation(mat_in, mat_out):
    for i in range(mat_in.shape[0]):
        for j in range(mat_in.shape[1]):
            mat_out[i, j] = mat_in[i, j] * 5


if __name__ == '__main__':
    n = 1000000
    rho1 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
    theta1 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    rho2 = np.random.uniform(0.5, 1.5, size=n).astype(np.float32)
    theta2 = np.random.uniform(-np.pi, np.pi, size=n).astype(np.float32)

    matrix_to_sum = np.random.randint(0, 5, (1000, 1000))
    result = vector_func(matrix_to_sum)

    my_matrix = np.ones((50, 50))
    result_m = np.ones((50, 50))
    silly_operation[64, 64](my_matrix, result_m)

    polar_distance(rho1, theta1, rho2, theta2)
