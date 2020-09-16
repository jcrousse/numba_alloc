import numpy as np
from numba import guvectorize


def maximal_values_row(opt_matrix, w_matrix, r_constraints, out):
    """ optimizes for the row constrains without taking col constraints into account"""
    for i in range(opt_matrix.shape[0]):
        for j in range(opt_matrix.shape[1]):
            out[i, j] = opt_matrix[i, j] + 1


mvr_signatures = [
    'void(int32[:, :], float32[:, :], int32[:], int32[:, :])',
    '(m,n),(m,n), (m)->(m,n)'
]

maximal_values_row_jit = guvectorize([mvr_signatures[0]], mvr_signatures[1], nopython=True)(maximal_values_row)
maximal_values_row_gpu = guvectorize([mvr_signatures[0]], mvr_signatures[1],  target='cuda')(maximal_values_row)

if __name__ == '__main__':
    n_row = 1000
    n_col = 20
    M = np.zeros((n_row, n_col)).astype(np.int32)
    W = np.random.random((n_row, n_col)).astype(np.float32)
    R = np.random.randint(0, n_col, (n_row)).astype(np.int32)
    C = np.random.randint(0, n_row, (n_col)).astype(np.int32)
    for f in [maximal_values_row, maximal_values_row_jit, maximal_values_row_gpu]:
        M_c, W_c, R_c = M.copy(), W.copy(), R.copy()
        f(M_c, W_c, R_c, M_c)
        print(M_c[0:1, 0:5])

