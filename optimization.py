from numba import njit, prange
import numpy as np


@njit
def get_ranks(w_vector):
    """returns an array with the rank of each item in descending order"""
    tmp = np.flip(w_vector.argsort())
    ranks = np.empty_like(tmp)
    ranks[tmp] = np.arange(len(w_vector))
    return ranks


@njit(parallel=True)
def optimal_result(opt_matrix, w_matrix, r_vect):
    """ Optimization step 1: get optimal M values for each row, without looking at col constraints
    """
    for idx in prange(opt_matrix.shape[0]):
        ranks = get_ranks(w_matrix[idx, :])
        opt_matrix[idx, :] = ranks < r_vect[idx]

    return opt_matrix


if __name__ == '__main__':
    np.random.seed(123)
    n_row = 1000
    n_col = 20
    M = np.zeros((n_row, n_col)).astype(np.int32)
    W = np.random.random((n_row, n_col)).astype(np.float32)
    R = np.random.randint(0, n_col, (n_row)).astype(np.int32)
    C = np.random.randint(0, n_row, (n_col)).astype(np.int32)

    M2 = optimal_result(M, W, R)
    _ = 1
