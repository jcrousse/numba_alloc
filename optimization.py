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


@njit
def n_add_remove(opt_matrix, c_min, c_max):
    """Computes the level of over/under allocation for each pack."""
    total_p_col = opt_matrix.sum(axis=0)

    to_remove = total_p_col - c_max
    to_remove[np.where(to_remove < 0)] = 0

    to_add = c_min - total_p_col
    to_add[np.where(to_add < 0)] = 0

    can_add = c_max - total_p_col
    can_add[np.where(can_add < 0)] = 0

    can_remove = total_p_col - c_min
    can_remove[np.where(can_remove < 0)] = 0

    return to_remove, to_add, can_add, can_remove


@njit
def iteration_improve(opt_matrix, c_min, c_max):
    """Iterative improvement for optimization  (one round, needs to be called in a loop)
    -Calculates how column constraints are breached
    -Calculates how many can be added/removed per column
    -Randomly adds / remove cases to get closer to target
    """

    to_remove, to_add, can_add, can_remove = n_add_remove(opt_matrix, c_min, c_max)

    # # Inmprove convergeance by using min(max) as target when items are over (under) allocated
    diff_max_min = c_max - c_min
    to_remove_adj = (to_remove + diff_max_min) * (to_remove > 0)
    to_add_adj = (to_add + diff_max_min) * (to_add > 0)

    over_alloc_pct = (to_remove_adj) / opt_matrix.shape[0]
    under_alloc_pct = (to_add_adj) / opt_matrix.shape[0]

    for idx in prange(opt_matrix.shape[0]):
        row_values = opt_matrix[idx, :]
        improve_single_row(row_values, over_alloc_pct, under_alloc_pct, can_add, can_remove)

    return opt_matrix


@njit
def improve_single_row(opt_vector, over_alloc_pct, under_alloc_pct, can_add, can_remove):
    """ randomly adds or remove items based on the over and under allocation percentages
     """

    remove_items = over_alloc_pct * opt_vector > np.random.random(len(opt_vector))
    add_items = (1 - opt_vector) * under_alloc_pct > np.random.random(len(opt_vector))

    indices_shuffled = np.arange(opt_vector.shape[0])
    np.random.shuffle(indices_shuffled)
    for idx in indices_shuffled:
        if remove_items[idx] and opt_vector[idx]:  # noqa
            opt_vector[idx] = False
            replacement_candidates = np.where(can_add > 1)[0]
            if len(replacement_candidates) > 0:
                replace_idx = np.random.choice(replacement_candidates)
                opt_vector[replace_idx] = True
                add_items[replace_idx] = False
            else:
                # can't find a replacement -> abort.
                opt_vector[idx] = True

        if add_items[idx]:
            opt_vector[idx] = True
            replacement_candidates = np.where(can_remove > 1)[0]
            if len(replacement_candidates) > 0:
                remove_idx = np.random.choice(replacement_candidates)
                opt_vector[remove_idx] = False
            else:
                # can't remove any breaching offer, then abort the allocation
                opt_vector[idx] = False


def iterative_improvement(opt_matrix, c_min, c_max):
    """ run the improvement iteration until constraints are met """
    pass


if __name__ == '__main__':
    np.random.seed(123)
    n_row = 1000
    n_col = 20
    M = np.zeros((n_row, n_col)).astype(np.int32)
    W = np.random.random((n_row, n_col)).astype(np.float32)
    R = np.random.randint(0, n_col, (n_row)).astype(np.int32)
    C_max = np.random.randint(150, n_row, (n_col)).astype(np.int32)
    C_min = C_max - 100

    M2 = optimal_result(M, W, R)
    M3 = iteration_improve(M2, C_min, C_max)
    _ = 1
