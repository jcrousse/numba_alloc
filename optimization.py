from numba import njit, prange
import numpy as np
import time


class Timer:
    def __init__(self, out_str, verbose=True):
        self.out_str = out_str
        self.time = time.time()
        self.verbose = verbose

    def __enter__(self):
        if self.verbose:
            print(self.out_str, end="")
        self.time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        seconds_taken = self.get_time_s()
        if self.verbose:
            print(str(seconds_taken), " seconds")

    def get_time_s(self):
        return round(time.time() - self.time, 2)


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

    over_alloc_pct = to_remove_adj / opt_matrix.shape[0]
    under_alloc_pct = to_add_adj / opt_matrix.shape[0]

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


def print_solution_diagnostic(opt_matrix, c_min, c_max, verbose):
    """prints a summary of solution status to keep track of improvements through iterations
    basic idea is to show how many constraints are breached and by how much"""
    over_alloc, under_alloc, _, _ = n_add_remove(opt_matrix, c_min, c_max)
    if verbose:
        if over_alloc.sum() == 0 and under_alloc.sum() == 0:
            print("all constraints are met!")
        else:
            for idx in prange(opt_matrix.shape[1]):
                if over_alloc[idx] > 0:
                    print(str(int(over_alloc[idx])) + " total too high for column " + str(idx))

                elif under_alloc[idx] > 0:
                    print(str(int(under_alloc[idx])) + " total too low for column " + str(idx))

    return over_alloc, under_alloc


def iterative_improvement(opt_matrix, w_matrix, r_vector,  c_min, c_max, max_iter=None, verbose=True):
    """
    get the solution from the optimal_result  function then iteratively calls the iteration_improve function
    until either a valid solution is found or a maximal number of iterations has been reached.
    :param opt_matrix: numpy array (I x J) boolean defining the solution
    :param w_matrix: numpy array (I x J) float defining the W matrix
    :param r_vector: row constraints vector R
    :param c_min: col constraints vector C_min
    :param c_max: col constraints vector C_max
    :param max_iter: maximal number of improvement iterations
    :param verbose: whether to print intermediary output or not
    """

    total_time = 0
    with Timer("Calculating initial solution...", verbose) as t:
        opt_matrix = optimal_result(opt_matrix, w_matrix, r_vector)
        total_time += t.get_time_s()

    over_alloc, under_alloc = print_solution_diagnostic(opt_matrix, c_min, c_max, verbose)

    solution_vals = []
    iteration_n = 0
    while (sum(over_alloc) > 1 or sum(under_alloc) > 1) and iteration_n != max_iter:
        solution_vals.append(np.sum(np.multiply(opt_matrix, w_matrix)))
        iteration_n += 1
        with Timer(f"Iteration {iteration_n} :\n", verbose) as t:
            opt_matrix = iteration_improve(opt_matrix, c_min, c_max)
            over_alloc, under_alloc = print_solution_diagnostic(opt_matrix, c_min, c_max, verbose)
            total_time += t.get_time_s()
    solution_vals.append(np.sum(np.multiply(opt_matrix, w_matrix)))

    return solution_vals, total_time


if __name__ == '__main__':
    np.random.seed(123)
    n_row = 2 ** 17
    n_col = 256
    M = np.zeros((n_row, n_col)).astype(np.int32)
    W = np.random.random((n_row, n_col)).astype(np.float32)
    R = np.random.randint(0, n_col, n_row).astype(np.int32)
    C_max = np.random.randint(0.1 * n_row, n_row, n_col).astype(np.int32)
    C_min = C_max - n_row * 0.05

    _, time_taken = iterative_improvement(M, W, R, C_min, C_max, max_iter=100, verbose=True)
    print(time_taken)
