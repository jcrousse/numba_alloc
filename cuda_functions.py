from numba import cuda
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from helpers import prod, blockspergrid_threadsperblock


@cuda.jit
def random_remove(opt_matrix_in, cnt_per_row, remove_prob, rn_states, opt_matrix_out):
    """
    Random removal of elements (switch value in opt_matrix from 1 to 0).
    Probability of being removed given by remove_prob
    keeps a count of the total number of elements removed per row in cnt_per_row
    random states for random number generator given in remove_prob
    """
    x, y = cuda.grid(2)
    if x >= opt_matrix_out.shape[0] and y >= opt_matrix_out.shape[1]:
        return

    flat_idx = opt_matrix_out.shape[1] * x + y
    rand_toss = xoroshiro128p_uniform_float32(rn_states, flat_idx)

    opt_matrix_out[x, y] = opt_matrix_in[x, y]

    if opt_matrix_in[x, y] == 1 and remove_prob[y] > rand_toss:
        opt_matrix_out[x, y] = 0
        opt_matrix_in[x, y] = 0
        _ = cuda.atomic.add(cnt_per_row, x, 1)

    cuda.syncthreads()


@cuda.jit
def adjust_after_remove(opt_matrix_in, ticket_count, idx_test, opt_matrix_out):
    """
    Finds the index
    :return:
    """
    x, y = cuda.grid(2)
    if x >= opt_matrix_in.shape[0] and y >= 1:
        return

    for k in range(8):
        idx_next = idx_test[k]
        opt_matrix_out[x, idx_next] = 10


def create_permutaion_table(can_add, n_permutations=64):
    """
    creates a table with random permutation of column indices.
    Goal is to order of column selection for adjustments.
    Columns with non-zero probabilities in prob_vector ar picked before the rest.
    Columns with higher probabilities in prob_vector are more likely to be high in the ordering
    """
    prob_vector = can_add / can_add.sum()
    non_zero_prob = np.array(np.where(prob_vector > 0))[0]
    zero_prob = np.array(np.where(prob_vector <= 0))[0]
    all_rows = []
    for i in range(n_permutations):
        shuffle_nonz = np.random.choice(non_zero_prob, non_zero_prob.shape, p=prob_vector[non_zero_prob], replace=False)
        shuffle_z = np.random.choice(zero_prob, zero_prob.shape, replace=False)
        full_row = np.hstack([shuffle_nonz, shuffle_z])
        all_rows.append(full_row)
    return np.vstack(all_rows)


if __name__ == '__main__':
    np.random.seed(123)

    n_rows = 64
    n_cols = 64

    opt_mat_in = np.random.randint(0, 2, (n_rows, n_cols))
    opt_mat_out_d = cuda.to_device(np.zeros((n_rows, n_cols)))
    opt_mat_in_d = cuda.to_device(opt_mat_in)

    over_under_columns = np.random.randint(0, 2, n_cols)

    row_change_cnt = cuda.to_device(np.zeros(n_rows))
    remove_prob = cuda.to_device(np.random.random(n_cols) * over_under_columns)


    blockspergrid, threadsperblock = blockspergrid_threadsperblock(n_rows, n_cols)

    rn_states = create_xoroshiro128p_states(prod(blockspergrid) * prod(threadsperblock),
                                            seed=np.random.randint(0, 10000))

    random_remove[blockspergrid, threadsperblock](
        opt_mat_in_d,
        row_change_cnt,
        remove_prob,
        rn_states,
        opt_mat_out_d
    )

    in_rows_totals = np.sum(opt_mat_in, axis=1)
    out_mat = opt_mat_out_d.copy_to_host()
    out_rows_totals = np.sum(out_mat, axis=1)
    added_cnt = row_change_cnt.copy_to_host()
    print(in_rows_totals - out_rows_totals)
    print(added_cnt)

    in_matrix_after = opt_mat_in_d.copy_to_host()

    comparison = out_mat == in_matrix_after
    equal_arrays = comparison.all()


    can_add = np.random.randint(1, 20, n_cols) * (1 - over_under_columns)
    can_add[20] = 500
    can_add[50] = 500
    perm_table = create_permutaion_table(can_add)

    # add_prob_scaled = add_prob / sum(add_prob)
    # add_prob_d = cuda.to_device(add_prob)
    idx_test = cuda.to_device(np.array([2, 3, 4, 10, 11, 12, 14, 16]))
    ticket_count = cuda.to_device(np.zeros(n_rows))

    adjust_after_remove[blockspergrid, threadsperblock](
        opt_mat_in_d,
        ticket_count,
        idx_test,
        opt_mat_out_d
    )

    transformed_out = opt_mat_out_d.copy_to_host()
    count_per_row = ticket_count.copy_to_host()


# TODO:
#  - If should add and can't (because already 1) -> atomic increment of add thing.
#  - Next function: compare to max ad add if necessary.
