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
def adjust_after_remove(opt_matrix_in, ticket_count, rn_states, add_probability, opt_matrix_out):
    """
    adjusts each row to get back to the initial number of 1s per row.
    Number of elements to add per row is given in cnt_per_row
    some elements have a higher weight (more likely to be added) given by add_probability
    if not enough elements have an add_probability to reach the target number, elements are picked from
    other columns at random.
    :return:
    """
    x, y = cuda.grid(2)
    if x >= opt_matrix_in.shape[0] and y >= opt_matrix_in.shape[1]:
        return

    flat_idx = opt_matrix_in.shape[1] * x + y
    rand_toss = xoroshiro128p_uniform_float32(rn_states, flat_idx)

    # todo: order based on probability
    if add_probability[y] > 0 and opt_matrix_in[x, y] == 0:
        # time_waster = 0.0
        # while time_waster < rand_toss:
        #     time_waster += 0.000001
        queue_number = cuda.atomic.add(ticket_count, x, 1)
        if queue_number > 0:
            opt_matrix_out[x, y] = queue_number


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


    add_prob = np.random.random(n_cols) * (1 - over_under_columns)
    add_prob_scaled = add_prob / sum(add_prob)
    add_prob_d = cuda.to_device(add_prob)
    idx_test = cuda.to_device(np.array([2, 3, 4, 10, 11, 12, 14, 16]))
    ticket_count = cuda.to_device(np.zeros(n_rows))

    adjust_after_remove[blockspergrid, threadsperblock](
        opt_mat_in_d,
        ticket_count,
        rn_states,
        add_prob_d,
        opt_mat_out_d
    )

    transformed_out = opt_mat_out_d.copy_to_host()
    count_per_row = ticket_count.copy_to_host()


# TODO:
#  - Fixed permutation table to give a incrementing number to each col
#  - ? how to put the cols with more can_add closer to low numbers in perm table?
#  - ? How to skip gaps in perm table ? (e.g. row by row gaps)
#       Change the thread/block structure, and do it sequentially (one thread per row..).
