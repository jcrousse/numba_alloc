from numba import cuda
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
from helpers import prod, blockspergrid_threadsperblock


@cuda.jit
def random_optimize(opt_matrix_in, cnt_per_row, prob_per_col, rn_states, opt_matrix_out, remove=False):
    """
    Random removal(adding) of elements by randomly switching value in opt_matrix from 1 to 0 (0 to 1).
    Probability of being removed(added) given by prob_per_col
    keeps a count of the total number of elements removed(added) per row in cnt_per_row
    random states for random number generator given in remove_prob
    """
    old_value = 1 if remove else 0
    new_value = 0 if remove else 1
    x, y = cuda.grid(2)
    if x >= opt_matrix_out.shape[0] and y >= opt_matrix_out.shape[1]:
        return

    flat_idx = opt_matrix_out.shape[1] * x + y
    rand_toss = xoroshiro128p_uniform_float32(rn_states, flat_idx)

    opt_matrix_out[x, y] = opt_matrix_in[x, y]

    if opt_matrix_in[x, y] == old_value and prob_per_col[y] > rand_toss:
        opt_matrix_out[x, y] = new_value
        opt_matrix_in[x, y] = new_value
        _ = cuda.atomic.add(cnt_per_row, x, 1)

    cuda.syncthreads()


@cuda.jit
def adjust_add_count(opt_matrix_in, cnt_per_row, perm_table, order_per_row, remove=False, overwrite=False):
    """
    Each element that should be added(removed) but can't be added(removed) increments the add_count.
    After this step, every element that can be added(removed) and which is n-th position in
    the ordered list of elements to add(remove) is added(removed) if n <= add_count
    :param perm_table: column order permutation to use for this row, among all column orders
    randomly generated
    :param
    """
    old_val = 0 if remove else 1
    x, y = cuda.grid(2)
    if x >= opt_matrix_in.shape[0] and y >= opt_matrix_in.shape[1]:
        return

    order_idx = order_per_row[x]
    if opt_matrix_in[x, y] == old_val and perm_table[order_idx, y] <= cnt_per_row[x]:
        if overwrite:
            opt_matrix_in[x, y] = (1 - old_val)
        else:
            _ = cuda.atomic.add(cnt_per_row, x, 1)


def create_permutaion_table(can_add, n_permutations=64):
    """
    creates a table with random permutation of column indices.
    Each row in resulting table is a random permutation
    Number of rows is given by n_permutations.
    Each permutation gives, in each column, the position in the permutation.

    Columns with non-zero probabilities in prob_vector ar ordered higher than the rest.
    Columns with higher probabilities in prob_vector are proportionaly more likely to be
      high in the ordering
    """
    prob_vector = can_add / can_add.sum()
    non_zero_prob = np.array(np.where(prob_vector > 0))[0]
    zero_prob = np.array(np.where(prob_vector <= 0))[0]
    all_rows = []
    for i in range(n_permutations):
        shuffle_nonz = np.random.choice(non_zero_prob, non_zero_prob.shape, p=prob_vector[non_zero_prob], replace=False)
        shuffle_z = np.random.choice(zero_prob, zero_prob.shape, replace=False)
        full_row = np.hstack([shuffle_nonz, shuffle_z])
        ordered = np.arange(len(full_row))
        ordered[full_row] = np.arange(len(full_row)) + 1
        all_rows.append(ordered)
    return np.vstack(all_rows)


class CudaIteration:
    def __init__(self, nr, nc, n_perm=128):
        self.blocks_per_grid, self.threads_per_block = blockspergrid_threadsperblock(nr, nc)
        self.n_permutations = n_perm

    def __call__(self, opt_matrix, prob_per_col, c_min, c_max, can_add):
        """
        single iteration of cuda optimization routine
        """
        opt_matrix_in = cuda.to_device(opt_matrix)
        opt_matrix_out = cuda.to_device(opt_matrix)
        prob_per_col_d = cuda.to_device(prob_per_col)

        row_change_cnt = cuda.to_device(np.zeros(n_rows))

        xrn_states = create_xoroshiro128p_states(prod(blockspergrid) * prod(threadsperblock),
                                                 seed=np.random.randint(0, 10000))

        # randomly remove
        random_optimize[self.blocks_per_grid, self.threads_per_block](
            opt_matrix_in,
            row_change_cnt,
            prob_per_col_d,
            xrn_states,
            opt_matrix_out,
            True
        )

        # re add to adjust
        perm_table = create_permutaion_table(can_add, self.n_permutations)

        adjust_add_count[blockspergrid, threadsperblock](
            opt_mat_in_d,
            row_addremove_cnt,
            perm_table_d,
            perm_per_row,
            False,
            False
        )
        adjust_add_count[blockspergrid, threadsperblock](
            opt_mat_in_d,
            row_addremove_cnt,
            perm_table_d,
            perm_per_row,
            False,
            True
        )

    def adjustment(self, opt_mat_in_d):
        pass

if __name__ == '__main__':
    np.random.seed(123)

    n_rows = 64
    n_cols = 64

    opt_mat_in = np.random.randint(0, 2, (n_rows, n_cols))
    opt_mat_out_d = cuda.to_device(np.zeros((n_rows, n_cols)))
    opt_mat_in_d = cuda.to_device(opt_mat_in)

    over_under_columns = np.random.randint(0, 2, n_cols)

    row_addremove_cnt = cuda.to_device(np.zeros(n_rows))
    remove_prob = cuda.to_device(np.random.random(n_cols) * over_under_columns)

    blockspergrid, threadsperblock = blockspergrid_threadsperblock(n_rows, n_cols)

    rn_states = create_xoroshiro128p_states(prod(blockspergrid) * prod(threadsperblock),
                                            seed=np.random.randint(0, 10000))

    random_optimize[blockspergrid, threadsperblock](
        opt_mat_in_d,
        row_addremove_cnt,
        remove_prob,
        rn_states,
        opt_mat_out_d,
        True
    )

    in_rows_totals = np.sum(opt_mat_in, axis=1)
    out_mat = opt_mat_out_d.copy_to_host()
    out_rows_totals = np.sum(out_mat, axis=1)
    added_cnt = row_addremove_cnt.copy_to_host()
    print(in_rows_totals - out_rows_totals)
    print(added_cnt)

    in_matrix_after = opt_mat_in_d.copy_to_host()

    comparison = out_mat == in_matrix_after
    equal_arrays = comparison.all()

    room_to_add = np.random.randint(1, 20, n_cols) * (1 - over_under_columns)
    add_mask = np.hstack([np.ones(10), np.zeros(n_cols - 10)])
    can_add_adjust = room_to_add * add_mask

    n_permutations = 2
    perm_table = create_permutaion_table(room_to_add, n_permutations)
    perm_table_d = cuda.to_device(perm_table)

    perm_per_row = cuda.to_device(np.random.randint(0, n_permutations, n_rows))

    # add_prob_scaled = add_prob / sum(add_prob)
    # add_prob_d = cuda.to_device(add_prob)
    idx_test = cuda.to_device(np.array([2, 3, 4, 10, 11, 12, 14, 16]))
    ticket_count = cuda.to_device(np.zeros(n_rows))


    adjust_add_count[blockspergrid, threadsperblock](
        opt_mat_in_d,
        row_addremove_cnt,
        perm_table_d,
        perm_per_row,
        False,
        False
    )
    print(row_addremove_cnt.copy_to_host())

    transformed_out = opt_mat_out_d.copy_to_host()
    count_per_row = ticket_count.copy_to_host()

    # do_adjustment[blockspergrid, threadsperblock](
    #     opt_mat_in_d,
    #     row_addremove_cnt,
    #     perm_table_d,
    #     perm_per_row,
    #     False
    # )
    adjust_add_count[blockspergrid, threadsperblock](
        opt_mat_in_d,
        row_addremove_cnt,
        perm_table_d,
        perm_per_row,
        False,
        True
    )

# TODO:
#  - Calculate can add/can remove, then prep the permutation table
#  - If should add and can't (because already 1) -> atomic increment of add_count. --DONE
#  - Next function: If can add (set to 0) and <= add_count -> Do the add!
#  - Same in reverse for removal.
#  - Problems with in/out matrices? First argument can't be output?
