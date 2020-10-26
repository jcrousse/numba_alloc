from numba import cuda
import numpy as np
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import time

from helpers import prod, blockspergrid_threadsperblock


@cuda.jit
def random_optimize(opt_matrix_in, cnt_per_row, prob_per_col, rn_states, remove=False):
    """
    Random removal(adding) of elements by randomly switching value in opt_matrix from 1 to 0 (0 to 1).
    Probability of being removed(added) given by prob_per_col
    keeps a count of the total number of elements removed(added) per row in cnt_per_row
    random states for random number generator given in remove_prob
    """
    old_value = 1 if remove else 0
    new_value = 0 if remove else 1
    x, y = cuda.grid(2)
    if x >= opt_matrix_in.shape[0] and y >= opt_matrix_in.shape[1]:
        return

    flat_idx = opt_matrix_in.shape[1] * x + y
    rand_toss = xoroshiro128p_uniform_float32(rn_states, flat_idx)

    if opt_matrix_in[x, y] == old_value and prob_per_col[y] > rand_toss:
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


@cuda.jit
def sum_per_col(opt_matrix_in, sum_vector):
    """
    naive approach to calculate the sum per column with CUDA (there are probably more efficient ways)
    """
    x, y = cuda.grid(2)
    if x >= opt_matrix_in.shape[0] and y >= opt_matrix_in.shape[1]:
        return
    if opt_matrix_in[x, y] == 1:
        _ = cuda.atomic.add(sum_vector, y, 1)


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
    def __init__(self, n_rows, n_cols, n_perm=128):
        self.blocks_per_grid, self.threads_per_block = blockspergrid_threadsperblock(n_rows, n_cols)
        self.n_permutations = n_perm
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.computation_time = 0

    def __call__(self, opt_matrix, prob_per_col_remove, prob_per_col_add, can_add, can_remove):
        """
        single iteration of cuda optimization routine
        """
        opt_matrix_in = cuda.to_device(opt_matrix)
        prob_per_col_r = cuda.to_device(prob_per_col_remove)
        prob_per_col_a = cuda.to_device(prob_per_col_add)
        # total_per_col = cuda.to_device(np.zeros(self.n_cols))

        row_change_cnt = cuda.to_device(np.zeros(self.n_rows))

        xrn_states = create_xoroshiro128p_states(prod(self.blocks_per_grid) * prod(self.threads_per_block),
                                                 seed=np.random.randint(0, 10000))

        permutation_table_remove = cuda.to_device(create_permutaion_table(can_remove, self.n_permutations))
        permutation_table_add = cuda.to_device(create_permutaion_table(can_add, self.n_permutations))
        permutation_per_row = cuda.to_device(np.random.randint(0, self.n_permutations, self.n_rows))

        start_compute = time.time()
        # removal of over-represented elements
        self.adjustment(opt_matrix_in, row_change_cnt, prob_per_col_r, xrn_states, permutation_table_remove,
                        permutation_per_row, True)
        # addition of under-represented elements
        self.adjustment(opt_matrix_in, row_change_cnt, prob_per_col_a, xrn_states, permutation_table_add,
                        permutation_per_row, False)

        # sum_per_col[self.blocks_per_grid, self.threads_per_block](opt_matrix_in, total_per_col)
        #sum_array = [sum_reduce(opt_matrix_in[:, idx]) for idx in range(self.n_cols)]  # todo: //

        compute_time = time.time() - start_compute

        self.computation_time += compute_time

        opt_mat = opt_matrix_in.copy_to_host()
        return opt_mat

    def adjustment(self, opt_matrix_in, row_change_cnt, prob_per_col, xrn_states, permutation_table,
                   permutation_per_row, remove_values=True):
        # randomly remove/add
        random_optimize[self.blocks_per_grid, self.threads_per_block](
            opt_matrix_in,
            row_change_cnt,
            prob_per_col,
            xrn_states,
            remove_values
        )

        # re add/remove to adjust
        adjust_add_count[self.blocks_per_grid, self.threads_per_block](
            opt_matrix_in,
            row_change_cnt,
            permutation_table,
            permutation_per_row,
            not remove_values,
            False
        )
        adjust_add_count[self.blocks_per_grid, self.threads_per_block](
            opt_matrix_in,
            row_change_cnt,
            permutation_table,
            permutation_per_row,
            not remove_values,
            True
        )


@cuda.reduce
def sum_reduce(a, b):
    return a + b


if __name__ == '__main__':

    A = cuda.to_device(np.ones((10, 10)))
    B = cuda.to_device(np.zeros(10))
    sum_per_col[(10,10), (10,10)](A, B)
    C = B.copy_to_host()
    got = sum_reduce(A, size=2)

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
