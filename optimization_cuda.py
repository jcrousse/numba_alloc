from numba import cuda, float32
import numpy as np
import math


from numba.cuda.random import xoroshiro128p_uniform_float32
# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16


@cuda.jit
def iteration_improve_cuda(opt_matrix_in, added_count, over_alloc_pct, can_add, rn_states,
                           get_position, opt_matrix_out):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    x, y = cuda.grid(2)

    # tx = cuda.threadIdx.x
    # ty = cuda.threadIdx.y

    if x >= opt_matrix_out.shape[0] and y >= opt_matrix_out.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    flat_idx = opt_matrix_out.shape[1] * x + y
    rand_toss = xoroshiro128p_uniform_float32(rn_states, flat_idx)

    opt_matrix_out[x, y] = opt_matrix_in[x, y]

    if opt_matrix_in[x, y] == 1 and over_alloc_pct[y] > rand_toss:
        opt_matrix_out[x, y] = 0
        total = cuda.atomic.add(added_count, x, 1)

    cuda.syncthreads()

    if opt_matrix_in[x, y] == 0 and can_add[y] > 0:
        position_queue = cuda.atomic.add(get_position, x, 1)
        if position_queue <= total:
            opt_matrix_out[x, y] = 1
            _ = cuda.atomic.add(added_count, x, -1)
        # else:
        #     out_solution[x, y] = added_count[x]

    #out_solution[x, y] = cuda.blockIdx.y


if __name__ == '__main__':

    n_r = TPB * 2 ** 13
    n_c = TPB ** 2

    # The data array
    np.random.seed(123)
    M = np.random.randint(0, 2, (n_r, n_c)).astype(np.int32)
    added_count = np.zeros(n_r, dtype=np.int32)


    M_global_mem = cuda.to_device(M)
    O_global_mem = cuda.device_array((n_r, n_c))  # [32 x 16] matrix result

    # Configure the blocks
    threadsperblock = n_c
    # blockspergrid_x = int(math.ceil(M.shape[0] / threadsperblock[1]))
    # blockspergrid_y = int(math.ceil(M.shape[1] / threadsperblock[0]))
    blockspergrid = n_r # (blockspergrid_x, blockspergrid_y)

    # Start the kernel
    iteration_improve_cuda[blockspergrid, threadsperblock](
        M_global_mem,
        added_count,
        O_global_mem)
    res = O_global_mem.copy_to_host()

    print(res)

# TODO:
#  - One array to keep count of added / removed items
#  - Do the random add and count added elements
#  >> Checkpoint here: Do we see the number only increasing and the add count being correct ?
#  - Separate array for 'queue'. Set to 0 for everyone in the thread then use
#       atomic add to get a queue number. If lower than number to adjust then adjust
#  - >> Checkpoint: Do we see the contraints being met again ? Do we get closer to solution ?
#  - Do the same for removal and adjust

