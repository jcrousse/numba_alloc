from numba import cuda, float32
import numpy as np
import math
from numba.cuda.random import xoroshiro128p_uniform_float32
# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16


@cuda.jit
def optimize(M, tmp_r,  O):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sM = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sR = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= O.shape[0] and y >= O.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    # tmp = 0.

    # Preload data into shared memory
    # sM[tx, ty] = M[x, y]

    # Wait until all threads finish preloading
    # cuda.syncthreads()

    # Computes partial product on the shared memory
    # for k in range(TPB):
    #     tmp += sM[tx, k]

    # Wait until all threads finish computing
    # cuda.syncthreads()


    sR[0, 0] = 0
    cuda.syncthreads()
    tmp = cuda.atomic.add(tmp_r, x, 1)

    O[x, y] = tmp


if __name__ == '__main__':

    n_r = TPB * 2 ** 13
    n_c = TPB ** 2
    # The data array
    np.random.seed(123)
    M = np.random.randint(0, 2, (n_r, n_c))
    added_count = np.zeros(n_r, dtype=np.int32)


    M_global_mem = cuda.to_device(M)
    O_global_mem = cuda.device_array((n_r, n_c))  # [32 x 16] matrix result

    # Configure the blocks
    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(M.shape[0] / threadsperblock[1]))
    blockspergrid_y = int(math.ceil(M.shape[1] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Start the kernel
    optimize[blockspergrid, threadsperblock](M_global_mem, added_count,  O_global_mem)
    res = O_global_mem.copy_to_host()

    print(res)

# TODO:
#  - Start from best solution per row
#  - Wrap all the stuff in the numba solution (start from best, diagnose at each round)
#  - One array to keep count of added / removed items
#  - Do the random add and count added elements
#  >> Checkpoint here: Do we see the number only increasing and the add count being correct ?
#  - Separate array for 'queue'. Set to 0 for everyone in the thread then use
#       atomic add to get a queue number. If lower than number to adjust then adjust
#  - >> Checkpoint: Do we see the contraints being met again ? Do we get closer to solution ?
#  - Do the same for removal and adjust

