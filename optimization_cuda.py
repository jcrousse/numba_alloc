from numba import cuda, float32
import numpy as np
import math
from numba.cuda.random import xoroshiro128p_uniform_float32
# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.
TPB = 16


@cuda.jit
def optimize(M, O):
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

    for k in range(TPB):
        sR[0, k] = k

    O[x, y] = (1 - M[x, y])


if __name__ == '__main__':

    n_r = TPB * 64
    n_c = TPB
    # The data array
    np.random.seed(123)
    M = np.random.randint(0, 2, (n_r, n_c))


    M_global_mem = cuda.to_device(M)
    O_global_mem = cuda.device_array((n_r, n_c))  # [32 x 16] matrix result

    # Configure the blocks
    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(M.shape[0] / threadsperblock[1]))
    blockspergrid_y = int(math.ceil(M.shape[1] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Start the kernel
    optimize[blockspergrid, threadsperblock](M_global_mem, O_global_mem)
    res = O_global_mem.copy_to_host()

    print(res)

# TODO:
#  How does the base example shared memory copy work? How to avoid doing the same copy as many
#  times as the number of threads per block?
#  - Benchmark move to/from GPU, maximal size, random number generation, ...
#  - Start from best solution per row
#  - Random assignment of can_add / can_remove per block.
#       - Give max, min, second max, second min...
#       - Adjsut value so that can add == can remove per block
#       - random split into row blocks
#       - Result into a matrix n_row_block rows and n_cols
#       - Need to find a way. Shared memory per thread with the M matrix, and swap values
#           where can add and can remove.
#  - Instead, 4 steps: 1)Add 2)adjust 3)remove 4)adjust
#  - Increment counter in shared memory when add/remove
#  - Generate an ordered list in shared memory to decide unequivocally which items should be.
#  Maybe try by running random num until get value >0.95 and take next memory space available
#  added or removed to adjust: Fille one shared array with random numbers, then another one with idx
#  - For each block, add and remove the ones that are randomly assigned to add/remove
#  - Need to assign same total add/remove per block
#  - random flip of rows/cols at each iteration

