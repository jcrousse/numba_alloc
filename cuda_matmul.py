from numba import cuda, float32
import numpy as np
import math
# Controls threads per block and shared memory usage.
# The computation will be done on blocks of TPBxTPB elements.


TPB = 16

# https://nyu-cds.github.io/python-numba/05-cuda/


@cuda.jit
def matmul(A, B, C):
    """Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp


@cuda.jit
def fast_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    Each thread computes one element of the result matrix C
    """

    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    if x >= C.shape[0] and y >= C.shape[1]:
        # Quit if (x, y) is outside of valid C boundary
        return

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = 0.
    for i in range(int(A.shape[1] / TPB)):
        # Preload data into shared memory
        sA[tx, ty] = A[x, ty + i * TPB]
        sB[tx, ty] = B[tx + i * TPB, y]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[tx, j] * sB[j, ty]

        # Wait until all threads finish computing
        cuda.syncthreads()

    C[x, y] = tmp


if __name__ == '__main__':
    m1 = np.ones((16, 16))
    m2 = np.ones((16, 16))
    m3 = np.zeros((16, 16))

    matmul[16, (TPB, TPB)](m1, m2, m3)
    print(m3)
    fast_matmul[16, (TPB, TPB)](m1, m2, m3)
    print(m3)

    # The data array
    A = np.full((TPB * 2, TPB * 3), 3, np.float)  # [32 x 48] matrix containing all 3's
    B = np.full((TPB * 3, TPB * 1), 4, np.float)  # [48 x 16] matrix containing all 4's

    A_global_mem = cuda.to_device(A)
    B_global_mem = cuda.to_device(B)
    C_global_mem = cuda.device_array((TPB * 2, TPB * 1))  # [32 x 16] matrix result

    # Configure the blocks
    threadsperblock = (TPB, TPB)
    blockspergrid_x = int(math.ceil(A.shape[0] / threadsperblock[1]))
    blockspergrid_y = int(math.ceil(B.shape[1] / threadsperblock[0]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Start the kernel
    fast_matmul[blockspergrid, threadsperblock](A_global_mem, B_global_mem, C_global_mem)
    res = C_global_mem.copy_to_host()

    print(res)
