from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numpy as np


@cuda.jit
def random_matrix(states, m_out):
    thread_id = cuda.grid(1)
    x = xoroshiro128p_uniform_float32(states, thread_id)
    m_out[thread_id] = x


if __name__ == '__main__':
    threads_per_block = 64
    blocks = 24
    rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
    out = np.zeros(threads_per_block * blocks, dtype=np.float32)

    random_matrix[blocks, threads_per_block](rng_states, out)
    print(out)
