from functools import reduce
import operator
import math


def prod(iterable):
    return reduce(operator.mul, iterable, 1)


def blockspergrid_threadsperblock(n_rows, n_cols):
    threadsperblock =  (1, n_cols) #(TPB, TPB) #
    blockspergrid_x = int(math.ceil(n_rows / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(n_cols / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    return blockspergrid, threadsperblock
