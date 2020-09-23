import numpy as np

from optimization import iterative_improvement
from linprog_scipy import run_linprog

if __name__ == '__main__':
    np.random.seed(123)

    RUN_GRID = False

    if RUN_GRID:
        rows_num = np.linspace(10, 1000, 10, dtype=int)
        cols_num = np.linspace(4, 50, 10, dtype=int)
        for n_row, n_col, run_id in zip(rows_num, cols_num, range(len(rows_num))):

            print(f"-----run {run_id}: {n_row} rows and {n_col} columns--------")

            M = np.zeros((n_row, n_col)).astype(np.int32)
            W = np.random.random((n_row, n_col)).astype(np.float32)
            R = np.random.randint(0, n_col, n_row).astype(np.int32)
            max_val = int(0.15 * n_row)
            diff_min_max = int(0.1 * n_row)
            C_max = np.random.randint(max_val, n_row, n_col).astype(np.int32)
            C_min = C_max - diff_min_max

            iterative_improvement(M, W, R, C_min, C_max, max_iter=10)
            res_, t = run_linprog(W, C_max.tolist(), R.tolist())

            print(f"Optimum found with solver in {t} seconds")
