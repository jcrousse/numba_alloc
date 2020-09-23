import numpy as np
import seaborn as sns
import pandas as pd

from optimization import iterative_improvement
from linprog_scipy import run_linprog


def get_matrices_opt(rows, cols):
    m_opt = np.zeros((rows, cols)).astype(np.int32)
    weights = np.random.random((rows, cols)).astype(np.float32)
    row_max = np.random.randint(0, cols, rows).astype(np.int32)
    max_val_col = int(0.15 * rows)
    diff_min_max = int(0.1 * rows)
    col_max = np.random.randint(max_val_col, rows, cols).astype(np.int32)
    col_min = col_max - diff_min_max

    return m_opt, weights, row_max, col_min, col_max


if __name__ == '__main__':
    np.random.seed(123)

    RUN_SINGLE = False
    RUN_GRID = True

    if RUN_SINGLE:
        M, W, R, C_min, C_max = get_matrices_opt(200, 20)
        res_, t_opt = run_linprog(W, C_max.tolist(), R.tolist())
        res_h, t_h = iterative_improvement(M, W, R, C_min, C_max, max_iter=10, verbose=False)

        plot_df = pd.DataFrame(data={
            'solution_value': [res_['fun'] * -1] * len(res_h) + list(res_h),
            'type': ['linprog'] * len(res_h) + ['heuristic'] * len(res_h),
            'iteration': list(range(len(res_h))) * 2
        })
        sns_plot = sns.relplot(data=plot_df, x='iteration', y='solution_value', hue='type', kind='line')
        sns_plot.set(ylim=(0, max(res_h) * 1.1))

        sns_plot.savefig("solution_per_iteration.png")

    if RUN_GRID:
        rows_num = np.linspace(10, 300, 5, dtype=int)
        cols_num = np.linspace(4, 30, 5, dtype=int)

        all_dfs = []
        for n_row, n_col, run_id in zip(rows_num, cols_num, range(len(rows_num))):

            problem_size = n_row * n_col

            print(f"-----run {run_id}: {n_row} rows and {n_col} columns--------")

            M, W, R, C_min, C_max = get_matrices_opt(n_row, n_col)

            res_h, t_h = iterative_improvement(M, W, R, C_min, C_max, max_iter=10, verbose=False)
            res_, t = run_linprog(W, C_max.tolist(), R.tolist())

            all_dfs.append(pd.DataFrame(
                data={
                    'time': [t_h, t],
                    'method': ['linprog', 'heuristic'],
                    'prob_size': [problem_size, problem_size]
                }))

        plot_df = pd.concat(all_dfs)
        sns_plot = sns.relplot(data=plot_df, x='prob_size', y='time', hue='method', kind='line')
        sns_plot.savefig("time_to_solution.png")
