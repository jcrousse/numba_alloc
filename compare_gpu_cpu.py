import numpy as np
import pandas as pd
import seaborn as sns

from optimization import iterative_improvement

if __name__ == '__main__':

    all_dfs = []

    for gpu in [True]:
        method = "gpu" if gpu else 'cpu'
        for prob_size in [6, 16]:
            n_r = 16 * 2 ** prob_size
            n_c = 16 ** 2

            np.random.seed(567)

            M = np.zeros((n_r, n_c)).astype(np.int32)
            W = np.random.random((n_r, n_c)).astype(np.float32)
            R = np.random.randint(0, n_c, n_r).astype(np.int32)
            C_max = np.random.randint(0.1 * n_r, n_r, n_c).astype(np.int32)
            C_min = C_max - n_r * 0.05

            _, time_taken, cuda_time = iterative_improvement(M, W, R, C_min, C_max, max_iter=30, verbose=True,
                                                             use_cuda=gpu)

            if prob_size > 6:
                all_dfs.append(pd.DataFrame(
                    data={
                        'time': [time_taken],
                        'method': [method],
                        'prob_size': [n_r]
                    }))

    plot_df = pd.concat(all_dfs)
    plot_df.to_csv("comparison_gpu_cpu.csv")
    sns_plot = sns.relplot(data=plot_df, x='prob_size', y='time', hue='method', kind='line')
    sns_plot.savefig("comparison_gpu_cpu.png")
