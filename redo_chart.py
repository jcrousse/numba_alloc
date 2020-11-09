import pandas as pd
import seaborn as sns


if __name__ == '__main__':
    plot_data = pd.read_csv("comparison_gpu_cpu2.csv")
    plot_data['prob_size'] = plot_data['prob_size'] * 256
    sns_plot = sns.relplot(data=plot_data, x='prob_size', y='time', hue='method', kind='line')
    sns_plot.savefig("comparison_gpu_cpu.png")
