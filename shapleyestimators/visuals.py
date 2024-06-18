import matplotlib.pyplot as plt
import numpy as np

def plot_data(results, dataset, filename=None):
    for name, data in results.items():
        sample_sizes = list(data.keys())
        mean_errors = [np.mean(data[sample_size]) for sample_size in sample_sizes]
        plt.plot(sample_sizes, mean_errors, label=name)
    plt.legend()
    plt.title(f'{dataset} Benchmark')
    plt.yscale('log')
    plt.xlabel('Sample Size')
    plt.ylabel('Squared Error')
    if filename is not None:
        plt.savefig(filename, dpi=1000, bbox_inches='tight')
    else:
        plt.show()
