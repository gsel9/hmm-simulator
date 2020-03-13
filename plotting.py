import numpy as np

import matplotlib.pyplot as plt


def plot_profile(x, axis, title=None, show=True, path_to_fig=None):

    if title is not None:
        plt.title(title, fontsize=22)

    _x = x.copy()
    _x[x == 0] = np.nan

    axis.plot(_x, '-o')

    axis.set_yticks(range(0, 5))
    axis.set_yticklabels(['D4', 'N0', 'L1', 'H2', 'C3'], fontsize=18)
    axis.set_ylabel('State', fontsize=20)

    axis.set_xticks(np.linspace(0, len(x), 6, dtype=int))
    axis.set_xticklabels(np.linspace(16, 96, 6, dtype=int), fontsize=18)
    axis.set_xlabel('Years', fontsize=20)

    plt.tight_layout()

    if show:
    	plt.show()

    if path_to_fig is not None:
    	plt.savefig(path_to_fig)


# TODO: Make discrete cbar with N0, L1, R2 etc. 
def plot_hmap(X, show=True, path_to_fig=None):

    n, p = np.shape(X)

    plt.figure()
    plt.imshow(X, aspect='auto')

    plt.xticks(np.linspace(0, n - 1, 6), np.linspace(1, n, 6, dtype=int), fontsize=16)
    plt.yticks(np.linspace(0, p - 1, 6), np.linspace(1, p, 6, dtype=int), fontsize=16)

    plt.xlabel('Female', fontsize=18)
    plt.ylabel('Time', fontsize=18)

    plt.colorbar()

    plt.tight_layout()

    if show:
        plt.show()

    if path_to_fig is not None:
        plt.savefig(path_to_fig)


def plot_histogram(X, show=True, path_to_fig=None):

    v, c = np.unique(X[X != 0].ravel(), return_counts=True)

    plt.figure(figsize=(8, 5))
    plt.title('Distribution simulated data', fontsize=20)

    plt.bar(v, c)

    plt.ylabel('Count', fontsize=18)
    plt.xlabel('State', fontsize=18)

    plt.xticks(np.arange(1, 5), ['N0', 'L1', 'H2', 'C4'], fontsize=16)

    plt.yticks(np.linspace(0, max(c), 6), np.linspace(0, max(c), 6, dtype=int),
        fontsize=16)

    for i in range(len(v)):
        label = '{} %'.format(np.round(c[i] / sum(c), 4))
        plt.text(x=v[i] - 0.2, y=c[i] + 100, s=label, size=16)

    plt.ylim(0, max(c) + 0.05 * max(c))
    
    plt.tight_layout()

    if show:
        plt.show()

    if path_to_fig is not None:
        plt.savefig(path_to_fig)

    
