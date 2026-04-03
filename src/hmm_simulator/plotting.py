import numpy as np

import matplotlib.pyplot as plt


def plot_profile(x, axis, title=None, show=True, path_to_fig=None):

    if title is not None:
        plt.title(title, fontsize=22)

    _x = x.copy()
    _x[x == 0] = np.nan

    axis.plot(_x, '-o')

    axis.set_yticks(range(1, 5))
    axis.set_yticklabels(['N0', 'L1', 'H2', 'C3'], fontsize=18)
    axis.set_ylabel('State', fontsize=20)
    axis.set_ylim(0.8, 4.2)

    axis.set_xticks(np.linspace(0, len(x), 6, dtype=int))
    axis.set_xticklabels(np.linspace(16, 96, 6, dtype=int), fontsize=18)
    axis.set_xlabel('Years', fontsize=20)

    plt.tight_layout()

    if show:
    	plt.show()

    if path_to_fig is not None:
    	plt.savefig(path_to_fig)


def plot_hmap(fig, X, show=True, path_to_fig=None):

    cmap = plt.get_cmap('viridis', 4)
    cmap.set_under('white', alpha=0.8)
    
    cax = plt.imshow(X, aspect='auto', cmap=cmap, vmin=1, vmax=4)

    n, p = np.shape(X)

    plt.yticks(np.linspace(0, n - 1, 6), np.linspace(1, n, 6, dtype=int), fontsize=16)
    plt.ylabel('Female', fontsize=18)
    
    plt.xticks(np.linspace(0, p - 1, 6), np.linspace(1, p, 6, dtype=int), fontsize=16)
    plt.xlabel('Time', fontsize=18)

    cbar = fig.colorbar(
        cax, extend='min', ticks=[1.35, 2.1, 2.9, 3.6], shrink=0.7, aspect=10, pad=0.08
    )
    cbar.ax.set_title('Diagnosis', fontsize=18, va='bottom')
    cbar.ax.set_yticklabels(['N0', 'L1', 'H2', 'C3'], fontsize=16, va='center')

    plt.tight_layout()

    if show:
        plt.show()

    if path_to_fig is not None:
        plt.savefig(path_to_fig)


def plot_histogram(X, show=True, path_to_fig=None):

    v, c = np.unique(X[X != 0].ravel(), return_counts=True)

    plt.title('Distribution simulated data', fontsize=20)

    plt.bar(v, c)

    plt.ylabel('Count', fontsize=18)
    plt.xlabel('State', fontsize=18)

    plt.xticks(np.arange(1, 5), ['N0', 'L1', 'H2', 'C4'], fontsize=16)
    plt.yticks(np.linspace(0, max(c), 6), np.linspace(0, max(c), 6, dtype=int),
               fontsize=16)

    y_shift = X.shape[0] * 0.08

    for num, x in enumerate(v):
        label = '{} %'.format(np.round(c[num] / sum(c), 4))
        plt.text(x=x - 0.2, y=c[num] + y_shift, s=label, size=16)

    plt.ylim(0, max(c) + y_shift * max(c))
    
    plt.tight_layout()

    if show:
        plt.show()

    if path_to_fig is not None:
        plt.savefig(path_to_fig)

    
