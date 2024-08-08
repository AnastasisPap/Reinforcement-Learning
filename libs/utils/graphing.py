import matplotlib.pyplot as plt
import numpy as np

COLORS = ['#2B2F42', '#8D99AE', '#EF233C', '#6482AD', '#BC9F8B',
          '#FF8225', '#E3A5C7', '#914F1E', '#BEC6A0', '#FFC7ED']

SIZE_DEFAULT = 12
SIZE_LARGE = 14
plt.rc('font', weight='normal')
plt.rc('font', size=SIZE_DEFAULT)
plt.rc('axes', titlesize=SIZE_LARGE)
plt.rc('axes', labelsize=SIZE_LARGE)
plt.rc('xtick', labelsize=SIZE_DEFAULT)
plt.rc('ytick', labelsize=SIZE_DEFAULT)

def plot_results(x, y, x_label, y_label, labels, file_path, ylim=None):
    """Plots a 2D line using matplotlib and saves it with high quality.

    Args:
        x (np.array of shape (1, n)): the x-values
        y (np.array of shape (k, n)): the y-values
        x_label (str)
        y_label (str)
        labels (list of strings of size k)
        file_path (str): the full (or relative) path in which the plot will be stored at
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    colors = plt.cm.rainbow(np.linspace(0, 1, len(y))) if len(y) > len(COLORS) else COLORS

    for i in range(len(y)):
        ax.plot(x, y[i], label=labels[i], color=colors[i], linewidth=2)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if ylim: ax.set_ylim(ylim)
    ax.legend(loc='upper right')
    plt.savefig(file_path, dpi=300)