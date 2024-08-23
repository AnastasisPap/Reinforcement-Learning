from __future__ import annotations
import matplotlib.pyplot as plt
import seaborn as sns
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

def plot_results(
        x: np.ndarray,
        y: np.ndarray,
        x_label: str,
        y_label: str,
        labels: list[str],
        file_path:str,
        ylim: tuple[int] | None=None):
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
    plt.close()

def plot_3d(
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        x_label: str,
        y_label: str,
        z_label: str,
        title: str,
        file_path: str,
        limits: list[tuple] | None=None):
    """Makes a 3D plot using matplotlib and saves it with high quality.

    Args:
        x (np.array of shape (m, n)): the x-values
        y (np.array of shape (m, n)): the y-values
        z (np.array of shape (m, n)): the z-values
        x_label (str)
        y_label (str)
        z_label (str)
        title (str)
        file_path (str): the full (or relative) path in which the plot will be stored at
        limits (list of tuples): the limits for the x, y and z axes
    """
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})

    ax.plot_surface(x, y, z, color=COLORS[0], linewidth=2, cmap='viridis', edgecolor='none')
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_zlabel(z_label)
    ax.set_title(title)

    if limits:
        ax.set_xlim(limits[0])
        ax.set_ylim(limits[1])
        ax.set_zlim(limits[2])

    plt.savefig(file_path, dpi=300)
    plt.close()

def plot_policy(
        policy_grid: np.ndarray | list,
        x_label: str,
        y_label: str,
        title: str,
        file_path: str,
        args: dict):
    """
    Args:
     - policy_grid (np.array of shape (m, n)): the policy grid
     - x_label (str): the x-axis label
     - y_label (str): the y-axis label
     - title: the title of the plot
     - file_path (str): the full (or relative) path in which the plot will be stored at
     - args (dict):
        - x_ticks (list of strings of size m): the x-axis ticks
        - y_ticks (list of strings of size n): the y-axis ticks
        - cbar_ticks (list of strings of size k): the colorbar ticks
        - cbat_labels (list of strings of size k): the colorbar labels
    """

    fig, ax = plt.subplots(figsize=(10, 6))

    cmap_name = args.get('cmap', 'Accent_r')
    origin = args.get('origin', None)
    cmap = plt.get_cmap(cmap_name, len(np.unique(policy_grid)))
    plot = ax.imshow(policy_grid, cmap=cmap, origin=origin)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)

    x_ticks = args.get('x_ticks', None)
    y_ticks = args.get('y_ticks', None)
    if x_ticks is not None:
        ax.set_xticks(ticks=range(len(x_ticks)), labels=x_ticks)
        ax.set_yticks(ticks=range(len(y_ticks)), labels=y_ticks)

    cbar = fig.colorbar(plot)
    cbar_ticks = args.get('cbar_ticks', None)
    cbar_labels= args.get('cbar_labels', None)
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks, labels=cbar_labels)

    plt.savefig(file_path, dpi=300)
    plt.close()

def graph_policy_trajectory(
        EnvClass: object,
        policy: dict | np.ndarray | list,
        args: dict,
        file_path: str):
    """Stores a heatmap of the agent trajectory using the policy. Also if the args
    include {'render_mode': human}, then the trajectory is displayed in a pygame window.

    Args:
        EnvClass (Env.gym): the environment class
        policy (np.array): the policy
        args (dict): the arguments to pass to the environment
        file_path (str): the full (or relative) path in which the plot will be stored at
    """
    env = EnvClass(args)

    grid = env.grid.copy()
    s = env.reset()
    is_term = False

    while not is_term:
        a = policy[s]
        grid[s] = env.agent_marker
        s, _, is_term = env.step(a)

    sns.heatmap(grid, cmap=sns.color_palette("Blues", as_cmap=True), xticklabels=False, yticklabels=False, cbar=False, square=True)
    plt.savefig(file_path, dpi=300)
    plt.close()