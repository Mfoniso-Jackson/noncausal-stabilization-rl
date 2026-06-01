import numpy as np

def visitation_map(trajectories, grid_size=10):
    """
    Returns normalized visitation frequency over grid cells.
    """

    grid = np.zeros((grid_size, grid_size))

    for traj in trajectories:
        for (x, y) in traj:
            grid[x, y] += 1

    # normalize
    grid = grid / (np.sum(grid) + 1e-8)

    return grid

def cue_proximity_bias(trajectories, cue_positions):
    """
    Measures whether agent spends disproportionate time near cues.
    """

    distances = []

    for traj in trajectories:
        for (x, y) in traj:
            d = min([
                abs(x - cx) + abs(y - cy)
                for (cx, cy) in cue_positions
            ])
            distances.append(d)

    return np.mean(distances)

import matplotlib.pyplot as plt

def plot_visitation(grid, title="Visitation Heatmap"):
    plt.imshow(grid, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    plt.show()

def visitation_shift(before_map, after_map):
    """
    Measures how much spatial behaviour changes after cue removal.
    """

    diff = np.abs(before_map - after_map)

    return np.sum(diff)

