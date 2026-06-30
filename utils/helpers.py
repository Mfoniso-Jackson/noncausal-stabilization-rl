import numpy as np

def moving_average(data, window=10):
    """
    Smooths noisy RL curves.
    """

    data = np.array(data)

    if len(data) < window:
        return data

    return np.convolve(data, np.ones(window)/window, mode='valid')

def moving_average(data, window=10):
    """
    Smooths noisy RL curves.
    """

    data = np.array(data)

    if len(data) < window:
        return data

    return np.convolve(data, np.ones(window)/window, mode='valid')

def normalize(x):
    """
    Min-max normalization.
    """

    x = np.array(x)

    return (x - np.min(x)) / (np.max(x) - np.min(x) + 1e-8)

def flatten_trajectories(trajectories):
    """
    Converts list of episodes into flat list of states/actions.
    """

    return [item for traj in trajectories for item in traj]

def safe_div(a, b):
    return a / (b + 1e-8)

def discounted_return(rewards, gamma=0.99):
    """
    Computes discounted return for an episode.
    """

    returns = []
    G = 0

    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)

    return np.array(returns)
