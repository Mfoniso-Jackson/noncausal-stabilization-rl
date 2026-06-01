import numpy as np

def action_entropy(action_probs):
    """
    Computes Shannon entropy of a probability distribution.

    action_probs: list or numpy array of action probabilities
    """

    probs = np.array(action_probs, dtype=np.float32)

    # numerical stability
    probs = probs + 1e-8
    probs = probs / np.sum(probs)

    entropy = -np.sum(probs * np.log(probs))

    return entropy

def episode_entropy(action_log):
    """
    Computes entropy over actions taken in an episode.

    action_log: list of integers (actions taken)
    """

    actions, counts = np.unique(action_log, return_counts=True)

    probs = counts / np.sum(counts)

    return action_entropy(probs)

def rolling_entropy(action_logs, window=50):
    """
    Tracks entropy decay over training.
    """

    entropies = []

    for i in range(len(action_logs)):
        start = max(0, i - window)
        window_actions = action_logs[start:i+1]

        flat = [a for episode in window_actions for a in episode]

        if len(flat) == 0:
            entropies.append(0)
            continue

        _, counts = np.unique(flat, return_counts=True)
        probs = counts / np.sum(counts)

        entropies.append(action_entropy(probs))

    return entropies

