import random
import numpy as np
import torch

def set_seed(seed: int = 42):
    """
    Sets seed for:
    - Python
    - NumPy
    - PyTorch
    - CUDA (if available)
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # makes results more deterministic (slower but important for papers)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def worker_seed(seed, worker_id):
    """
    Used in multiprocessing / vectorized RL environments.
    """

    return seed + worker_id
