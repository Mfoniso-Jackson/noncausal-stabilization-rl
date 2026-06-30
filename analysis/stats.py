from scipy.stats import ttest_ind

def compare_groups(group_a, group_b):
    """
    Compare low vs high uncertainty conditions.
    """

    t_stat, p_value = ttest_ind(group_a, group_b)

    return {
        "t_stat": t_stat,
        "p_value": p_value
    }

import numpy as np

def cohens_d(a, b):
    """
    Measures effect size between two conditions.
    """

    a = np.array(a)
    b = np.array(b)

    diff = np.mean(a) - np.mean(b)
    pooled_std = np.sqrt((np.std(a)**2 + np.std(b)**2) / 2)

    return diff / (pooled_std + 1e-8)

from scipy.stats import pearsonr

def uncertainty_correlation(uncertainty_levels, metric_values):
    """
    Correlation between uncertainty and behavioural stabilization.
    """

    r, p = pearsonr(uncertainty_levels, metric_values)

    return {
        "correlation": r,
        "p_value": p
    }

import numpy as np

def confidence_interval(data, confidence=0.95):
    """
    Compute simple confidence interval.
    """

    mean = np.mean(data)
    std = np.std(data)
    n = len(data)

    margin = 1.96 * (std / np.sqrt(n))

    return (mean - margin, mean + margin)

def stability_across_seeds(metric_by_seed):
    """
    Measures variance across random seeds.
    """

    return {
        "mean": np.mean(metric_by_seed),
        "std": np.std(metric_by_seed),
        "stability_score": 1 / (np.std(metric_by_seed) + 1e-8)
    }

