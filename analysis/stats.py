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


import random


def bootstrap_mean_ci(values, n_resamples=10000, confidence=0.95, seed=0):
    """Nonparametric bootstrap CI for the mean, as an alternative to
    confidence_interval()'s normal-theory formula: does not assume the
    sampling distribution of the mean is normal, which matters most for
    small samples (this repository's typical 15-30 seeds per cell) or
    visibly skewed per-seed values. Returns (lo, hi, point_estimate)."""
    values = list(values)
    n = len(values)
    rng = random.Random(seed)
    resampled_means = []
    for _ in range(n_resamples):
        resample = [values[rng.randrange(n)] for _ in range(n)]
        resampled_means.append(sum(resample) / n)
    resampled_means.sort()
    lo_idx = int((1 - confidence) / 2 * n_resamples)
    hi_idx = int((1 + confidence) / 2 * n_resamples) - 1
    point_estimate = sum(values) / n
    return resampled_means[lo_idx], resampled_means[hi_idx], point_estimate


def bootstrap_diff_ci(group_a, group_b, n_resamples=10000, confidence=0.95, seed=0):
    """Nonparametric bootstrap CI for the difference in means
    (mean(group_a) - mean(group_b)), resampling each group independently.
    Useful alongside compare_groups()'s t-test for the same
    decorrelate-vs-naive style comparisons used throughout this
    repository's experiments, without assuming normality. Returns
    (lo, hi, point_estimate)."""
    group_a = list(group_a)
    group_b = list(group_b)
    na, nb = len(group_a), len(group_b)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_resamples):
        resample_a = [group_a[rng.randrange(na)] for _ in range(na)]
        resample_b = [group_b[rng.randrange(nb)] for _ in range(nb)]
        diffs.append(sum(resample_a) / na - sum(resample_b) / nb)
    diffs.sort()
    lo_idx = int((1 - confidence) / 2 * n_resamples)
    hi_idx = int((1 + confidence) / 2 * n_resamples) - 1
    point_estimate = sum(group_a) / na - sum(group_b) / nb
    return diffs[lo_idx], diffs[hi_idx], point_estimate

