from analysis.stats import bootstrap_diff_ci, bootstrap_mean_ci


def test_bootstrap_mean_ci_point_estimate_matches_sample_mean():
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    lo, hi, point = bootstrap_mean_ci(values, n_resamples=2000, seed=0)
    assert abs(point - 3.0) < 1e-9
    assert lo <= point <= hi


def test_bootstrap_mean_ci_narrows_with_lower_variance():
    low_var = [5.0, 5.1, 4.9, 5.0, 5.05, 4.95] * 5
    high_var = [0.0, 10.0, 1.0, 9.0, 2.0, 8.0] * 5
    lo1, hi1, _ = bootstrap_mean_ci(low_var, n_resamples=2000, seed=0)
    lo2, hi2, _ = bootstrap_mean_ci(high_var, n_resamples=2000, seed=0)
    assert (hi1 - lo1) < (hi2 - lo2)


def test_bootstrap_mean_ci_degenerate_case_is_a_point():
    values = [7.0] * 10
    lo, hi, point = bootstrap_mean_ci(values, n_resamples=500, seed=0)
    assert abs(lo - 7.0) < 1e-9
    assert abs(hi - 7.0) < 1e-9
    assert abs(point - 7.0) < 1e-9


def test_bootstrap_diff_ci_point_estimate_matches_mean_difference():
    group_a = [10.0, 11.0, 12.0]
    group_b = [1.0, 2.0, 3.0]
    lo, hi, point = bootstrap_diff_ci(group_a, group_b, n_resamples=2000, seed=0)
    assert abs(point - 9.0) < 1e-9
    assert lo <= point <= hi


def test_bootstrap_diff_ci_excludes_zero_for_a_clear_separation():
    group_a = [10.0] * 20
    group_b = [0.0] * 20
    lo, hi, point = bootstrap_diff_ci(group_a, group_b, n_resamples=2000, seed=0)
    assert lo > 0.0, "a clearly-separated pair of groups should have a CI entirely above zero"


if __name__ == "__main__":
    test_bootstrap_mean_ci_point_estimate_matches_sample_mean()
    test_bootstrap_mean_ci_narrows_with_lower_variance()
    test_bootstrap_mean_ci_degenerate_case_is_a_point()
    test_bootstrap_diff_ci_point_estimate_matches_mean_difference()
    test_bootstrap_diff_ci_excludes_zero_for_a_clear_separation()
    print("All bootstrap stats tests passed.")
