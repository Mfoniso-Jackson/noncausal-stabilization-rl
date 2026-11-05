from metrics.cue_validity import (
    cue_dependence_score,
    extinction_half_life,
    persistence_index,
    softmax_mass,
)

CAUSAL, CUE, NEUTRAL_A, NEUTRAL_B = 0, 1, 2, 3
AVAILABLE = (CAUSAL, CUE, NEUTRAL_A, NEUTRAL_B)


def test_cds_near_zero_when_causal_action_dominates():
    q_values = {CAUSAL: 10.0, CUE: 0.0, NEUTRAL_A: 0.0, NEUTRAL_B: 0.0}
    cds = cue_dependence_score(q_values, AVAILABLE, CAUSAL, CUE, (NEUTRAL_A, NEUTRAL_B))
    assert cds < 0.001


def test_cds_near_tie_breaking_floor_when_all_q_equal():
    q_values = {CAUSAL: 0.0, CUE: 0.0, NEUTRAL_A: 0.0, NEUTRAL_B: 0.0}
    cds = cue_dependence_score(q_values, AVAILABLE, CAUSAL, CUE, (NEUTRAL_A, NEUTRAL_B))
    assert abs(cds - 0.75) < 1e-9  # 3 of 4 actions are non-causal under a uniform tie


def test_cds_high_when_cue_action_dominates():
    q_values = {CAUSAL: 0.0, CUE: 10.0, NEUTRAL_A: 0.0, NEUTRAL_B: 0.0}
    cds = cue_dependence_score(q_values, AVAILABLE, CAUSAL, CUE, (NEUTRAL_A, NEUTRAL_B))
    assert cds > 0.999


def test_softmax_mass_sums_to_one():
    q_values = {CAUSAL: 1.3, CUE: -0.5, NEUTRAL_A: 2.0, NEUTRAL_B: 0.1}
    masses = softmax_mass(q_values, AVAILABLE)
    assert abs(sum(masses.values()) - 1.0) < 1e-9


def test_persistence_index_basic_ratio():
    assert abs(persistence_index(0.8, 0.4) - 0.5) < 1e-9
    assert persistence_index(0.8, 0.0) == 0.0


def test_persistence_index_none_when_denominator_negligible():
    assert persistence_index(0.0, 0.5) is None
    assert persistence_index(1e-10, 0.5) is None


def test_extinction_half_life_basic():
    baseline = 0.8
    sequence = [0.8, 0.6, 0.5, 0.39, 0.2, 0.1]
    assert extinction_half_life(sequence, baseline) == 3  # first block <= 0.4


def test_extinction_half_life_censored_when_never_halves():
    baseline = 0.8
    sequence = [0.8, 0.7, 0.6, 0.5, 0.45]
    assert extinction_half_life(sequence, baseline) == len(sequence)


def test_extinction_half_life_zero_when_baseline_negligible():
    assert extinction_half_life([0.1, 0.2, 0.0], 0.0) == 0


if __name__ == "__main__":
    test_cds_near_zero_when_causal_action_dominates()
    test_cds_near_tie_breaking_floor_when_all_q_equal()
    test_cds_high_when_cue_action_dominates()
    test_softmax_mass_sums_to_one()
    test_persistence_index_basic_ratio()
    test_persistence_index_none_when_denominator_negligible()
    test_extinction_half_life_basic()
    test_extinction_half_life_censored_when_never_halves()
    test_extinction_half_life_zero_when_baseline_negligible()
    print("All cue_validity metrics tests passed.")
