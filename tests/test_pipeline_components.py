from env import GridCueEnv
from metrics import cue_dependence_score, stabilization_persistence_index


def test_environment_observation_size():
    env = GridCueEnv(grid_size=10, num_cues=5, seed=1)
    assert len(env.reset()) == env.state_size


def test_cues_do_not_change_goal_reward_mechanism():
    no_cue = GridCueEnv(grid_size=3, num_cues=0, seed=1, max_steps=20)
    with_cue = GridCueEnv(grid_size=3, num_cues=2, seed=1, max_steps=20)
    assert no_cue.goal == with_cue.goal


def test_cds_is_higher_for_cue_zone_visitation():
    cues = [(1, 1)]
    concentrated = [(1, 1), (1, 2), (2, 1)] * 10
    dispersed = [(0, 0), (3, 3), (4, 4)] * 10
    assert cue_dependence_score(concentrated, cues, 5) > cue_dependence_score(dispersed, cues, 5)


def test_spi_normalized_persistence():
    assert stabilization_persistence_index(0.5, 0.25) == 0.5
    assert stabilization_persistence_index(0.0, 0.25) == 0.0

