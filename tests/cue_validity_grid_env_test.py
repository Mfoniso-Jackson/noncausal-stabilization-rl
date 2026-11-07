from env.cue_validity_grid_env import ACTION_CAUSAL, CueValidityGridEnv


def test_cue_tracks_hidden_state_before_decorrelation():
    agreements = 0
    trials = 2000
    for i in range(trials):
        env = CueValidityGridEnv(cue_validity=1.0, seed=i)
        agreements += int(env.cue_present == env.hidden_state)
    rate = agreements / trials
    assert rate > 0.85, f"expected strong cue/state agreement at validity=1.0, got {rate}"


def test_decorrelate_removes_cue_state_relationship():
    env = CueValidityGridEnv(cue_validity=1.0, seed=0)
    env.decorrelate()
    assert env.is_decorrelated

    agreements = 0
    trials = 4000
    for i in range(trials):
        env.rng.seed(i)
        env.reset()
        agreements += int(env.cue_present == env.hidden_state)
    rate = agreements / trials
    assert 0.45 < rate < 0.55, f"expected ~50% agreement after decorrelation, got {rate}"


def test_reverse_flips_cue_state_correlation():
    env = CueValidityGridEnv(cue_validity=1.0, seed=0)
    env.reverse()
    assert env.is_reversed

    agreements = 0
    trials = 4000
    for i in range(trials):
        env.rng.seed(i)
        env.reset()
        agreements += int(env.cue_present == env.hidden_state)
    rate = agreements / trials
    assert rate < 0.15, f"expected strong anti-correlation after reverse(), got agreement rate {rate}"


def test_cue_only_visible_at_decision_cell():
    env = CueValidityGridEnv(cue_validity=1.0, seed=0)
    env.reset()
    saw_nonzero_elsewhere = False
    while not env.done:
        if not env.at_decision_cell():
            state = env.observe()
            cue_bit = state % 2
            if cue_bit != 0:
                saw_nonzero_elsewhere = True
            env.step(2)  # move left, a neutral action pre-decision
        else:
            break
    assert not saw_nonzero_elsewhere, "cue bit must be 0 in the observation everywhere except the decision cell"


def test_reward_depends_only_on_causal_action_and_hidden_state_at_decision_cell():
    """Reward must never depend on the cue, cue_validity, or which action is
    chosen anywhere except exactly at the decision cell -- must hold before
    and after decorrelate()."""
    for validity in (0.0, 0.5, 1.0):
        for decorrelated in (False, True):
            env = CueValidityGridEnv(cue_validity=validity, grid_size=4, seed=1)
            if decorrelated:
                env.decorrelate()
            env.reset()
            # Walk straight to the decision cell via neutral/right/up moves.
            total_reward = 0.0
            info = {}
            while not env.done:
                if env.at_decision_cell():
                    action = ACTION_CAUSAL
                else:
                    # move right then up toward (grid_size-1, grid_size-1)
                    x, y = env.pos
                    action = 3 if x < env.grid_size - 1 else 0
                _, reward, done, info = env.step(action)
                total_reward += reward
            expected_goal = env.hidden_state == 1
            assert info["reward_obtained"] == expected_goal


def test_episode_ends_immediately_at_decision_cell_regardless_of_action():
    env = CueValidityGridEnv(cue_validity=0.5, grid_size=3, seed=0)
    env.reset()
    # drive to decision cell
    while not env.at_decision_cell():
        x, y = env.pos
        action = 3 if x < env.grid_size - 1 else 0
        env.step(action)
    assert env.at_decision_cell()
    assert not env.done
    _, _, done, _ = env.step(1)  # ACTION_CUE, never rewarded
    assert done


def test_truncation_when_decision_cell_never_reached():
    env = CueValidityGridEnv(cue_validity=0.5, grid_size=5, seed=0)
    env.reset()
    steps = 0
    while not env.done and steps < 10_000:
        # bounce left/right forever, never approaching the decision cell's
        # far corner via the diagonal
        env.step(2)
        steps += 1
    assert env.done
    assert steps <= env.max_steps


def test_observation_vector_matches_observe_cue_bit():
    env = CueValidityGridEnv(cue_validity=1.0, grid_size=4, seed=0)
    env.reset()
    vector = env.observation_vector()
    state = env.observe()
    cue_bit_from_state = state % 2
    assert abs(vector[2] - float(cue_bit_from_state)) < 1e-9
    assert len(vector) == 6


def test_state_count_matches_observe_range():
    env = CueValidityGridEnv(cue_validity=0.5, grid_size=5, seed=0)
    assert env.state_count == 5 * 5 * 2
    for _ in range(200):
        env.reset()
        while not env.done:
            state = env.observe()
            assert 0 <= state < env.state_count
            x, y = env.pos
            action = 3 if x < env.grid_size - 1 else 0
            env.step(action)


def test_available_actions_fixed_across_conditions():
    for validity in (0.0, 0.5, 1.0):
        env = CueValidityGridEnv(cue_validity=validity, seed=0)
        assert env.available_actions() == (0, 1, 2, 3)


def test_reverse_and_decorrelate_never_both_true_by_default():
    env = CueValidityGridEnv(cue_validity=0.5, seed=0)
    assert not env.is_decorrelated
    assert not env.is_reversed


if __name__ == "__main__":
    test_cue_tracks_hidden_state_before_decorrelation()
    test_decorrelate_removes_cue_state_relationship()
    test_reverse_flips_cue_state_correlation()
    test_cue_only_visible_at_decision_cell()
    test_reward_depends_only_on_causal_action_and_hidden_state_at_decision_cell()
    test_episode_ends_immediately_at_decision_cell_regardless_of_action()
    test_truncation_when_decision_cell_never_reached()
    test_observation_vector_matches_observe_cue_bit()
    test_state_count_matches_observe_range()
    test_available_actions_fixed_across_conditions()
    test_reverse_and_decorrelate_never_both_true_by_default()
    print("All cue_validity_grid_env tests passed.")
