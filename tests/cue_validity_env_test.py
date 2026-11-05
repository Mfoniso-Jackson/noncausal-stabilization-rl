from env.cue_validity_env import CueValidityEnv, validity_to_probs, ACTION_CAUSAL


def test_validity_to_probs_endpoints():
    p1, p0 = validity_to_probs(0.0)
    assert abs(p1 - 0.5) < 1e-9
    assert abs(p0 - 0.5) < 1e-9

    p1, p0 = validity_to_probs(1.0)
    assert abs(p1 - 0.95) < 1e-9
    assert abs(p0 - 0.05) < 1e-9


def test_cue_tracks_hidden_state_before_decorrelation():
    env = CueValidityEnv(cue_validity=1.0, seed=0)
    agreements = 0
    trials = 2000
    for i in range(trials):
        env = CueValidityEnv(cue_validity=1.0, seed=i)
        agreements += int(env.cue_present == env.hidden_state)
    rate = agreements / trials
    assert rate > 0.85, f"expected strong cue/state agreement at validity=1.0, got {rate}"


def test_decorrelate_removes_cue_state_relationship():
    env = CueValidityEnv(cue_validity=1.0, seed=0)
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


def test_cue_and_cue_action_never_affect_reward():
    """Reward must depend only on hidden_state and the causal action at
    target_phase, regardless of cue_validity, cue_present, or which
    action is chosen at non-target phases -- this must hold both before
    and after decorrelate()."""
    for validity in (0.0, 0.5, 1.0):
        for decorrelated in (False, True):
            env = CueValidityEnv(cue_validity=validity, seed=1)
            if decorrelated:
                env.decorrelate()
            env.reset()
            total_reward = 0.0
            while not env.done:
                action = 1  # always pick the cue action except forced causal check below
                if env.phase == env.target_phase:
                    action = ACTION_CAUSAL
                _, reward, done, info = env.step(action)
                total_reward += reward
            expected_goal = env.hidden_state == 1
            assert info["reward_obtained"] == expected_goal


def test_available_actions_and_action_space_fixed_across_conditions():
    for validity in (0.0, 0.5, 1.0):
        env = CueValidityEnv(cue_validity=validity, seed=0)
        assert env.available_actions() == (0, 1, 2, 3)


if __name__ == "__main__":
    test_validity_to_probs_endpoints()
    test_cue_tracks_hidden_state_before_decorrelation()
    test_decorrelate_removes_cue_state_relationship()
    test_cue_and_cue_action_never_affect_reward()
    test_available_actions_and_action_space_fixed_across_conditions()
    print("All cue_validity_env tests passed.")
