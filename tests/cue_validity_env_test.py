from env.cue_validity_env import CueValidityEnv, validity_to_probs, ACTION_CAUSAL, ACTION_CUE


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


def test_observation_vector_matches_observe_cue_bit():
    env = CueValidityEnv(cue_validity=1.0, cue_phase=4, target_phase=4, seed=0)
    env.reset()
    vector = env.observation_vector()
    state = env.observe()
    cue_bit_from_state = state % 2
    assert abs(vector[1] - float(cue_bit_from_state)) < 1e-9
    assert len(vector) == 6


def test_reverse_flips_cue_state_correlation():
    env = CueValidityEnv(cue_validity=1.0, seed=0)
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


def test_reward_noise_defaults_to_zero_and_is_backward_compatible():
    env = CueValidityEnv(cue_validity=1.0, seed=0)
    assert env.reward_noise == 0.0


def test_reward_noise_withholds_earned_reward_at_the_stated_rate():
    """reward_noise=1.0 should withhold every earned reward; reward_noise=0.0
    should withhold none. Intermediate rates should land close to the stated
    probability over many trials."""
    for noise, expect_low, expect_high in ((0.0, 0.99, 1.01), (1.0, -0.01, 0.01), (0.5, 0.40, 0.60)):
        obtained = 0
        trials = 3000
        for i in range(trials):
            env = CueValidityEnv(cue_validity=1.0, reward_noise=noise, seed=i)
            env.reset()
            env.hidden_state = 1  # force the rewarding case to isolate the noise mechanism
            while env.phase < env.target_phase:
                env.step(2)
            _, reward, done, info = env.step(ACTION_CAUSAL)
            if info["reward_obtained"]:
                obtained += 1
        rate = obtained / trials
        assert expect_low <= rate <= expect_high, f"reward_noise={noise}: expected obtained-rate in [{expect_low}, {expect_high}], got {rate}"


def test_reward_noise_never_lets_the_cue_action_earn_reward():
    """The one thing reward_noise must never do is make ACTION_CUE
    rewarding under any noise level -- it only ever withholds an
    already-earned causal reward, never grants an unearned one."""
    for noise in (0.0, 0.3, 0.6, 1.0):
        for seed in range(50):
            env = CueValidityEnv(cue_validity=1.0, reward_noise=noise, seed=seed)
            env.reset()
            env.hidden_state = 1
            while env.phase < env.target_phase:
                env.step(2)
            _, reward, done, info = env.step(ACTION_CUE)
            assert not info["reward_obtained"]
            assert reward <= 0.0 + 1e-9


def test_reward_noise_does_not_affect_cue_validity_mechanics():
    """reward_noise must be fully orthogonal to the cue/hidden-state
    relationship -- adding reward noise must not change cue_validity's
    own, already-tested behavior."""
    for noise in (0.0, 0.5, 1.0):
        agreements = 0
        trials = 2000
        for i in range(trials):
            env = CueValidityEnv(cue_validity=1.0, reward_noise=noise, seed=i)
            agreements += int(env.cue_present == env.hidden_state)
        rate = agreements / trials
        assert rate > 0.85, f"reward_noise={noise} should not change cue/state agreement, got {rate}"


def test_reverse_and_decorrelate_never_both_true_by_default():
    env = CueValidityEnv(cue_validity=0.5, seed=0)
    assert not env.is_decorrelated
    assert not env.is_reversed


def test_observation_vector_target_flags():
    env = CueValidityEnv(cue_validity=0.5, cue_phase=3, target_phase=4, seed=0)
    env.reset()
    while env.phase < env.target_phase:
        vector = env.observation_vector()
        assert vector[2] == 0.0  # not target phase
        assert vector[3] == 1.0  # pre-target
        env.step(2)  # neutral action, never ends episode early
    vector = env.observation_vector()
    assert vector[2] == 1.0
    assert vector[3] == 0.0


if __name__ == "__main__":
    test_validity_to_probs_endpoints()
    test_cue_tracks_hidden_state_before_decorrelation()
    test_decorrelate_removes_cue_state_relationship()
    test_cue_and_cue_action_never_affect_reward()
    test_available_actions_and_action_space_fixed_across_conditions()
    test_observation_vector_matches_observe_cue_bit()
    test_observation_vector_target_flags()
    test_reverse_flips_cue_state_correlation()
    test_reverse_and_decorrelate_never_both_true_by_default()
    test_reward_noise_defaults_to_zero_and_is_backward_compatible()
    test_reward_noise_withholds_earned_reward_at_the_stated_rate()
    test_reward_noise_never_lets_the_cue_action_earn_reward()
    test_reward_noise_does_not_affect_cue_validity_mechanics()
    print("All cue_validity_env tests passed.")
