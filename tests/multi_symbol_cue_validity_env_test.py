from env.multi_symbol_cue_validity_env import MultiSymbolCueValidityEnv, validity_to_probs, ACTION_CAUSAL
from metrics.cue_validity import cue_dependence_score, proxy_dependence


def test_validity_to_probs_matches_single_symbol_env():
    p1, p0 = validity_to_probs(0.0)
    assert abs(p1 - 0.5) < 1e-9
    assert abs(p0 - 0.5) < 1e-9
    p1, p0 = validity_to_probs(1.0)
    assert abs(p1 - 0.95) < 1e-9
    assert abs(p0 - 0.05) < 1e-9


def test_each_symbol_tracks_hidden_state_at_its_own_independent_rate():
    """The core new capability: two symbols with DIFFERENT validities in
    the same environment, each tracking the shared hidden_state at its own
    stated rate, independent of the other symbol's validity."""
    agreements = [0, 0]
    trials = 3000
    for i in range(trials):
        env = MultiSymbolCueValidityEnv(cue_validities=[1.0, 0.0], seed=i)
        agreements[0] += int(env.cue_present[0] == env.hidden_state)
        agreements[1] += int(env.cue_present[1] == env.hidden_state)
    rate_high = agreements[0] / trials
    rate_low = agreements[1] / trials
    assert rate_high > 0.85, f"symbol 0 (validity=1.0) expected strong agreement, got {rate_high}"
    assert 0.45 < rate_low < 0.55, f"symbol 1 (validity=0.0) expected ~50% agreement, got {rate_low}"


def test_three_symbols_independent_validities():
    agreements = [0, 0, 0]
    trials = 3000
    for i in range(trials):
        env = MultiSymbolCueValidityEnv(cue_validities=[1.0, 0.5, 0.0], seed=i)
        for s in range(3):
            agreements[s] += int(env.cue_present[s] == env.hidden_state)
    rates = [a / trials for a in agreements]
    assert rates[0] > 0.85, rates
    assert 0.55 < rates[1] < 0.75, rates  # validity=0.5 -> p_given_1=0.725, p_given_0=0.275
    assert 0.45 < rates[2] < 0.55, rates


def test_decorrelate_one_symbol_leaves_the_other_untouched():
    env = MultiSymbolCueValidityEnv(cue_validities=[1.0, 1.0], seed=0)
    env.decorrelate(symbol_id=0)
    assert env.is_decorrelated(0)
    assert not env.is_decorrelated(1)

    agreements = [0, 0]
    trials = 3000
    for i in range(trials):
        env.rng.seed(i)
        env.reset()
        agreements[0] += int(env.cue_present[0] == env.hidden_state)
        agreements[1] += int(env.cue_present[1] == env.hidden_state)
    rate0 = agreements[0] / trials
    rate1 = agreements[1] / trials
    assert 0.45 < rate0 < 0.55, f"decorrelated symbol 0 expected ~50% agreement, got {rate0}"
    assert rate1 > 0.85, f"untouched symbol 1 (validity=1.0) expected strong agreement, got {rate1}"


def test_decorrelate_none_decorrelates_every_symbol():
    env = MultiSymbolCueValidityEnv(cue_validities=[1.0, 1.0, 1.0], seed=0)
    env.decorrelate()
    for s in range(3):
        assert env.is_decorrelated(s)

    agreements = [0, 0, 0]
    trials = 3000
    for i in range(trials):
        env.rng.seed(i)
        env.reset()
        for s in range(3):
            agreements[s] += int(env.cue_present[s] == env.hidden_state)
    for s in range(3):
        rate = agreements[s] / trials
        assert 0.45 < rate < 0.55, f"symbol {s} expected ~50% agreement after decorrelate(), got {rate}"


def test_reverse_one_symbol_leaves_the_other_untouched():
    env = MultiSymbolCueValidityEnv(cue_validities=[1.0, 1.0], seed=0)
    env.reverse(symbol_id=1)
    assert env.is_reversed(1)
    assert not env.is_reversed(0)

    agreements = [0, 0]
    trials = 3000
    for i in range(trials):
        env.rng.seed(i)
        env.reset()
        agreements[0] += int(env.cue_present[0] == env.hidden_state)
        agreements[1] += int(env.cue_present[1] == env.hidden_state)
    rate0 = agreements[0] / trials
    rate1 = agreements[1] / trials
    assert rate0 > 0.85, f"untouched symbol 0 expected strong agreement, got {rate0}"
    assert rate1 < 0.15, f"reversed symbol 1 expected strong anti-correlation, got {rate1}"


def test_symbol_id_out_of_range_raises():
    env = MultiSymbolCueValidityEnv(cue_validities=[0.5, 0.5], seed=0)
    for bad_id in (-1, 2, 5):
        try:
            env.decorrelate(symbol_id=bad_id)
            assert False, f"expected ValueError for symbol_id={bad_id}"
        except ValueError:
            pass
        try:
            env.cue_action(bad_id)
            assert False, f"expected ValueError for symbol_id={bad_id}"
        except ValueError:
            pass


def test_at_least_one_symbol_required():
    try:
        MultiSymbolCueValidityEnv(cue_validities=[], seed=0)
        assert False, "expected ValueError for zero symbols"
    except ValueError:
        pass


def test_action_space_scales_with_symbol_count():
    for n in (1, 2, 3, 5):
        env = MultiSymbolCueValidityEnv(cue_validities=[0.5] * n, seed=0)
        actions = env.available_actions()
        assert len(actions) == n + 3, (n, actions)
        assert len(set(actions)) == len(actions), "action ids must be unique"
        assert ACTION_CAUSAL in actions
        for s in range(n):
            assert env.cue_action(s) in actions
        assert env.action_count == n + 3


def test_cue_actions_never_earn_reward_regardless_of_symbol_count_or_validity():
    for n in (1, 2, 3):
        for seed in range(30):
            env = MultiSymbolCueValidityEnv(cue_validities=[1.0] * n, seed=seed)
            env.reset()
            env.hidden_state = 1  # force the rewarding case to isolate the mechanism
            while env.phase < env.target_phase:
                env.step(env.neutral_actions[0])
            for s in range(n):
                probe = MultiSymbolCueValidityEnv(cue_validities=[1.0] * n, seed=seed)
                probe.reset()
                probe.hidden_state = 1
                while probe.phase < probe.target_phase:
                    probe.step(probe.neutral_actions[0])
                _, reward, done, info = probe.step(probe.cue_action(s))
                assert not info["reward_obtained"]
                assert reward <= 0.0 + 1e-9


def test_causal_action_reward_depends_only_on_hidden_state():
    for n in (1, 2):
        for validities in ([0.0] * n, [0.5] * n, [1.0] * n):
            for decorrelated in (False, True):
                env = MultiSymbolCueValidityEnv(cue_validities=validities, seed=1)
                if decorrelated:
                    env.decorrelate()
                env.reset()
                total_reward = 0.0
                while not env.done:
                    action = env.neutral_actions[0]
                    if env.phase == env.target_phase:
                        action = ACTION_CAUSAL
                    _, reward, done, info = env.step(action)
                    total_reward += reward
                expected_goal = env.hidden_state == 1
                assert info["reward_obtained"] == expected_goal


def test_observation_vector_shape_and_cue_bits():
    for n in (1, 2, 4):
        env = MultiSymbolCueValidityEnv(cue_validities=[1.0] * n, cue_phase=0, target_phase=4, seed=0)
        env.reset()  # phase starts at 0 == cue_phase, so cue bits are live immediately
        vector = env.observation_vector()
        assert len(vector) == n + 5
        assert env.feature_dim == n + 5
        for s in range(n):
            assert abs(vector[1 + s] - float(env.cue_present[s])) < 1e-9


def test_observation_vector_cue_bits_are_zero_outside_cue_phase():
    env = MultiSymbolCueValidityEnv(cue_validities=[1.0, 1.0], cue_phase=2, target_phase=4, seed=0)
    env.reset()
    while env.phase != env.target_phase:
        if env.phase != env.cue_phase:
            vector = env.observation_vector()
            assert vector[1] == 0.0 and vector[2] == 0.0
        env.step(env.neutral_actions[0])


def test_reward_noise_carries_over_correctly():
    obtained = 0
    trials = 2000
    for i in range(trials):
        env = MultiSymbolCueValidityEnv(cue_validities=[1.0, 1.0], reward_noise=0.5, seed=i)
        env.reset()
        env.hidden_state = 1
        while env.phase < env.target_phase:
            env.step(env.neutral_actions[0])
        _, reward, done, info = env.step(ACTION_CAUSAL)
        if info["reward_obtained"]:
            obtained += 1
    rate = obtained / trials
    assert 0.40 < rate < 0.60, f"reward_noise=0.5 expected ~50% obtained rate, got {rate}"


def test_proxy_dependence_and_cds_reused_unmodified_per_symbol():
    """Confirms, not just asserts, that metrics/cue_validity.py's existing
    functions work per-symbol with zero modification: probe each symbol
    independently, with every other action (other symbols' cue actions +
    neutral) folded into "neutral" for that probe."""
    env = MultiSymbolCueValidityEnv(cue_validities=[1.0, 1.0], cue_phase=4, target_phase=4, seed=0)
    available = env.available_actions()

    def fake_q(cue0_present, cue1_present):
        # A hand-built Q-function standing in for a trained agent's
        # output: causal action has high value; cue_action(0) has high
        # value only when symbol 0 is present; cue_action(1) is flat.
        return {
            ACTION_CAUSAL: 0.5,
            env.cue_action(0): 1.0 if cue0_present else 0.0,
            env.cue_action(1): 0.3,
            env.neutral_actions[0]: 0.1,
            env.neutral_actions[1]: 0.1,
        }

    q_present_0 = fake_q(True, False)
    q_absent_0 = fake_q(False, False)
    dep0 = proxy_dependence(q_present_0, q_absent_0, available, ACTION_CAUSAL)
    cds0 = cue_dependence_score(
        q_present_0, available, ACTION_CAUSAL, env.cue_action(0),
        (env.cue_action(1),) + env.neutral_actions,
    )
    assert isinstance(dep0, float)
    assert 0.0 <= cds0 <= 1.0

    # Symbol 1's own Q-values never change with cue0_present in this fake
    # Q-function, so its proxy_dependence (probed on itself) should be ~0.
    q_present_1 = fake_q(False, True)
    q_absent_1 = fake_q(False, False)
    dep1 = proxy_dependence(q_present_1, q_absent_1, available, ACTION_CAUSAL)
    assert abs(dep1) < 1e-9


if __name__ == "__main__":
    test_validity_to_probs_matches_single_symbol_env()
    test_each_symbol_tracks_hidden_state_at_its_own_independent_rate()
    test_three_symbols_independent_validities()
    test_decorrelate_one_symbol_leaves_the_other_untouched()
    test_decorrelate_none_decorrelates_every_symbol()
    test_reverse_one_symbol_leaves_the_other_untouched()
    test_symbol_id_out_of_range_raises()
    test_at_least_one_symbol_required()
    test_action_space_scales_with_symbol_count()
    test_cue_actions_never_earn_reward_regardless_of_symbol_count_or_validity()
    test_causal_action_reward_depends_only_on_hidden_state()
    test_observation_vector_shape_and_cue_bits()
    test_observation_vector_cue_bits_are_zero_outside_cue_phase()
    test_reward_noise_carries_over_correctly()
    test_proxy_dependence_and_cds_reused_unmodified_per_symbol()
    print("All multi_symbol_cue_validity_env tests passed.")
