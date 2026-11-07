import torch

from agents.ppo_agent import ActorCriticNet, PPOAgent


def test_network_output_shapes():
    net = ActorCriticNet(input_dim=6, action_dim=4, hidden_size=16)
    x = torch.zeros((5, 6))
    logits, values = net(x)
    assert logits.shape == (5, 4)
    assert values.shape == (5,)


def test_act_returns_valid_action_and_fills_buffer():
    agent = PPOAgent(feature_dim=6, seed=0)
    action = agent.act([0.0] * 6)
    assert action in range(4)
    assert agent.buffer_size() == 1
    agent.record_outcome(reward=0.1, done=False)
    assert len(agent.rewards) == 1


def test_gae_matches_hand_computed_monte_carlo_case():
    """With gamma=lambda=1 and all values=0, GAE must reduce to plain
    Monte Carlo returns: advantages = [3, 2, 1] for rewards [1, 1, 1]
    ending in a done flag."""
    agent = PPOAgent(feature_dim=2, seed=0, gamma=1.0, gae_lambda=1.0)
    agent.values = [0.0, 0.0, 0.0]
    agent.rewards = [1.0, 1.0, 1.0]
    agent.dones = [False, False, True]
    advantages, returns = agent._compute_gae(last_value=0.0)
    assert advantages == [3.0, 2.0, 1.0]
    assert returns == [3.0, 2.0, 1.0]


def test_update_clears_buffer():
    agent = PPOAgent(feature_dim=4, seed=0, minibatch_size=4, epochs=1)
    for _ in range(8):
        agent.act([0.1, 0.2, 0.3, 0.4])
        agent.record_outcome(reward=0.0, done=False)
    agent.dones[-1] = True
    assert agent.buffer_size() == 8
    agent.update(last_value=0.0)
    assert agent.buffer_size() == 0


def test_policy_logits_for_features_shape():
    agent = PPOAgent(feature_dim=6, seed=0)
    logits = agent.policy_logits_for_features([0.0] * 6)
    assert len(logits) == 4


def test_ppo_learns_a_trivial_one_step_bandit():
    """Sanity check that update() actually improves the policy, not just
    that it runs: a 1-step bandit where action 0 always yields +1 reward
    and every other action yields 0, regardless of the (irrelevant,
    fixed) observation. After several rollout+update cycles, the policy
    must concentrate probability mass on action 0."""
    agent = PPOAgent(feature_dim=3, action_dim=4, seed=0, lr=0.01, epochs=4, minibatch_size=16)
    state = [0.5, -0.5, 0.0]

    def rollout_and_update(n_steps):
        for _ in range(n_steps):
            action = agent.act(state)
            reward = 1.0 if action == 0 else 0.0
            agent.record_outcome(reward=reward, done=True)
        agent.update(last_value=0.0)

    logits_before = agent.policy_logits_for_features(state)
    probs_before = torch.softmax(torch.tensor(logits_before), dim=0)[0].item()

    for _ in range(40):
        rollout_and_update(32)

    logits_after = agent.policy_logits_for_features(state)
    probs_after = torch.softmax(torch.tensor(logits_after), dim=0)[0].item()

    assert probs_after > probs_before
    assert probs_after > 0.8, f"expected policy to concentrate on the rewarding action, got P(action=0)={probs_after}"


if __name__ == "__main__":
    test_network_output_shapes()
    test_act_returns_valid_action_and_fills_buffer()
    test_gae_matches_hand_computed_monte_carlo_case()
    test_update_clears_buffer()
    test_policy_logits_for_features_shape()
    test_ppo_learns_a_trivial_one_step_bandit()
    print("All ppo_agent tests passed.")
