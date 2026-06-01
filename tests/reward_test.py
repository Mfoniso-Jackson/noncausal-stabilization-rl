from env.gridworld import GridWorld


def test_reward_only_at_goal():
    env = GridWorld(size=10, max_steps=200)

    env.reset()

    assert env._get_reward() == 0.0

    env.agent_pos = [9, 9]

    assert env._get_reward() == 1.0

    print("Reward test passed.")


if __name__ == "__main__":
    test_reward_only_at_goal()
