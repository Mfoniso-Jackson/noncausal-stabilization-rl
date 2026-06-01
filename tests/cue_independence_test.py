import numpy as np
from env.wrapper import RLWrapper


def make_config(num_cues):
    return {
        "grid_size": 10,
        "max_steps": 200,
        "num_cues": num_cues,
        "reward_sparse_prob": 1.0,
        "delay": 0,
        "noise": 0.0,
    }


def run_random_policy(config, episodes=100):
    env = RLWrapper(config)
    total_rewards = []

    for _ in range(episodes):
        obs = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = np.random.randint(4)
            obs, reward, done, info = env.step(action)
            ep_reward += reward

        total_rewards.append(ep_reward)

    return np.mean(total_rewards)


def test_cue_independence():
    np.random.seed(42)

    reward_with_cues = run_random_policy(make_config(num_cues=5))
    reward_without_cues = run_random_policy(make_config(num_cues=0))

    diff = abs(reward_with_cues - reward_without_cues)

    assert diff < 0.2, (
        f"Cue independence failed. "
        f"with_cues={reward_with_cues}, "
        f"without_cues={reward_without_cues}, "
        f"diff={diff}"
    )

    print("Cue independence test passed.")


if __name__ == "__main__":
    test_cue_independence()
