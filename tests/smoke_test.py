import numpy as np
from env.gridworld import GridWorld


def test_smoke():
    env = GridWorld(size=10, max_steps=200)

    obs = env.reset()
    assert obs.shape == (2,)

    for _ in range(1000):
        action = np.random.randint(4)
        obs, reward, done, info = env.step(action)

        assert obs.shape == (2,)
        assert reward in [0.0, 1.0]
        assert "steps" in info

        if done:
            obs = env.reset()

    print("Smoke test passed.")


if __name__ == "__main__":
    test_smoke()
