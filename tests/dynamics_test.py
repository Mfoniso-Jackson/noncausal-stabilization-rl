import numpy as np
from env.dynamics import DynamicsModel


def test_deterministic_dynamics():
    dyn = DynamicsModel(grid_size=10, noise=0.0)

    pos = [5, 5]

    assert dyn.transition(pos, 0) == [5, 6]
    assert dyn.transition(pos, 1) == [5, 4]
    assert dyn.transition(pos, 2) == [4, 5]
    assert dyn.transition(pos, 3) == [6, 5]

    print("Deterministic dynamics test passed.")


def test_boundary_conditions():
    dyn = DynamicsModel(grid_size=10, noise=0.0)

    assert dyn.transition([0, 0], 1) == [0, 0]
    assert dyn.transition([0, 0], 2) == [0, 0]
    assert dyn.transition([9, 9], 0) == [9, 9]
    assert dyn.transition([9, 9], 3) == [9, 9]

    print("Boundary dynamics test passed.")


def test_stochastic_dynamics():
    np.random.seed(42)

    dyn = DynamicsModel(grid_size=10, noise=1.0)

    pos = [5, 5]
    outcomes = set()

    for _ in range(100):
        outcomes.add(tuple(dyn.transition(pos, 0)))

    assert len(outcomes) > 1, "Noise is not changing transitions."

    print("Stochastic dynamics test passed.")


if __name__ == "__main__":
    test_deterministic_dynamics()
    test_boundary_conditions()
    test_stochastic_dynamics()
