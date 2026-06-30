import itertools
import numpy as np

from env.wrapper import RLWrapper
from agents.dqn_agent import DQNAgent
from experiments.train import train_one_run

def generate_ablation_configs():
    sparsity = [0.05, 0.1, 0.3]
    noise = [0.0, 0.1, 0.3]
    delay = [0, 3, 6]

    configs = []

    for s, n, d in itertools.product(sparsity, noise, delay):
        configs.append({
            "reward_sparse_prob": s,
            "noise": n,
            "delay": d,
            "grid_size": 10,
            "num_cues": 5
        })

    return configs

def run_ablation(config, seed=0):
    np.random.seed(seed)

    env = RLWrapper(config)

    obs_dim = 10 + 10*10   # grid + cues (approx)
    action_dim = 4

    agent = DQNAgent(obs_dim, action_dim)

    rewards = train_one_run(env, agent, timesteps=5000)

    return {
        "config": config,
        "mean_reward": np.mean(rewards),
        "final_reward": np.mean(rewards[-100:])
    }

def run_full_ablation():
    configs = generate_ablation_configs()

    results = []

    for i, config in enumerate(configs):
        print(f"Running ablation {i+1}/{len(configs)}")

        result = run_ablation(config, seed=i)
        results.append(result)

    return results

def analyze_uncertainty_effect(results):
    """
    Extract relationship between uncertainty and performance collapse / stabilization.
    """

    summary = []

    for r in results:
        u = r["config"]["noise"] + r["config"]["delay"] + r["config"]["reward_sparse_prob"]

        summary.append({
            "uncertainty": u,
            "performance": r["mean_reward"]
        })

    return summary
