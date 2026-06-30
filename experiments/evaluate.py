import numpy as np
import torch

from env.wrapper import RLWrapper
from agents.dqn_agent import DQNAgent
from metrics.cds import cue_dependence_score
from metrics.spi import superstition_persistence_index
from analysis.heatmaps import plot_heatmap

def evaluate(agent, env, episodes=50):
    rewards = []
    trajectories = []

    for _ in range(episodes):
        obs = env.reset()
        done = False

        trajectory = []

        while not done:
            action = agent.act(obs)

            next_obs, reward, done, _ = env.step(action)

            trajectory.append(env.grid.agent_pos)

            obs = next_obs
            rewards.append(reward)

        trajectories.append(trajectory)

    return rewards, trajectories

def cue_removal_test(agent, env, episodes=50):
    """
    Key experiment:
    Train agent assumes cues exist → remove cues → test persistence.
    """

    env.remove_cues()

    rewards, trajectories = evaluate(agent, env, episodes)

    return rewards, trajectories

def trajectory_heatmap(trajectories, grid_size=10):
    heatmap = np.zeros((grid_size, grid_size))

    for traj in trajectories:
        for x, y in traj:
            heatmap[x, y] += 1

    heatmap /= len(trajectories)
    return heatmap

def run_evaluation(agent, config):
    env = RLWrapper(config)

    print("Running standard evaluation...")
    rewards, trajs = evaluate(agent, env)

    heatmap_before = trajectory_heatmap(trajs)

    print("Running cue removal intervention...")
    rewards_removed, trajs_removed = cue_removal_test(agent, env)

    heatmap_after = trajectory_heatmap(trajs_removed)

    # plot
    plot_heatmap(heatmap_before)
    plot_heatmap(heatmap_after)

    return {
        "rewards": rewards,
        "rewards_removed": rewards_removed,
        "heatmap_before": heatmap_before,
        "heatmap_after": heatmap_after
    }
