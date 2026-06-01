import os
import argparse
import numpy as np
import torch

from env.wrapper import RLWrapper
from agents.dqn_agent import DQNAgent

from logging.logger import ExperimentLogger

from utils.config_loader import load_config
from utils.seed import set_seed

from metrics.entropy import episode_entropy

from analysis.plots import (
    plot_training_curve,
    plot_entropy_curve
)


def train_one_run(
    env,
    agent,
    logger,
    episodes=1000,
    checkpoint_dir="checkpoints"
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    reward_history = []
    entropy_history = []

    best_reward = -float("inf")

    for episode in range(episodes):

        obs = env.reset()
        done = False

        episode_reward = 0
        action_log = []

        while not done:

            action = agent.act(obs)

            next_obs, reward, done, _ = env.step(action)

            agent.update(
                (
                    obs,
                    action,
                    reward,
                    next_obs,
                    done
                )
            )

            episode_reward += reward
            action_log.append(action)

            obs = next_obs

        entropy = episode_entropy(action_log)

        reward_history.append(episode_reward)
        entropy_history.append(entropy)

        logger.log_step(
            reward=episode_reward,
            entropy=entropy
        )

        if episode_reward > best_reward:
            best_reward = episode_reward

            torch.save(
                agent.q_net.state_dict(),
                os.path.join(
                    checkpoint_dir,
                    "best_model.pt"
                )
            )

        if episode % 50 == 0:

            avg_reward = np.mean(
                reward_history[-50:]
            )

            avg_entropy = np.mean(
                entropy_history[-50:]
            )

            print(
                f"[Episode {episode}] "
                f"Reward={avg_reward:.4f} "
                f"Entropy={avg_entropy:.4f}"
            )

    logger.save()

    return reward_history, entropy_history


def load_model(
    agent,
    path
):
    state_dict = torch.load(
        path,
        map_location=torch.device("cpu")
    )

    agent.q_net.load_state_dict(state_dict)
    agent.target_net.load_state_dict(state_dict)

    return agent


def evaluate_agent(
    env,
    agent,
    episodes=100
):
    rewards = []

    for _ in range(episodes):

        obs = env.reset()
        done = False

        total_reward = 0

        while not done:

            action = agent.act(obs)

            obs, reward, done, _ = env.step(action)

            total_reward += reward

        rewards.append(total_reward)

    return {
        "mean_reward": np.mean(rewards),
        "std_reward": np.std(rewards),
        "all_rewards": rewards
    }


def build_agent(env):

    obs_dim = len(env.reset())
    action_dim = 4

    agent = DQNAgent(
        obs_dim=obs_dim,
        action_dim=action_dim
    )

    return agent


def run_training(config):

    env = RLWrapper(
        config["env"]
    )

    agent = build_agent(env)

    logger = ExperimentLogger(
        run_name="train"
    )

    logger.log_config(config)

    rewards, entropies = train_one_run(
        env=env,
        agent=agent,
        logger=logger,
        episodes=config["training"]["episodes"]
    )

    print("\nTraining complete.")

    results = evaluate_agent(
        env,
        agent,
        episodes=100
    )

    print(
        f"Evaluation Mean Reward: "
        f"{results['mean_reward']:.4f}"
    )

    plot_training_curve(rewards)

    plot_entropy_curve(entropies)

    return {
        "agent": agent,
        "rewards": rewards,
        "entropies": entropies,
        "evaluation": results
    }


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    set_seed(args.seed)

    config = load_config(
        args.config
    )

    print("=" * 80)
    print("NON-CAUSAL STABILIZATION RL TRAINING")
    print("=" * 80)

    run_training(config)


if __name__ == "__main__":
    main()
