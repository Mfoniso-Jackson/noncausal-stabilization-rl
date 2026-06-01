from env.gridworld import GridWorld
from env.cues import CueLayer
from agents.ppo_agent import PPOAgent
import numpy as np

def train():
    env = GridWorld()
    cues = CueLayer(10)

    agent = PPOAgent(obs_dim=2, action_dim=4)

    for episode in range(1000):
        obs = env.reset()
        done = False

        while not done:
            cue_map = cues.get_cue_map()
            state = np.concatenate([obs, cue_map.flatten()])

            action = agent.act(state)

            next_obs, reward, done, _ = env.step(action)

            agent.update((state, action, reward))

            obs = next_obs

        print(f"Episode {episode} complete")

if __name__ == "__main__":
    train()
