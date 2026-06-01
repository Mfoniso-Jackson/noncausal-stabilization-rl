from env.gridworld import GridWorld
from env.cues import CueLayer

def run_intervention(agent, cues):
    cues.remove_cues()  # CRITICAL TEST

    for episode in range(100):
        obs = env.reset()
        done = False

        while not done:
            cue_map = cues.get_cue_map()
            state = np.concatenate([obs, cue_map.flatten()])

            action = agent.act(state)

            obs, _, done, _ = env.step(action)
