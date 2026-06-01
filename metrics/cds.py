import numpy as np

def cue_dependence_score(actions_with_cue, actions_without_cue):
    return np.mean(actions_with_cue == actions_without_cue)
