import numpy as np

class CueLayer:
    def __init__(self, grid_size, num_cues=5):
        self.grid_size = grid_size
        self.num_cues = num_cues
        self.cues = self._generate_cues()

    def _generate_cues(self):
        return np.random.randint(
            0, self.grid_size, (self.num_cues, 2)
        )

    def get_cue_map(self):
        cue_map = np.zeros((self.grid_size, self.grid_size))
        for c in self.cues:
            cue_map[c[0], c[1]] = 1
        return cue_map

    def remove_cues(self):
        self.cues = []
