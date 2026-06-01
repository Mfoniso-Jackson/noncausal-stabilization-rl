import os
import json
import numpy as np
from datetime import datetime

class ExperimentLogger:
    def __init__(self, run_name="experiment", save_dir="runs"):
        self.run_name = run_name
        self.save_dir = save_dir

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_path = os.path.join(save_dir, f"{run_name}_{self.timestamp}")

        os.makedirs(self.run_path, exist_ok=True)

        self.metrics = {
            "reward": [],
            "entropy": [],
            "cue_dependence": [],
            "visitation": []
        }

        self.config = None

      def log_config(self, config):
        self.config = config

        with open(os.path.join(self.run_path, "config.json"), "w") as f:
            json.dump(config, f, indent=4)

      def log_step(self, reward=None, entropy=None, cue_dependence=None, visitation=None):
        if reward is not None:
            self.metrics["reward"].append(reward)

        if entropy is not None:
            self.metrics["entropy"].append(entropy)

        if cue_dependence is not None:
            self.metrics["cue_dependence"].append(cue_dependence)

        if visitation is not None:
            self.metrics["visitation"].append(visitation)

    def save(self):
        for key, value in self.metrics.items():
            np.save(
                os.path.join(self.run_path, f"{key}.npy"),
                np.array(value)
            )

    def load(self, path):
        self.run_path = path

        for key in self.metrics.keys():
            file_path = os.path.join(path, f"{key}.npy")
            if os.path.exists(file_path):
                self.metrics[key] = np.load(file_path).tolist()



