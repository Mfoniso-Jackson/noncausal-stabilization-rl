import os
import json
import numpy as np

class RunStorage:
    def __init__(self, base_dir="runs"):
        self.base_dir = base_dir

    def list_runs(self):
        return [
            os.path.join(self.base_dir, d)
            for d in os.listdir(self.base_dir)
            if os.path.isdir(os.path.join(self.base_dir, d))
        ]
    def load_run(self, run_path):
        data = {}

        for file in os.listdir(run_path):
            if file.endswith(".npy"):
                key = file.replace(".npy", "")
                data[key] = np.load(os.path.join(run_path, file))

            if file == "config.json":
                with open(os.path.join(run_path, file), "r") as f:
                    data["config"] = json.load(f)

        return data

      def aggregate_metric(self, metric_name):
        runs = self.list_runs()

        all_values = []

        for run in runs:
            path = os.path.join(run, f"{metric_name}.npy")

            if os.path.exists(path):
                values = np.load(path)
                all_values.append(values)

        return all_values

    def compute_summary(self, metric_name):
        runs = self.aggregate_metric(metric_name)

        flat = np.concatenate(runs)

        return {
            "mean": np.mean(flat),
            "std": np.std(flat),
            "min": np.min(flat),
            "max": np.max(flat)
        }

