import yaml
import json
import os

def load_yaml_config(path):
    """
    Loads YAML configuration file.
    """

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return config

def load_json_config(path):
    """
    Loads JSON configuration file.
    """

    with open(path, "r") as f:
        config = json.load(f)

    return config

def load_config(path):
    """
    Automatically detects file type.
    """

    ext = os.path.splitext(path)[-1]

    if ext in [".yaml", ".yml"]:
        return load_yaml_config(path)

    if ext == ".json":
        return load_json_config(path)

    raise ValueError(f"Unsupported config format: {ext}")
