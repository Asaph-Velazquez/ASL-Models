from pathlib import Path
import yaml


def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    if "data_config" in cfg and cfg["data_config"]:
        with open(cfg["data_config"]) as f:
            cfg["data"] = yaml.safe_load(f)
    return cfg


def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
