"""FASE 2: clasificacion aislada sobre todo el vocabulario, inicializando el encoder desde la Fase 1."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.common import train_classification
from utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase2.yaml")
    args = ap.parse_args()
    train_classification(load_config(args.config), "phase2")


if __name__ == "__main__":
    main()
