"""FASE 1: pre-entreno del encoder con letras/numeros (imagenes -> mini-secuencia)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from train.common import train_classification
from utils.config import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase1.yaml")
    args = ap.parse_args()
    train_classification(load_config(args.config), "phase1")


if __name__ == "__main__":
    main()
