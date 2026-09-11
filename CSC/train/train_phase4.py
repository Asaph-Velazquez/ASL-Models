"""FASE 4: traduccion glosa->ingles afinando t5-small con el corpus paralelo sintetico (acotado al dominio)."""
import argparse
import csv
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.translator import Translator
from train.common import RunLogger, get_device, set_seed
from utils.config import load_config


def load_corpus(dir_, split):
    p = Path(dir_) / f"parallel_corpus_{split}.csv"
    if not p.is_file():
        return []
    with open(p, newline="", encoding="utf-8-sig") as f:
        return [(r["gloss_sequence"], r["english"]) for r in csv.DictReader(f)]


def batches(pairs, bs):
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        yield [s for s, _ in chunk], [t for _, t in chunk]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase4.yaml")
    args = ap.parse_args()
    train_phase4(load_config(args.config))


def train_phase4(cfg, translator=None):
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = get_device()
    tcfg = cfg["train"]

    train_pairs = load_corpus(cfg["parallel_corpus_dir"], "train")
    val_pairs = load_corpus(cfg["parallel_corpus_dir"], "val") or train_pairs[:50]
    if not train_pairs:
        raise SystemExit("[phase4] Sin corpus paralelo. Corre antes el servicio synthetic.")
    print(f"[phase4] corpus: train={len(train_pairs)} val={len(val_pairs)}")

    if translator is None:  # el smoke test inyecta un Translator.tiny()
        translator = Translator.from_pretrained(cfg["model_name"],
                                                cfg["source_prefix"], device)
    opt = torch.optim.AdamW(translator.model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    logger = RunLogger(cfg["log_dir"], "phase4")
    ckpt_dir = Path(cfg["checkpoint_dir"])
    best_val, bad = float("inf"), 0

    rng = torch.Generator().manual_seed(cfg["seed"])
    t0 = time.time()
    for epoch in range(tcfg["epochs"]):
        order = torch.randperm(len(train_pairs), generator=rng).tolist()
        shuffled = [train_pairs[i] for i in order]
        tr_loss, nb = 0.0, 0
        for src, tgt in batches(shuffled, tcfg["batch_size"]):
            tr_loss += translator.train_step(src, tgt, opt, tcfg["max_source_len"],
                                             tcfg["max_target_len"])
            nb += 1
        va_loss, mv = 0.0, 0
        for src, tgt in batches(val_pairs, tcfg["batch_size"]):
            va_loss += translator.eval_loss(src, tgt, tcfg["max_source_len"],
                                            tcfg["max_target_len"])
            mv += 1
        metrics = {"train_loss": tr_loss / max(1, nb), "val_loss": va_loss / max(1, mv)}
        logger.log(epoch, metrics)
        print(f"[phase4] epoca {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        if metrics["val_loss"] < best_val:
            best_val = metrics["val_loss"]
            bad = 0
            translator.save(ckpt_dir / "best")
        else:
            bad += 1
        if bad >= tcfg["early_stop_patience"]:
            print(f"[phase4] early stop en epoca {epoch}.")
            break

    ejemplos = [s for s, _ in val_pairs[:3]]
    if ejemplos:
        outs = translator.translate(ejemplos, tcfg["max_target_len"])
        for g, e in zip(ejemplos, outs):
            print(f"    [{g}] -> {e}")
    logger.close()
    print(f"[phase4] mejor val_loss={best_val:.4f}; modelo en {ckpt_dir}/best "
          f"({time.time() - t0:.0f}s)")
    return translator


if __name__ == "__main__":
    main()
