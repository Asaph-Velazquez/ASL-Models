"""Utilidades comunes de entrenamiento: semillas, device, logging, checkpoints y loop de clasificacion."""
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import IsolatedDataset, collate_isolated, read_manifest
from data.vocab import Vocab
from models.heads import ClassificationModel, transfer_encoder
from utils.trace import TraceRecorder


def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def get_device():
    """CUDA cuando este disponible y usable; si falla, cae a CPU."""
    if not torch.cuda.is_available():
        print("Device: CPU (CUDA no disponible)")
        return torch.device("cpu")

    try:
        probe = nn.Linear(1, 1)
        probe.to("cuda")
        sample = torch.randn(1, 1, device="cuda")
        _ = probe(sample)
        torch.cuda.synchronize()
        print(f"Device: CUDA ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    except Exception as exc:
        print(f"AVISO: CUDA no usable ({exc}); usando CPU")
        return torch.device("cpu")


class RunLogger:
    """CSV siempre; TensorBoard si el paquete esta disponible."""

    def __init__(self, log_dir, phase):
        self.dir = Path(log_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.dir / f"{phase}_log.csv"
        self._csv = None
        self._writer = None
        self.tb = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tb = SummaryWriter(str(self.dir))
        except Exception as e:
            print(f"AVISO: TensorBoard no disponible ({e}); logging solo CSV.")

    def log(self, step, metrics):
        if self._csv is None:
            self._csv = open(self.csv_path, "a", newline="")
            self._writer = csv.DictWriter(self._csv, fieldnames=["step"] + sorted(metrics))
            if self.csv_path.stat().st_size == 0:
                self._writer.writeheader()
        self._writer.writerow({"step": step, **{k: round(float(v), 6)
                                                for k, v in metrics.items()}})
        self._csv.flush()
        if self.tb:
            for k, v in metrics.items():
                self.tb.add_scalar(k, v, step)

    def close(self):
        if self._csv:
            self._csv.close()
        if self.tb:
            self.tb.close()


class Checkpointer:
    def __init__(self, ckpt_dir):
        self.dir = Path(ckpt_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save(self, state, is_best):
        torch.save(state, self.dir / "last.pt")
        if is_best:
            torch.save(state, self.dir / "best.pt")

    def resume(self):
        """Devuelve el estado del ultimo checkpoint si existe (reanudacion automatica)."""
        last = self.dir / "last.pt"
        if last.is_file():
            print(f"Reanudando desde {last}")
            return torch.load(last, map_location="cpu")
        return None

    def write_summary(self, summary):
        (self.dir / "summary.json").write_text(json.dumps(summary, indent=2,
                                                          ensure_ascii=False))


def topk_accuracy(logits, labels, ks=(1, 5)):
    out = {}
    maxk = min(max(ks), logits.shape[1])
    _, pred = logits.topk(maxk, dim=1)
    correct = pred.eq(labels[:, None])
    for k in ks:
        out[f"top{k}"] = float(correct[:, :min(k, maxk)].any(dim=1).float().mean())
    return out


def train_classification(cfg, phase_name):
    """Loop compartido de las Fases 1 y 2 (cabeza 4A). Devuelve resumen reproducible."""
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = get_device()
    data_cfg = cfg["data"]
    tcfg = cfg["train"]
    tracer = TraceRecorder(enabled=cfg.get("trace", False),
                           reports_dir=data_cfg["paths"]["reports_dir"])

    vocab = Vocab(data_cfg["paths"]["vocab_csv"],
                  special_classes=data_cfg["folders"]["special_classes"])
    manifest = read_manifest(Path(data_cfg["paths"]["cache_dir"]) / "clips_manifest.csv")
    ds_train = IsolatedDataset(manifest, vocab, data_cfg, "train", cfg["sources"],
                               tcfg["max_seq_len"], cfg["seed"])
    ds_test = IsolatedDataset(manifest, vocab, data_cfg, "test", cfg["sources"],
                              tcfg["max_seq_len"], cfg["seed"])
    if len(ds_train) == 0:
        raise SystemExit(f"[{phase_name}] Sin datos de train para sources={cfg['sources']}. "
                         f"¿Corriste preprocess?")
    print(f"[{phase_name}] train={len(ds_train)} test={len(ds_test)} "
          f"clases={vocab.num_head_classes} (N={vocab.N} + especiales)")

    dl_train = DataLoader(ds_train, batch_size=tcfg["batch_size"], shuffle=True,
                          num_workers=tcfg["num_workers"], collate_fn=collate_isolated)
    dl_test = DataLoader(ds_test, batch_size=tcfg["batch_size"], shuffle=False,
                         num_workers=tcfg["num_workers"], collate_fn=collate_isolated)

    sample_feats, _ = ds_train[0]
    input_dim = sample_feats.shape[1]  # derivado de los datos, jamas hardcodeado
    model = ClassificationModel(cfg["model"], input_dim, vocab.num_head_classes).to(device)

    if cfg.get("init_from"):
        if Path(cfg["init_from"]).is_file():
            info = transfer_encoder(model, cfg["init_from"], device)
            print(f"[{phase_name}] encoder inicializado desde {cfg['init_from']}: {info}")
        else:
            print(f"AVISO [{phase_name}]: init_from={cfg['init_from']} no existe; "
                  f"entrenando desde cero.")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    crit = nn.CrossEntropyLoss()
    logger = RunLogger(cfg["log_dir"], phase_name)
    ckpt = Checkpointer(cfg["checkpoint_dir"])

    start_epoch, best_acc, bad_epochs = 0, -1.0, 0
    state = ckpt.resume()
    if state and state.get("phase") == phase_name:
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_acc = state.get("best_acc", -1.0)

    freeze_epochs = cfg.get("freeze_spatial_epochs", 0)
    t0 = time.time()
    epoch = start_epoch
    for epoch in range(start_epoch, tcfg["epochs"]):
        for p in model.encoder.spatial.parameters():
            p.requires_grad = epoch >= freeze_epochs
        model.train()
        tr_loss, n = 0.0, 0
        for bi, (x, lens, y) in enumerate(dl_train):
            x, lens, y = x.to(device), lens.to(device), y.to(device)
            if epoch == start_epoch and bi == 0 and tracer.enabled:
                _trace_classification_batch(tracer, model, x, lens, y, vocab)
            logits = model(x, lens)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr_loss += float(loss) * len(y)
            n += len(y)

        model.eval()
        accs, va_loss, m = {"top1": 0.0, "top5": 0.0}, 0.0, 0
        with torch.no_grad():
            for x, lens, y in dl_test:
                x, lens, y = x.to(device), lens.to(device), y.to(device)
                logits = model(x, lens)
                va_loss += float(crit(logits, y)) * len(y)
                a = topk_accuracy(logits, y)
                for k in accs:
                    accs[k] += a[k] * len(y)
                m += len(y)
        accs = {k: v / max(1, m) for k, v in accs.items()}
        metrics = {"train_loss": tr_loss / max(1, n),
                   "val_loss": va_loss / max(1, m) if m else 0.0,
                   "val_top1": accs["top1"], "val_top5": accs["top5"]}
        logger.log(epoch, metrics)
        print(f"[{phase_name}] epoca {epoch}: " +
              " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        is_best = accs["top1"] > best_acc
        if is_best:
            best_acc = accs["top1"]
            bad_epochs = 0
        else:
            bad_epochs += 1
        ckpt.save({"phase": phase_name, "epoch": epoch, "model": model.state_dict(),
                   "optimizer": opt.state_dict(), "best_acc": best_acc,
                   "input_dim": input_dim, "num_classes": vocab.num_head_classes,
                   "model_cfg": cfg["model"]}, is_best)
        if bad_epochs >= tcfg["early_stop_patience"]:
            print(f"[{phase_name}] early stop en epoca {epoch}.")
            break

    summary = {"phase": phase_name, "best_val_top1": best_acc,
               "epochs_run": epoch - start_epoch + 1 if tcfg["epochs"] else 0,
               "train_samples": len(ds_train), "test_samples": len(ds_test),
               "num_classes": vocab.num_head_classes, "input_dim": input_dim,
               "seed": cfg["seed"], "seconds": round(time.time() - t0, 1)}
    ckpt.write_summary(summary)
    logger.close()
    tracer.save(f"pipeline_trace_{phase_name}")
    print(f"[{phase_name}] mejor top-1: {best_acc:.4f}. Checkpoints en {cfg['checkpoint_dir']}")
    return summary


def _trace_classification_batch(tracer, model, x, lens, y, vocab):
    import torch as _t
    with tracer.stage("Capa 2", "encoder espacial por frame (MLP/GCN)") as st:
        with _t.no_grad():
            h_sp = model.encoder.spatial(x)
        st.io(inp=x, out=h_sp)
    with tracer.stage("Capa 3", "encoder temporal (Bi-LSTM/TCN)") as st:
        with _t.no_grad():
            h_tp = model.encoder.temporal(h_sp, lens)
        st.io(inp=h_sp, out=h_tp)
    with tracer.stage("Capa 4A", f"cabeza clasificacion ({vocab.num_head_classes} clases)") as st:
        with _t.no_grad():
            logits = model(x, lens)
        st.io(inp=h_tp, out=logits)
        a = topk_accuracy(logits, y)
        st.metric("accuracy_batch_top1", round(a["top1"], 4))
        conf = float(_t.softmax(logits, dim=1).max(dim=1).values.mean())
        st.metric("confianza_media", round(conf, 4))
