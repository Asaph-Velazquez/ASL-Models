"""FASE 3: CTC continuo mezclando sintetico + frases reales + How2Sign (pseudo-glosas, peso debil)."""
import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.datasets import PhraseCTCDataset, collate_ctc, load_phrase_items
from data.vocab import Vocab
from eval.metrics import wer_score
from models.ctc_decoder import GlossNGram, greedy_decode
from models.heads import CTCModel, transfer_encoder
from train.common import Checkpointer, RunLogger, get_device, set_seed
from utils.config import load_config
from utils.trace import TraceRecorder


def run_ctc_epoch(model, loader, vocab, device, opt=None, filter_kind=None):
    training = opt is not None
    model.train() if training else model.eval()
    tot_loss, n, refs, hyps = 0.0, 0, [], []
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for x, lens, targets, tlens, weights, seqs in loader:
            if filter_kind is not None:
                keep = [i for i, w in enumerate(weights.tolist())
                        if (w > 1.0) == (filter_kind == "real")]
                if not keep:
                    continue
            x, lens = x.to(device), lens.to(device)
            targets, tlens = targets.to(device), tlens.to(device)
            logits = model(x, lens)                       # [B, T, C+1]
            logp = F.log_softmax(logits, dim=-1).transpose(0, 1)  # [T, B, C+1]
            losses = F.ctc_loss(logp, targets, lens, tlens, blank=vocab.ctc_blank,
                                reduction="none", zero_infinity=True)
            losses = losses / tlens.clamp(min=1).float()
            loss = (losses * weights.to(device)).mean()
            if training:
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            else:
                lp = logp.transpose(0, 1).cpu().numpy()
                for i, seq in enumerate(seqs):
                    hyps.append(greedy_decode(lp[i, :int(lens[i])], vocab))
                    refs.append(seq)
            tot_loss += float(loss) * len(seqs)
            n += len(seqs)
    out = {"loss": tot_loss / max(1, n), "n": n}
    if refs:
        out["wer"] = wer_score(refs, hyps)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/phase3.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    train_phase3(cfg)


def train_phase3(cfg):
    set_seed(cfg["seed"], cfg.get("deterministic", True))
    device = get_device()
    data_cfg = cfg["data"]
    tcfg = cfg["train"]
    cache = data_cfg["paths"]["cache_dir"]
    tracer = TraceRecorder(enabled=cfg.get("trace", False),
                           reports_dir=data_cfg["paths"]["reports_dir"])

    vocab = Vocab(data_cfg["paths"]["vocab_csv"],
                  special_classes=data_cfg["folders"]["special_classes"])
    h2s_w = float(tcfg.get("how2sign_weight", 0.0))
    h2s_eval_w = h2s_w if tcfg.get("how2sign_in_val", False) else 0.0
    items_tr = load_phrase_items(cache, cfg["synthetic_dir"], "train",
                                 tcfg["real_weight"], h2s_w)
    items_va = load_phrase_items(cache, cfg["synthetic_dir"], "val", 1.0, h2s_eval_w)
    items_te = load_phrase_items(cache, cfg["synthetic_dir"], "test", 1.0, h2s_eval_w)
    if not items_tr:
        raise SystemExit("[phase3] Sin frases de entrenamiento. Corre preprocess y synthetic.")
    n_real = sum(1 for i in items_tr if i["kind"] == "real")
    n_h2s = sum(1 for i in items_tr if i["kind"] == "how2sign")
    print(f"[phase3] train={len(items_tr)} (reales={n_real}, how2sign={n_h2s} "
          f"peso={h2s_w}) val={len(items_va)} test={len(items_te)} | "
          f"vocab CTC={vocab.num_ctc_classes}")
    if n_real == 0:
        print("AVISO [phase3]: NO hay frases reales validas; el puente aislado->continuo "
              "queda solo sintetico y la brecha real no podra medirse.")

    ds_tr_syn = PhraseCTCDataset([i for i in items_tr if i["kind"] == "synthetic"],
                                 vocab, data_cfg, tcfg["max_seq_len"])
    ds_tr_all = PhraseCTCDataset(items_tr, vocab, data_cfg, tcfg["max_seq_len"])
    ds_va = PhraseCTCDataset(items_va, vocab, data_cfg, tcfg["max_seq_len"])
    mk = lambda ds, sh: DataLoader(ds, batch_size=tcfg["batch_size"], shuffle=sh,
                                   num_workers=tcfg["num_workers"], collate_fn=collate_ctc)

    sample = ds_tr_all[0][0]
    input_dim = sample.shape[1]
    model = CTCModel(cfg["model"], input_dim, vocab.num_ctc_classes).to(device)
    if cfg.get("init_from") and Path(cfg["init_from"]).is_file():
        info = transfer_encoder(model, cfg["init_from"], device)
        print(f"[phase3] encoder desde {cfg['init_from']}: {info}")
    else:
        print(f"AVISO [phase3]: sin init_from valido; CTC desde cero.")

    opt = torch.optim.AdamW(model.parameters(), lr=tcfg["lr"],
                            weight_decay=tcfg["weight_decay"])
    logger = RunLogger(cfg["log_dir"], "phase3")
    ckpt = Checkpointer(cfg["checkpoint_dir"])

    start_epoch, best_wer, bad = 0, float("inf"), 0
    state = ckpt.resume()
    if state and state.get("phase") == "phase3":
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_wer = state.get("best_wer", float("inf"))

    t0 = time.time()
    epoch = start_epoch
    for epoch in range(start_epoch, tcfg["epochs"]):
        use_syn_only = epoch < cfg.get("synthetic_only_epochs", 0) and len(ds_tr_syn) > 0
        loader = mk(ds_tr_syn if use_syn_only else ds_tr_all, True)
        tr = run_ctc_epoch(model, loader, vocab, device, opt)
        va = run_ctc_epoch(model, mk(ds_va, False), vocab, device) if len(ds_va) else {"loss": 0, "wer": 1.0}
        metrics = {"train_loss": tr["loss"], "val_loss": va["loss"],
                   "val_wer": va.get("wer", 1.0),
                   "curriculo_solo_sintetico": 1.0 if use_syn_only else 0.0}
        logger.log(epoch, metrics)
        print(f"[phase3] epoca {epoch}: " + " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        if tracer.enabled and epoch == start_epoch:
            _trace_ctc(tracer, model, mk(ds_va if len(ds_va) else ds_tr_all, False),
                       vocab, device)

        is_best = metrics["val_wer"] < best_wer
        if is_best:
            best_wer = metrics["val_wer"]
            bad = 0
        else:
            bad += 1
        ckpt.save({"phase": "phase3", "epoch": epoch, "model": model.state_dict(),
                   "optimizer": opt.state_dict(), "best_wer": best_wer,
                   "input_dim": input_dim, "num_ctc_classes": vocab.num_ctc_classes,
                   "model_cfg": cfg["model"], "decode_cfg": cfg["decode"]}, is_best)
        if bad >= tcfg["early_stop_patience"]:
            print(f"[phase3] early stop en epoca {epoch}.")
            break

    ckpt.write_summary({"phase": "phase3", "best_val_wer": best_wer, "seed": cfg["seed"],
                        "train_items": len(items_tr), "real_items": n_real,
                        "how2sign_items": n_h2s, "how2sign_weight": h2s_w,
                        "input_dim": input_dim, "seconds": round(time.time() - t0, 1)})
    logger.close()
    tracer.save("pipeline_trace_phase3")
    print(f"[phase3] mejor WER val (sintetico): {best_wer:.4f}. "
          f"El WER que VALE es el de frases reales: correr eval.")


def _trace_ctc(tracer, model, loader, vocab, device):
    import numpy as np
    batch = next(iter(loader), None)
    if batch is None:
        return
    x, lens, targets, tlens, weights, seqs = batch
    with torch.no_grad():
        logits = model(x.to(device), lens.to(device))
        logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
    hyps = [greedy_decode(logp[i, :int(lens[i])], vocab) for i in range(len(seqs))]
    with tracer.stage("Capa 4B", "logits CTC por frame") as st:
        st.io(inp=x, out=logits)
        st.metric("wer_batch", round(wer_score(list(seqs), hyps), 4))
    with tracer.stage("Capa 5", "decodificacion CTC -> glosas") as st:
        st.io(inp=logits, out=None)
        st.metric("glosas_decodificadas", hyps[:3])
        st.metric("glosas_referencia", list(seqs)[:3])


if __name__ == "__main__":
    main()
