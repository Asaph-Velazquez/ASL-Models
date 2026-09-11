"""Evaluacion: WER, BLEU/ROUGE-L/EM, accuracy top-1/5, matriz de confusion y brecha sintetico->real."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# ---------------- metricas puras (importables sin ejecutar nada) ----------------

def wer_score(refs, hyps):
    """WER de secuencias de glosas (listas de strings 'A B C')."""
    import jiwer
    refs = [r if r.strip() else "<vacio>" for r in refs]
    hyps = [h if h.strip() else "<vacio>" for h in hyps]
    return float(jiwer.wer(refs, hyps))


def bleu_score(refs, hyps):
    import sacrebleu
    return float(sacrebleu.corpus_bleu(hyps, [refs]).score)


def rouge_l(refs, hyps):
    """ROUGE-L (media de F1 por oracion, LCS a nivel de palabra)."""
    def lcs(a, b):
        dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
        for i in range(len(a)):
            for j in range(len(b)):
                dp[i + 1][j + 1] = dp[i][j] + 1 if a[i] == b[j] \
                    else max(dp[i][j + 1], dp[i + 1][j])
        return dp[-1][-1]

    f1s = []
    for r, h in zip(refs, hyps):
        rt, ht = r.lower().split(), h.lower().split()
        if not rt or not ht:
            f1s.append(0.0)
            continue
        l = lcs(rt, ht)
        p, rec = l / len(ht), l / len(rt)
        f1s.append(0.0 if p + rec == 0 else 2 * p * rec / (p + rec))
    return float(np.mean(f1s)) if f1s else 0.0


def exact_match(refs, hyps):
    norm = lambda s: " ".join(s.lower().replace(".", "").replace(",", "").split())
    return float(np.mean([norm(r) == norm(h) for r, h in zip(refs, hyps)])) if refs else 0.0


# ---------------- evaluaciones por etapa ----------------

def eval_isolated(cfg, vocab, data_cfg, device, report):
    import torch
    from torch.utils.data import DataLoader
    from data.datasets import IsolatedDataset, collate_isolated, read_manifest
    from eval.plots import confusion_matrix, plot_confusion_matrix
    from models.heads import ClassificationModel

    ckpt_path = Path(cfg["checkpoints"]["isolated"])
    if not ckpt_path.is_file():
        report["aislado"] = {"estado": f"no disponible (falta {ckpt_path})"}
        return
    state = torch.load(ckpt_path, map_location=device)
    model = ClassificationModel(state["model_cfg"], state["input_dim"],
                                state["num_classes"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()

    manifest = read_manifest(Path(data_cfg["paths"]["cache_dir"]) / "clips_manifest.csv")
    ds = IsolatedDataset(manifest, vocab, data_cfg, "test", None, 128, cfg["seed"])
    if not len(ds):
        report["aislado"] = {"estado": "sin datos de test"}
        return
    dl = DataLoader(ds, batch_size=cfg["batch_size"], collate_fn=collate_isolated)
    ys, ps, top5 = [], [], 0
    with torch.no_grad():
        for x, lens, y in dl:
            logits = model(x.to(device), lens.to(device))
            k = min(5, logits.shape[1])
            top = logits.topk(k, dim=1).indices.cpu()
            ys += y.tolist()
            ps += top[:, 0].tolist()
            top5 += int(top.eq(y[:, None]).any(1).sum())
    top1 = float(np.mean([a == b for a, b in zip(ys, ps)]))
    cm = confusion_matrix(ys, ps, vocab.num_head_classes)
    out_png = Path(cfg["reports_dir"]) / "confusion_matrix.png"
    plot_confusion_matrix(cm, vocab.class_names, vocab.confusable_pairs(), out_png)
    report["aislado"] = {"top1": round(top1, 4), "top5": round(top5 / len(ys), 4),
                         "n_test": len(ys), "matriz_confusion": str(out_png)}
    print(f"[aislado] top-1={top1:.4f} top-5={top5 / len(ys):.4f} (n={len(ys)}) "
          f"-> {out_png}")


def eval_ctc(cfg, vocab, data_cfg, device, report):
    """WER en sintetico test Y frases reales test POR SEPARADO: la brecha es el dato clave.
    Devuelve filas para la tabla cualitativa."""
    import torch
    from torch.utils.data import DataLoader
    from data.datasets import PhraseCTCDataset, collate_ctc, load_phrase_items, read_manifest
    from models.ctc_decoder import GlossNGram, beam_search_decode
    from models.heads import CTCModel

    ckpt_path = Path(cfg["checkpoints"]["ctc"])
    if not ckpt_path.is_file():
        report["continuo"] = {"estado": f"no disponible (falta {ckpt_path})"}
        return []
    state = torch.load(ckpt_path, map_location=device)
    model = CTCModel(state["model_cfg"], state["input_dim"],
                     state["num_ctc_classes"]).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    dcfg = state.get("decode_cfg", {"beam_size": 8, "lm_ngram": 2,
                                    "lm_alpha": 0.3, "lm_beta": 0.6})

    lm = None
    syn_train = Path(cfg["synthetic_dir"]) / "synthetic_manifest_train.csv"
    if syn_train.is_file():
        seqs = [r["gloss_sequence"] for r in read_manifest(syn_train)]
        lm = GlossNGram.from_sequences(seqs, order=dcfg["lm_ngram"])

    cache = data_cfg["paths"]["cache_dir"]
    items = load_phrase_items(cache, cfg["synthetic_dir"], "test", 1.0)
    if not items:
        report["continuo"] = {"estado": "sin frases de test (sintetico ni real)"}
        return []

    rows, results = [], {}
    for kind in ("synthetic", "real"):
        sub = [i for i in items if i["kind"] == kind]
        if not sub:
            results[kind] = {"estado": "sin datos"}
            continue
        ds = PhraseCTCDataset(sub, vocab, data_cfg, 512)
        dl = DataLoader(ds, batch_size=cfg["batch_size"], collate_fn=collate_ctc)
        refs, hyps = [], []
        with torch.no_grad():
            for x, lens, targets, tlens, w, seqs in dl:
                logits = model(x.to(device), lens.to(device))
                logp = torch.log_softmax(logits, dim=-1).cpu().numpy()
                for i, seq in enumerate(seqs):
                    hyp = beam_search_decode(logp[i, :int(lens[i])], vocab,
                                             beam_size=dcfg["beam_size"], lm=lm,
                                             lm_alpha=dcfg["lm_alpha"],
                                             lm_beta=dcfg["lm_beta"])
                    refs.append(seq)
                    hyps.append(hyp)
        results[kind] = {"wer": round(wer_score(refs, hyps), 4), "n": len(refs)}
        rows += [{"glosa_real": r, "glosa_pred": h, "kind": kind}
                 for r, h in zip(refs, hyps)]
        print(f"[continuo/{kind}] WER={results[kind]['wer']:.4f} (n={len(refs)})")

    if "wer" in results.get("synthetic", {}) and "wer" in results.get("real", {}):
        gap = results["real"]["wer"] - results["synthetic"]["wer"]
        results["brecha_sintetico_a_real"] = round(gap, 4)
        print(f"[continuo] BRECHA sintetico->real: {gap:+.4f} "
              f"(la metrica REAL es la que vale)")
    else:
        results["brecha_sintetico_a_real"] = ("no medible: faltan frases reales validas "
                                              "o sintetico de test")
    report["continuo"] = results
    return rows


def eval_translation(cfg, vocab, device, report, gloss_rows):
    from data.datasets import read_manifest
    from models.translator import Translator, translate_glosses

    corpus = Path(cfg["synthetic_dir"]) / "parallel_corpus_test.csv"
    if not corpus.is_file():
        report["traduccion"] = {"estado": f"no disponible (falta {corpus})"}
        return []
    pairs = [(r["gloss_sequence"], r["english"]) for r in read_manifest(corpus)]

    translator = None
    tdir = Path(cfg["checkpoints"]["translator"]) / "best"
    if (tdir / "config.json").is_file():
        try:
            translator = Translator.from_pretrained(tdir, device=device)
        except Exception as e:
            print(f"AVISO: no se pudo cargar T5 de {tdir} ({e}); solo plantillas.")

    refs, hyps, fuente = [], [], {"t5": 0, "plantillas": 0}
    for g, ref in pairs:
        hyp, src = translate_glosses(g, vocab, translator)
        refs.append(ref)
        hyps.append(hyp)
        fuente[src] += 1
    report["traduccion"] = {
        "bleu": round(bleu_score(refs, hyps), 2),
        "rouge_l": round(rouge_l(refs, hyps), 4),
        "exact_match": round(exact_match(refs, hyps), 4),
        "n": len(refs), "fuente": fuente,
        "caveat": "ingles acotado a plantillas del dominio hotelero",
    }
    print(f"[traduccion] BLEU={report['traduccion']['bleu']} "
          f"ROUGE-L={report['traduccion']['rouge_l']} "
          f"EM={report['traduccion']['exact_match']} (n={len(refs)})")

    # ingles para la tabla cualitativa (sobre las glosas decodificadas del CTC)
    qrows = []
    ref_by_gloss = {g: e for g, e in pairs}
    for r in gloss_rows[:cfg["qualitative_n"]]:
        en_pred, _ = translate_glosses(r["glosa_pred"], vocab, translator)
        qrows.append({"glosa_real": r["glosa_real"], "glosa_pred": r["glosa_pred"],
                      "ingles_pred": en_pred,
                      "ingles_ref": ref_by_gloss.get(r["glosa_real"], "")})
    return qrows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/eval.yaml")
    args = ap.parse_args()
    run_eval(args.config)


def run_eval(config_path):
    from data.normalize import load_layout
    from data.vocab import Vocab
    from eval.plots import plot_training_curves, skeleton_animation
    from eval.qualitative import write_qualitative_table
    from train.common import get_device, set_seed
    from utils.config import load_config

    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    set_seed(cfg["seed"])
    device = get_device()
    rdir = Path(cfg["reports_dir"])
    rdir.mkdir(parents=True, exist_ok=True)
    vocab = Vocab(data_cfg["paths"]["vocab_csv"],
                  special_classes=data_cfg["folders"]["special_classes"])
    report = {}

    for fn, argsx in ((eval_isolated, (cfg, vocab, data_cfg, device, report)),):
        try:
            fn(*argsx)
        except Exception as e:
            report["aislado"] = {"estado": f"ERROR: {e}"}
            print(f"AVISO [aislado]: {e}")

    gloss_rows = []
    try:
        gloss_rows = eval_ctc(cfg, vocab, data_cfg, device, report)
    except Exception as e:
        report["continuo"] = {"estado": f"ERROR: {e}"}
        print(f"AVISO [continuo]: {e}")

    try:
        qrows = eval_translation(cfg, vocab, device, report, gloss_rows)
        if qrows:
            paths = write_qualitative_table(qrows, rdir)
            report["tabla_cualitativa"] = [str(p) for p in paths]
            print(f"[cualitativa] {paths[0]}")
    except Exception as e:
        report["traduccion"] = {"estado": f"ERROR: {e}"}
        print(f"AVISO [traduccion]: {e}")

    try:
        p = plot_training_curves([rdir / "tensorboard" / f"phase{i}" for i in (1, 2, 3, 4)],
                                 rdir / "training_curves.png")
        if p:
            report["curvas"] = str(p)
    except Exception as e:
        print(f"AVISO [curvas]: {e}")

    try:
        layout = load_layout(data_cfg["paths"]["cache_dir"])
        syn = sorted(Path(cfg["synthetic_dir"]).glob("syn_*.npy"))
        gifs = []
        for npy in syn[:cfg["skeleton_animation_n"]]:
            gifs.append(str(skeleton_animation(npy, layout,
                                               rdir / f"esqueleto_{npy.stem}.gif")))
        if gifs:
            report["animaciones_esqueleto"] = gifs
            print(f"[esqueleto] {len(gifs)} animaciones en {rdir}")
    except Exception as e:
        print(f"AVISO [esqueleto]: {e}")

    out = rdir / "metrics_summary.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nResumen de metricas -> {out}")
    return report


if __name__ == "__main__":
    main()
