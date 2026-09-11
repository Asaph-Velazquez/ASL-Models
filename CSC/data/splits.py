"""splits automaticos, signer-independent por defecto"""
import csv
import random
from collections import defaultdict
from pathlib import Path


def load_signer_manifest(manifest_csv, dataset_root):
    """dict path resuelto -> signer_id; rutas 'Dataset/...' se reanclan en dataset_root"""
    mapping = {}
    if not Path(manifest_csv).is_file():
        return mapping
    root = Path(dataset_root)
    with open(manifest_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            raw = r["video_path"].strip()
            parts = Path(raw).parts
            if parts and parts[0].lower() == root.name.lower():
                resolved = root / Path(*parts[1:])
            else:
                resolved = Path(raw)
            mapping[str(resolved)] = r["signer_id"].strip()
    return mapping


def _split_signers(signers, test_fraction, seed):
    signers = sorted(signers)
    rng = random.Random(seed)
    rng.shuffle(signers)
    n_test = max(1, round(len(signers) * test_fraction)) if len(signers) > 1 else 0
    return set(signers[n_test:]), set(signers[:n_test])  # train, test


def assign_splits(samples, signer_map, test_fraction=0.2, seed=42):
    """anota cada sample con split y signer_id"""
    warnings = []

    # paso 1: split_hint de letras/numeros manda
    hinted = [s for s in samples if s["split_hint"]]
    for s in hinted:
        s["split"] = s["split_hint"]
        s["signer_id"] = None

    # paso 2: signer-independent si hay firmante, si no por-video
    rest = [s for s in samples if not s["split_hint"]]
    for s in rest:
        s["signer_id"] = signer_map.get(s["path"])

    with_signer = [s for s in rest if s["signer_id"]]
    without = [s for s in rest if not s["signer_id"]]

    if with_signer:
        train_sg, test_sg = _split_signers({s["signer_id"] for s in with_signer},
                                           test_fraction, seed)
        for s in with_signer:
            s["split"] = "train" if s["signer_id"] in train_sg else "test"
        warnings.append(f"Split signer-independent: {len(train_sg)} firmantes a train, "
                        f"{len(test_sg)} a test.")

    if without:
        warnings.append(f"AVISO: {len(without)} videos sin signer_id en el manifiesto -> "
                        f"split por-video estratificado (NO signer-independent).")
        by_gloss = defaultdict(list)
        for s in without:
            by_gloss[s["gloss"]].append(s)
        rng = random.Random(seed)
        for gloss, group in sorted(by_gloss.items()):
            rng.shuffle(group)
            n_test = int(len(group) * test_fraction)
            for i, s in enumerate(group):
                s["split"] = "test" if i < n_test else "train"

    # paso 3: toda clase con >=1 ejemplo en train
    by_gloss = defaultdict(lambda: {"train": [], "test": []})
    for s in samples:
        by_gloss[s["gloss"]][s["split"]].append(s)
    for gloss, g in sorted(by_gloss.items()):
        if not g["train"] and g["test"]:
            moved = g["test"][0]
            moved["split"] = "train"
            warnings.append(f"AVISO: '{gloss}' no tenia ejemplos en train; se movio 1 desde "
                            f"test. Su test NO es representativo.")
        n = len(g["train"]) + len(g["test"])
        if n < 4:
            warnings.append(f"AVISO: '{gloss}' tiene solo {n} ejemplos; metricas poco fiables.")
    return samples, warnings


def assign_phrase_splits(phrases, test_fraction=0.2, seed=42):
    """split de frases: signer-independent si hay signer_id, si no por-video"""
    warnings = []
    if not phrases:
        return phrases, warnings
    with_signer = [p for p in phrases if p.get("signer_id")]
    if with_signer and len(with_signer) == len(phrases):
        train_sg, test_sg = _split_signers({p["signer_id"] for p in phrases},
                                           test_fraction, seed)
        for p in phrases:
            p["split"] = "train" if p["signer_id"] in train_sg else "test"
        warnings.append("Frases: split signer-independent.")
    else:
        warnings.append("AVISO: frases sin signer_id -> split por-video (NO signer-independent).")
        ps = sorted(phrases, key=lambda p: p["path"])
        rng = random.Random(seed)
        rng.shuffle(ps)
        n_test = max(1, int(len(ps) * test_fraction)) if len(ps) > 1 else 0
        for i, p in enumerate(ps):
            p["split"] = "test" if i < n_test else "train"
    return phrases, warnings
