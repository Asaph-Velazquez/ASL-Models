"""datasets pytorch sobre el cache de landmarks"""
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from data.normalize import (add_motion, load_layout, normalize_clip, path_for_class)


def read_manifest(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def expand_static(arr, n_frames, jitter_std, rng):
    """[1, D] -> [n_frames, D] con jitter leve"""
    out = np.repeat(arr, n_frames, axis=0).astype(np.float32)
    noise = rng.normal(0, jitter_std, out.shape).astype(np.float32)
    noise[:, -3:] = 0  # no tocar la mascara
    return out + noise


def subsample(arr, max_len):
    if len(arr) <= max_len:
        return arr
    idx = np.linspace(0, len(arr) - 1, max_len).round().astype(int)
    return arr[idx]


class IsolatedDataset(Dataset):
    def __init__(self, manifest_rows, vocab, data_cfg, split, sources=None,
                 max_seq_len=128, seed=42):
        self.vocab = vocab
        self.data_cfg = data_cfg
        self.layout = load_layout(data_cfg["paths"]["cache_dir"])
        self.max_seq_len = max_seq_len
        self.rng = np.random.default_rng(seed)
        self.hand_classes = data_cfg["normalization"]["hand_relative_classes"]
        self.s2s = data_cfg["static_to_sequence"]
        src_ok = {"videos": "isolated", "letters": "letters", "numbers": "numbers"}
        allowed = {src_ok[s] for s in sources} if sources else None
        self.rows = [r for r in manifest_rows
                     if r["split"] == split and (allowed is None or r["source"] in allowed)]

    def __len__(self):
        return len(self.rows)

    def class_counts(self):
        c = {}
        for r in self.rows:
            c[r["gloss"]] = c.get(r["gloss"], 0) + 1
        return c

    def _functional_class(self, gloss):
        if gloss in self.vocab.functional_class:
            return self.vocab.functional_class[gloss]
        return "letter"  # clases especiales (BLANK) vienen del set de alfabeto

    def __getitem__(self, i):
        r = self.rows[i]
        arr = np.load(r["npy_path"]).astype(np.float32)
        if r["kind"] == "image" or len(arr) == 1:
            arr = expand_static(arr[:1], self.s2s["n_frames"], self.s2s["jitter_std"], self.rng)
        arr = subsample(arr, self.max_seq_len)
        fc = self._functional_class(r["gloss"])
        path = path_for_class(fc, self.hand_classes)
        feats, _ = normalize_clip(arr, self.layout, path,
                                  self.data_cfg["normalization"]["interpolate_missing"])
        label = self.vocab.class2id[r["gloss"]]
        return torch.from_numpy(feats), label


class PhraseCTCDataset(Dataset):
    """frases continuas para ctc; items {npy_path, gloss_sequence, kind, weight}"""

    def __init__(self, items, vocab, data_cfg, max_seq_len=512):
        self.items = items
        self.vocab = vocab
        self.data_cfg = data_cfg
        self.layout = load_layout(data_cfg["paths"]["cache_dir"])
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        arr = np.load(it["npy_path"]).astype(np.float32)
        if it["kind"] == "synthetic":
            feats = arr  # el generador ya normalizo y agrego movimiento
        else:
            arr = subsample(arr, self.max_seq_len)
            feats, _ = normalize_clip(arr, self.layout, "body",
                                      self.data_cfg["normalization"]["interpolate_missing"])
        feats = subsample(feats, self.max_seq_len)
        targets = torch.tensor(self.vocab.encode_ctc(it["gloss_sequence"]), dtype=torch.long)
        return (torch.from_numpy(np.ascontiguousarray(feats)), targets,
                float(it.get("weight", 1.0)), it["gloss_sequence"])


def collate_isolated(batch):
    feats, labels = zip(*batch)
    lens = torch.tensor([len(f) for f in feats], dtype=torch.long)
    T = int(lens.max())
    D = feats[0].shape[1]
    out = torch.zeros(len(feats), T, D)
    for i, f in enumerate(feats):
        out[i, :len(f)] = f
    return out, lens, torch.tensor(labels, dtype=torch.long)


def collate_ctc(batch):
    feats, targets, weights, seqs = zip(*batch)
    lens = torch.tensor([len(f) for f in feats], dtype=torch.long)
    tlens = torch.tensor([len(t) for t in targets], dtype=torch.long)
    T = int(lens.max())
    D = feats[0].shape[1]
    out = torch.zeros(len(feats), T, D)
    for i, f in enumerate(feats):
        out[i, :len(f)] = f
    flat_targets = torch.cat(targets)
    return out, lens, flat_targets, tlens, torch.tensor(weights), list(seqs)


def load_phrase_items(cache_dir, synthetic_dir, split, real_weight=1.0,
                      how2sign_weight=0.0):
    """mezcla sintetico + real + how2sign para la fase 3"""
    items = []
    syn = Path(synthetic_dir) / f"synthetic_manifest_{split}.csv"
    if syn.is_file():
        for r in read_manifest(syn):
            items.append({"npy_path": str(Path(synthetic_dir) / f"{r['synthetic_id']}.npy"),
                          "gloss_sequence": r["gloss_sequence"], "kind": "synthetic",
                          "weight": 1.0})
    real = Path(cache_dir) / "phrases_manifest.csv"
    if real.is_file():
        for r in read_manifest(real):
            if r["split"] == split:
                items.append({"npy_path": r["npy_path"],
                              "gloss_sequence": r["gloss_sequence"], "kind": "real",
                              "weight": real_weight})
    h2s = Path(cache_dir) / "how2sign_phrases_manifest.csv"
    if how2sign_weight > 0 and h2s.is_file():
        for r in read_manifest(h2s):
            if r["split"] == split:
                items.append({"npy_path": r["npy_path"],
                              "gloss_sequence": r["gloss_sequence"], "kind": "how2sign",
                              "weight": how2sign_weight})
    return items
