"concatena clips aislados en frases CTC + corpus paralelo ingles (T5)"
import argparse
import csv
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.vocab import Vocab
from synthetic.english_templates import TEMPLATES, build_english
from utils.config import load_config


def load_clip_index(clips_csv, split):
    index = defaultdict(list)
    with open(clips_csv, newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r.get("split", "train") != split:
                continue
            index[r["gloss"]].append((r["npy_path"], r["signer_id"] or "sin_firmante"))
    return dict(index)


def plan_sentence(vocab_by_class, available, rng):
    branch = rng.choice(list(TEMPLATES.keys()))
    template = rng.choice(TEMPLATES[branch])
    items = []
    for slot, spec, prob in template:
        if rng.random() > prob:
            continue
        choices = spec["gloss"] if "gloss" in spec else \
            [g for c in spec["class"] for g in vocab_by_class.get(c, [])]
        choices = [g for g in choices if g in available]
        if not choices:
            continue
        g = rng.choice(choices)
        if items and items[-1][1] == g:
            continue
        items.append((slot, g))
    return branch, items


def pick_clips(glosses, clip_index, rng):
    """Prefiere clips del MISMO firmante para toda la frase (consistencia corporal)."""
    common = None
    for g in glosses:
        signers = {s for _, s in clip_index.get(g, [])}
        common = signers if common is None else (common & signers)
    if common:
        s = rng.choice(sorted(common))
        return [rng.choice([p for p, sg in clip_index[g] if sg == s]) for g in glosses], [s]
    paths, signers = [], []
    for g in glosses:
        p, sg = rng.choice(clip_index[g])
        paths.append(p)
        signers.append(sg)
    return paths, sorted(set(signers))


def trim_rest(pos, thresh_ratio=0.15, pad=2):
    """Recorta reposo inicial/final por velocidad (evita 'pausas' irreales al concatenar)."""
    if len(pos) < 3:
        return pos
    vel = np.linalg.norm(np.diff(pos, axis=0), axis=1)
    if vel.max() == 0:
        return pos
    active = np.where(vel > thresh_ratio * vel.max())[0]
    if len(active) == 0:
        return pos
    return pos[max(0, active[0] - pad): min(len(pos), active[-1] + pad + 2)]


def make_transition(a, b, n):
    """Transicion smoothstep entre clips (aproxima coarticulacion; el blank del CTC la absorbe)."""
    out = []
    for i in range(1, n + 1):
        t = i / (n + 1)
        s = 3 * t ** 2 - 2 * t ** 3
        out.append((1 - s) * a + s * b)
    return np.array(out, dtype=np.float32) if out else np.empty((0,) + a.shape, np.float32)


def concat_sentence(paths, ccfg, rng):
    segs = []
    for p in paths:
        pos = np.load(p).astype(np.float32)
        if len(pos) == 1:  # imagen estatica -> mini-secuencia sostenida
            pos = np.repeat(pos, 8, axis=0)
        pos = trim_rest(pos, ccfg["trim_rest_ratio"])
        f = rng.uniform(*ccfg["time_warp"])
        idx = np.clip((np.arange(max(1, int(len(pos) * f))) / f).astype(int), 0, len(pos) - 1)
        segs.append(pos[idx])
    full = [segs[0]]
    for prev, nxt in zip(segs, segs[1:]):
        n = rng.randint(ccfg["transition_frames_min"], ccfg["transition_frames_max"])
        full.append(make_transition(prev[-1], nxt[0], n))
        full.append(nxt)
    return np.concatenate(full, axis=0)


def add_motion(pos):
    vel = np.zeros_like(pos)
    vel[1:] = np.diff(pos, axis=0)
    acc = np.zeros_like(vel)
    acc[1:] = np.diff(vel, axis=0)
    return np.concatenate([pos, vel, acc], axis=1)


def augment(feats, jitter, drop_p, rng):
    feats = feats + np.random.default_rng(rng.randint(0, 2 ** 31)).normal(
        0, jitter, feats.shape).astype(np.float32)
    keep = np.random.default_rng(rng.randint(0, 2 ** 31)).random(len(feats)) > drop_p
    return feats[keep] if keep.sum() >= 2 else feats


def generate_split(clip_index, vocab, n, out_dir, split, cfg, seed):
    rng = random.Random(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    available = set(clip_index.keys())
    rows, made, tries = [], 0, 0
    while made < n and tries < n * 30:
        tries += 1
        branch, items = plan_sentence(vocab.by_class, available, rng)
        if len(items) < 2:
            continue
        glosses = [g for _, g in items]
        paths, signers = pick_clips(glosses, clip_index, rng)
        try:
            feats = augment(add_motion(concat_sentence(paths, cfg["concat"], rng)),
                            cfg["augment"]["jitter"], cfg["augment"]["frame_drop_p"], rng)
        except Exception:
            continue
        english = build_english(branch, items, vocab.english, rng)
        sid = f"syn_{split}_{made:06d}"
        np.save(out / f"{sid}.npy", feats.astype(np.float32))
        rows.append({"synthetic_id": sid, "gloss_sequence": " ".join(glosses),
                     "english": english, "branch": branch,
                     "source_clips": "|".join(paths),
                     "signers": "|".join(map(str, signers)), "split": split})
        made += 1

    with open(out / f"synthetic_manifest_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["synthetic_id", "gloss_sequence", "english",
                                          "branch", "source_clips", "signers", "split"])
        w.writeheader()
        w.writerows(rows)
    with open(out / f"parallel_corpus_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gloss_sequence", "english"])
        w.writeheader()
        for r in rows:
            w.writerow({"gloss_sequence": r["gloss_sequence"], "english": r["english"]})

    print(f"[{split}] {made} frases sinteticas -> {out}")
    for r in rows[:5]:
        print(f"    [{r['gloss_sequence']}]  ->  {r['english']}")
    if made < n:
        print(f"    AVISO [{split}]: solo se lograron {made}/{n} "
              f"(vocabulario/clips disponibles limitan las plantillas).")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/synthetic.yaml")
    args = ap.parse_args()
    run_generator(args.config)


def run_generator(config_path):
    cfg = load_config(config_path)
    data_cfg = cfg["data"]
    vocab = Vocab(data_cfg["paths"]["vocab_csv"],
                  special_classes=data_cfg["folders"]["special_classes"])
    clips = cfg["clips_manifest"]
    if not Path(clips).is_file():
        sys.exit(f"No existe {clips}. Corre antes: docker compose run --rm preprocess")

    idx_train = load_clip_index(clips, "train")
    idx_test = load_clip_index(clips, "test")
    if not idx_train:
        sys.exit(f"Sin clips de train en {clips}.")
    seed = cfg["seed"]
    out = cfg["out_dir"]
    all_rows = []
    all_rows += generate_split(idx_train, vocab, cfg["n_train"], out, "train", cfg, seed)
    all_rows += generate_split(idx_train, vocab, cfg["n_val"], out, "val", cfg, seed + 1)
    if idx_test:
        # test SOLO con clips de firmantes de test: mide generalizacion a firmantes no vistos
        all_rows += generate_split(idx_test, vocab, cfg["n_test"], out, "test", cfg, seed + 2)
    else:
        print("AVISO: sin clips de test; no se genero split test sintetico.")
    return all_rows


if __name__ == "__main__":
    main()
