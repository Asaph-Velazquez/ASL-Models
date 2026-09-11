#!/usr/bin/env python3
"""Generador sintetico standalone (CLI --dummy/--clips): frases CTC + corpus paralelo ingles."""
# NOTA: el orden ASL de estas plantillas es aproximado; revisar con una persona senante.
import argparse, csv, random, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

# slot = (nombre, {"class":[...]}|{"gloss":[...]}, prob)
TEMPLATES = {
    "room_service": [
        [("when",    {"gloss": ["NOW", "LATER", "TODAY"]},           0.30),
         ("subject", {"gloss": ["I"]},                              0.50),
         ("verb",    {"gloss": ["WANT", "NEED", "ORDER", "BRING"]}, 1.00),
         ("qty",     {"gloss": ["MORE", "EXTRA", "ANOTHER"]},       0.25),
         ("object",  {"class": ["object", "food"]},                1.00),
         ("place",   {"gloss": ["ROOM", "MY", "BATHROOM"]},         0.30),
         ("polite",  {"gloss": ["PLEASE"]},                        0.50)],
    ],
    "problem": [
        [("poss",    {"gloss": ["MY"]},                            0.55),
         ("topic",   {"class": ["place", "object"]},               1.00),
         ("state",   {"class": ["adjective", "problem"]},          1.00),
         ("neg",     {"gloss": ["NOT"]},                           0.20),
         ("verb",    {"gloss": ["FIX", "CHECK"]},                  0.40),
         ("polite",  {"gloss": ["PLEASE"]},                        0.40)],
    ],
    "servicios": [
        [("topic",   {"class": ["place", "service"]},              1.00),
         ("wh",      {"gloss": ["WHERE", "WHEN", "HOW-MUCH"]},      0.60),
         ("verb",    {"gloss": ["INCLUDE", "OPEN", "CLOSE"]},       0.30)],
    ],
    "movilidad": [
        [("when",    {"gloss": ["NOW", "LATER"]},                  0.25),
         ("subject", {"gloss": ["I"]},                             0.40),
         ("verb",    {"gloss": ["WANT", "NEED", "CALL"]},          0.60),
         ("service", {"gloss": ["TAXI", "CAR", "VALET", "AIRPORT"]},1.00),
         ("polite",  {"gloss": ["PLEASE"]},                        0.40)],
    ],
}
ENGLISH_TEMPLATES = {
    "room_service": [
        "I would like {qty} {object} {when} .",
        "Could you bring {qty} {object} to my {place} {when} , please ?",
        "I need {qty} {object} , please .",
    ],
    "problem": [
        "The {topic} in my room is {state} .",
        "There is a problem with the {topic} .",
        "My {topic} is {state} , please {verb} it .",
    ],
    "servicios": [
        "Where is the {topic} ?",
        "Is the {topic} {verb} ?",
        "What time does the {topic} {verb} ?",
    ],
    "movilidad": [
        "I need a {service} {when} .",
        "Please call a {service} .",
        "Can I get a {service} {when} ?",
    ],
}


def load_vocab(path):
    by_class, gloss_en = {}, {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            g = r["canonical_gloss"]
            by_class.setdefault(r["functional_class"], []).append(g)
            gloss_en[g] = r.get("english", g)
    return by_class, gloss_en


def load_clip_index(clips_csv, split):
    index = {}
    with open(clips_csv, newline="") as f:
        for r in csv.DictReader(f):
            if r.get("split", "train") != split:
                continue
            index.setdefault(r["gloss"], []).append((r["npy_path"], r["signer_id"]))
    return index


def plan_sentence(vocab_by_class, available):
    branch = random.choice(list(TEMPLATES.keys()))
    template = random.choice(TEMPLATES[branch])
    items = []
    for slot, spec, prob in template:
        if random.random() > prob:
            continue
        choices = spec["gloss"] if "gloss" in spec else \
            [g for c in spec["class"] for g in vocab_by_class.get(c, [])]
        choices = [g for g in choices if g in available]
        if not choices:
            continue
        g = random.choice(choices)
        if items and items[-1][1] == g:
            continue
        items.append((slot, g))
    return branch, items


def build_english(branch, items, gloss_en):
    en = defaultdict(str, {slot: gloss_en.get(g, g).lower() for slot, g in items})
    s = random.choice(ENGLISH_TEMPLATES[branch]).format_map(en)
    s = re.sub(r"\s+", " ", s).strip()
    for a, b in [(" ,", ","), (" .", "."), (" ?", "?"), (" !", "!")]:
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\ba\s+([aeiou])", r"an \1", s)
    if s:
        s = s[0].upper() + s[1:]
        if s[-1] not in ".?!":
            s += "."
    return s


def pick_clips(glosses, clip_index):
    common = None
    for g in glosses:
        signers = {s for _, s in clip_index.get(g, [])}
        common = signers if common is None else (common & signers)
    if common:
        s = random.choice(list(common))
        return [random.choice([p for p, sg in clip_index[g] if sg == s]) for g in glosses], [s]
    paths, signers = [], []
    for g in glosses:
        p, sg = random.choice(clip_index[g]); paths.append(p); signers.append(sg)
    return paths, list(set(signers))


def trim_rest(pos, thresh_ratio=0.15, pad=2):
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
    out = []
    for i in range(1, n + 1):
        t = i / (n + 1); s = 3 * t**2 - 2 * t**3
        out.append((1 - s) * a + s * b)
    return np.array(out, dtype=np.float32) if out else np.empty((0,) + a.shape, np.float32)


def concat_sentence(paths, n_min=3, n_max=8, warp=(0.85, 1.15)):
    segs = []
    for p in paths:
        pos = np.load(p).astype(np.float32)
        pos = trim_rest(pos)
        f = random.uniform(*warp)
        idx = np.clip((np.arange(max(1, int(len(pos) * f))) / f).astype(int), 0, len(pos) - 1)
        segs.append(pos[idx])
    full = [segs[0]]
    for prev, nxt in zip(segs, segs[1:]):
        full.append(make_transition(prev[-1], nxt[0], random.randint(n_min, n_max)))
        full.append(nxt)
    return np.concatenate(full, axis=0)


def add_motion(pos):
    vel = np.zeros_like(pos); vel[1:] = np.diff(pos, axis=0)
    acc = np.zeros_like(pos); acc[1:] = np.diff(vel, axis=0)
    return np.concatenate([pos, vel, acc], axis=1)


def augment(feats, jitter=0.01, drop_p=0.05):
    feats = feats + np.random.randn(*feats.shape).astype(np.float32) * jitter
    keep = np.random.rand(len(feats)) > drop_p
    return feats[keep] if keep.sum() >= 2 else feats


def generate(clip_index, vocab_by_class, gloss_en, n, out_dir, split, seed=42):
    random.seed(seed); np.random.seed(seed)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    available = set(clip_index.keys())
    rows, made, tries = [], 0, 0
    while made < n and tries < n * 30:
        tries += 1
        branch, items = plan_sentence(vocab_by_class, available)
        if len(items) < 2:
            continue
        glosses = [g for _, g in items]
        paths, signers = pick_clips(glosses, clip_index)
        try:
            feats = augment(add_motion(concat_sentence(paths)))
        except Exception:
            continue
        english = build_english(branch, items, gloss_en)
        sid = f"syn_{split}_{made:06d}"
        np.save(out / f"{sid}.npy", feats)
        rows.append({"synthetic_id": sid, "gloss_sequence": " ".join(glosses),
                     "english": english, "source_clips": "|".join(paths),
                     "signers": "|".join(map(str, signers)), "split": split})
        made += 1

    with open(out / f"synthetic_manifest_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["synthetic_id", "gloss_sequence", "english",
                                          "source_clips", "signers", "split"])
        w.writeheader(); w.writerows(rows)
    with open(out / f"parallel_corpus_{split}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gloss_sequence", "english"])
        w.writeheader()
        for r in rows:
            w.writerow({"gloss_sequence": r["gloss_sequence"], "english": r["english"]})

    print(f"Generadas {made} frases -> {out}/  (+ parallel_corpus para el T5)")
    for r in rows[:8]:
        print(f"  [{r['gloss_sequence']}]  ->  {r['english']}")


def make_dummy(tmp_dir, D=100):
    ILUS = {
        "I": ("I", "pronoun"), "MY": ("my", "pronoun"),
        "WANT": ("want", "verb"), "NEED": ("need", "verb"), "BRING": ("bring", "verb"),
        "ORDER": ("order", "verb"), "FIX": ("fix", "verb"), "CHECK": ("check", "verb"),
        "CALL": ("call", "verb"), "INCLUDE": ("included", "verb"), "OPEN": ("open", "verb"),
        "TOWEL": ("towel", "object"), "BLANKET": ("blanket", "object"),
        "PILLOW": ("pillow", "object"), "AC": ("air conditioning", "object"),
        "TV": ("TV", "object"), "SHOWER": ("shower", "object"), "SOAP": ("soap", "object"),
        "WATER": ("water", "food"), "COFFEE": ("coffee", "food"), "BREAKFAST": ("breakfast", "food"),
        "ROOM": ("room", "place"), "BATHROOM": ("bathroom", "place"), "POOL": ("pool", "place"),
        "GYM": ("gym", "place"), "RESTAURANT": ("restaurant", "place"),
        "TAXI": ("taxi", "service"), "CAR": ("car", "service"), "VALET": ("valet", "service"),
        "AIRPORT": ("airport", "service"),
        "NOW": ("now", "time"), "LATER": ("later", "time"), "TODAY": ("today", "time"),
        "COLD": ("cold", "adjective"), "HOT": ("hot", "adjective"), "BROKEN": ("broken", "adjective"),
        "DIRTY": ("dirty", "adjective"), "NOISE": ("noise", "problem"), "LEAK": ("leak", "problem"),
        "MORE": ("more", "quantity"), "EXTRA": ("extra", "quantity"), "ANOTHER": ("another", "quantity"),
        "WHERE": ("where", "wh_question"), "WHEN": ("when", "wh_question"), "HOW-MUCH": ("how much", "wh_question"),
        "PLEASE": ("please", "courtesy"), "NOT": ("not", "negation"),
    }
    tmp = Path(tmp_dir); tmp.mkdir(parents=True, exist_ok=True)
    by_class, gloss_en, rows = {}, {}, []
    for g, (en, cls) in ILUS.items():
        by_class.setdefault(cls, []).append(g); gloss_en[g] = en
        for k in range(3):
            T = random.randint(15, 40)
            arr = np.cumsum(np.random.randn(T, D).astype(np.float32) * 0.1, axis=0)
            np.save(tmp / f"{g}_{k}.npy", arr)
            rows.append({"gloss": g, "npy_path": str(tmp / f"{g}_{k}.npy"),
                         "signer_id": f"signer_{k % 2}", "split": "train"})
    man = tmp / "clips_manifest.csv"
    with open(man, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gloss", "npy_path", "signer_id", "split"])
        w.writeheader(); w.writerows(rows)
    return str(man), by_class, gloss_en


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips"); ap.add_argument("--vocab", default="asl_hotel_vocabulary.csv")
    ap.add_argument("--split", default="train"); ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--out", default="synthetic_out"); ap.add_argument("--dummy", action="store_true")
    args = ap.parse_args()
    if args.dummy:
        print("[MODO DUMMY] vocabulario hotelero ilustrativo + clips falsos\n")
        clips_csv, by_class, gloss_en = make_dummy("dummy_clips")
    else:
        if not args.clips:
            sys.exit("Falta --clips (o usa --dummy).")
        by_class, gloss_en = load_vocab(args.vocab); clips_csv = args.clips
    idx = load_clip_index(clips_csv, args.split)
    if not idx:
        sys.exit(f"Sin clips para split='{args.split}' en {clips_csv}")
    generate(idx, by_class, gloss_en, args.n, args.out, args.split)


if __name__ == "__main__":
    main()
