"""puente how2sign -> pseudo-glosas: mapea su traduccion inglesa al vocabulario (supervision debil)"""
import csv
import re
from pathlib import Path

# stopwords del ingles; las que esten en el vocabulario se re-activan como contenido
STOPWORDS = {
    "a", "an", "the", "to", "of", "at", "by", "for", "from", "in", "into", "on",
    "onto", "up", "down", "out", "over", "under", "about", "as", "than", "then",
    "and", "or", "but", "if", "so", "too", "also", "just", "really", "very",
    "is", "are", "am", "was", "were", "be", "been", "being",
    "do", "does", "did", "will", "would", "could", "shall", "should", "may",
    "might", "must", "it", "its", "this", "that", "these", "those", "there",
    "here", "he", "she", "they", "them", "him", "his", "her", "their", "we",
    "us", "our", "gonna", "going", "get", "got", "well", "okay", "ok", "oh",
    "with", "have", "has", "had", "can", "i", "you", "me", "my", "your", "what",
    "where",
}

NUMBER_WORDS = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
                "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
                "ten": "10"}

# flexiones irregulares -> forma base
IRREGULAR = {"has": "have", "had": "have", "having": "have",
             "until": "till", "favourite": "favorite", "idea": "ideas",
             "studies": "study", "studied": "study", "studying": "study"}


def build_gloss_mapper(vocab):
    """construye (word2gloss, stopwords efectivas) desde el vocabulario"""
    w2g = {}
    for gloss in vocab.glosses:
        eng = vocab.english[gloss].strip().lower()
        if eng.startswith("letter "):
            continue
        if eng.isdigit():
            w2g[eng] = gloss
            for word, digit in NUMBER_WORDS.items():
                if digit == eng:
                    w2g[word] = gloss
            continue
        w2g[eng] = gloss
        g_low = gloss.lower()
        if re.fullmatch(r"[a-z]+", g_low):
            w2g.setdefault(g_low, gloss)
    stop = STOPWORDS - set(w2g)
    return w2g, stop


def _inflection_bases(token):
    """candidatos de forma base: plural, -ing, -ed"""
    bases = []
    if token.endswith("ies") and len(token) > 4:
        bases.append(token[:-3] + "y")
    if token.endswith("es") and len(token) > 3:
        bases.append(token[:-2])
    if token.endswith("s") and len(token) > 2:
        bases.append(token[:-1])
    if token.endswith("ing") and len(token) > 4:
        bases += [token[:-3], token[:-3] + "e"]
    if token.endswith("ed") and len(token) > 3:
        bases += [token[:-2], token[:-2] + "e", token[:-1]]
    return bases


def _lookup(token, w2g):
    if token in w2g:
        return w2g[token]
    base = IRREGULAR.get(token)
    if base and base in w2g:
        return w2g[base]
    for b in _inflection_bases(token):
        if b in w2g:
            return w2g[b]
    return None


def tokenize_english(sentence):
    """minusculas, sin contracciones ni puntuacion"""
    s = sentence.lower()
    s = re.sub(r"n't\b", " not", s)
    s = re.sub(r"'(re|ve|ll|d|m|s)\b", " ", s)
    return re.findall(r"[a-z]+|\d+", s)


def map_sentence(sentence, w2g, stopwords, min_coverage=0.8,
                 min_mapped_tokens=2, max_tokens=12):
    """oracion inglesa -> (pseudo-glosas | None si no supera los umbrales, info)"""
    tokens = tokenize_english(sentence)
    content = [t for t in tokens if t not in stopwords]
    mapped = [g for g in (_lookup(t, w2g) for t in content) if g is not None]
    coverage = len(mapped) / len(content) if content else 0.0
    info = {"content": len(content), "mapped": len(mapped),
            "coverage": round(coverage, 3)}
    if (not content or coverage < min_coverage
            or len(mapped) < min_mapped_tokens or len(content) > max_tokens):
        return None, info
    return " ".join(mapped), info


# lectura de los csv oficiales de how2sign

_COLMAP = {"video_id": "VIDEO_ID", "video_name": "VIDEO_NAME",
           "sentence_id": "SENTENCE_ID", "sentence_name": "SENTENCE_NAME",
           "sentence": "SENTENCE"}


def read_how2sign_csv(path):
    """lee el csv oficial (tab o coma); claves minusculas"""
    text = Path(path).read_text(encoding="utf-8-sig", errors="replace")
    first = text.splitlines()[0] if text else ""
    delim = "\t" if "\t" in first else ","
    rows = []
    reader = csv.DictReader(text.splitlines(), delimiter=delim)
    fields = {(f or "").strip().upper() for f in (reader.fieldnames or [])}
    missing = [c for c in ("SENTENCE_NAME", "SENTENCE") if c not in fields]
    if missing:
        raise ValueError(f"{path}: faltan columnas {missing}; "
                         f"encontradas: {sorted(fields)}")
    for r in reader:
        norm = {(k or "").strip().upper(): (v or "").strip() for k, v in r.items()}
        rows.append({low: norm.get(up, "") for low, up in _COLMAP.items()})
    return rows


def filter_split(csv_path, vocab, fcfg):
    """filtra un csv de how2sign contra el vocabulario; devuelve (accepted, stats)"""
    w2g, stop = build_gloss_mapper(vocab)
    rows = read_how2sign_csv(csv_path)
    accepted, n_seen = [], 0
    for r in rows:
        if not r["sentence_name"]:
            continue
        n_seen += 1
        seq, _ = map_sentence(r["sentence"], w2g, stop,
                              fcfg["min_coverage"], fcfg["min_mapped_tokens"],
                              fcfg["max_tokens"])
        if seq:
            accepted.append({"sentence_name": r["sentence_name"],
                             "video_id": r["video_id"] or r["sentence_name"],
                             "gloss_sequence": seq, "sentence": r["sentence"]})
    cap = int(fcfg.get("max_clips_per_split", 0) or 0)
    if cap and len(accepted) > cap:
        accepted = accepted[:cap]
    return accepted, {"total": n_seen, "accepted": len(accepted)}
