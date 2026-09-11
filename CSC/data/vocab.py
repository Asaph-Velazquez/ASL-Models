"""vocabulario dirigido por asl_hotel_vocabulary.csv; N siempre = len(csv)"""
import csv
import re


class Vocab:
    def __init__(self, vocab_csv, special_classes=None):
        self.rows = []
        with open(vocab_csv, newline="", encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                g = r["canonical_gloss"].strip()
                if not g:
                    continue
                self.rows.append({
                    "gloss": g,
                    "english": (r.get("english") or g).strip(),
                    "functional_class": (r.get("functional_class") or "").strip(),
                    "notes": (r.get("notes") or "").strip(),
                })
        if not self.rows:
            raise ValueError(f"Vocabulario vacio: {vocab_csv}")

        self.glosses = [r["gloss"] for r in self.rows]
        self.N = len(self.glosses)
        self.english = {r["gloss"]: r["english"] for r in self.rows}
        self.functional_class = {r["gloss"]: r["functional_class"] for r in self.rows}
        self.by_class = {}
        for r in self.rows:
            self.by_class.setdefault(r["functional_class"], []).append(r["gloss"])

        # cabeza 4A: vocabulario + clases especiales
        self.special_classes = list(special_classes or [])
        self.class_names = self.glosses + [s for s in self.special_classes
                                           if s not in self.glosses]
        self.num_head_classes = len(self.class_names)
        self.class2id = {g: i for i, g in enumerate(self.class_names)}
        self.id2class = {i: g for g, i in self.class2id.items()}

        # capa 4B ctc: blank=0, glosas 1..N
        self.ctc_blank = 0
        self.gloss2ctc = {g: i + 1 for i, g in enumerate(self.glosses)}
        self.ctc2gloss = {i + 1: g for i, g in enumerate(self.glosses)}
        self.num_ctc_classes = self.N + 1

    def __len__(self):
        return self.N

    def __contains__(self, gloss):
        return gloss in self.english

    def encode_ctc(self, gloss_sequence):
        """'ROOM WANT TOWEL' -> [ids]; KeyError si un token no esta en el vocabulario"""
        return [self.gloss2ctc[t] for t in gloss_sequence.split()]

    def decode_ctc(self, ids):
        return " ".join(self.ctc2gloss[i] for i in ids if i != self.ctc_blank)

    def confusable_pairs(self):
        """pares confundibles declarados en la columna notes"""
        pairs = set()
        for r in self.rows:
            m = re.search(r"confusi?on con (.+)", r["notes"], re.IGNORECASE)
            if not m:
                continue
            for tok in re.split(r"[/,]", m.group(1)):
                tok = tok.strip().upper().rstrip(".")
                if not tok:
                    continue
                cand = None
                if tok in self.english:
                    cand = tok
                elif len(tok) == 1 and tok.isalpha() and f"FS-{tok}" in self.english:
                    cand = f"FS-{tok}"
                elif tok.isdigit() and f"NUM-{tok}" in self.english:
                    cand = f"NUM-{tok}"
                if cand and cand != r["gloss"]:
                    pairs.add(tuple(sorted((r["gloss"], cand))))
        return sorted(pairs)

    def validate(self):
        """chequeos estructurales del CSV; lista de problemas (vacia = ok)"""
        problems = []
        seen = set()
        for r in self.rows:
            g = r["gloss"]
            if " " in g:
                problems.append(f"Glosa con espacios internos: '{g}' (usa guion, ej. REMOTE-CONTROL)")
            if g != g.upper():
                problems.append(f"Glosa no esta en MAYUSCULAS: '{g}'")
            if g in seen:
                problems.append(f"Glosa duplicada: '{g}'")
            seen.add(g)
            if not r["functional_class"]:
                problems.append(f"Glosa sin functional_class: '{g}'")
        return problems
