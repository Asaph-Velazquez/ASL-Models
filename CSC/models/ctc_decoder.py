"""capa 5: decodificacion CTC (greedy y beam search) + n-grama de glosas del corpus sintetico"""
import math
from collections import defaultdict

import numpy as np


class GlossNGram:
    def __init__(self, order=2, smoothing=0.5):
        self.order = order
        self.smoothing = smoothing
        self.counts = defaultdict(lambda: defaultdict(float))
        self.vocab = set()

    @classmethod
    def from_sequences(cls, sequences, order=2):
        lm = cls(order=order)
        for seq in sequences:
            toks = ["<s>"] * (order - 1) + seq.split() + ["</s>"]
            for i in range(order - 1, len(toks)):
                ctx = tuple(toks[i - order + 1: i])
                lm.counts[ctx][toks[i]] += 1
                lm.vocab.add(toks[i])
        return lm

    def logp(self, context_tokens, token):
        ctx = tuple((["<s>"] * (self.order - 1) + list(context_tokens))[-(self.order - 1):]) \
            if self.order > 1 else tuple()
        c = self.counts.get(ctx, {})
        num = c.get(token, 0.0) + self.smoothing
        den = sum(c.values()) + self.smoothing * max(1, len(self.vocab))
        return math.log(num / den)


def greedy_decode(log_probs, vocab, blank=0):
    """log_probs [T, C] -> glosas (colapso CTC estandar)"""
    ids = log_probs.argmax(axis=-1)
    out, prev = [], blank
    for i in ids:
        if i != blank and i != prev:
            out.append(int(i))
        prev = i
    return vocab.decode_ctc(out)


def beam_search_decode(log_probs, vocab, blank=0, beam_size=8, lm=None,
                       lm_alpha=0.3, lm_beta=0.6):
    """beam search por prefijos con n-grama opcional"""
    log_probs = np.asarray(log_probs, dtype=np.float64)
    T, C = log_probs.shape
    NEG = -1e30

    def lm_score(prefix, tok):
        if lm is None:
            return lm_beta
        g = vocab.ctc2gloss[tok]
        ctx = [vocab.ctc2gloss[t] for t in prefix]
        return lm_alpha * lm.logp(ctx, g) + lm_beta

    # beams: prefijo -> (logp_blank, logp_no_blank)
    beams = {(): (0.0, NEG)}
    for t in range(T):
        new = defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beams.items():
            p_tot = np.logaddexp(pb, pnb)
            # paso 1: blank mantiene el prefijo
            nb = new[prefix]
            nb[0] = np.logaddexp(nb[0], p_tot + log_probs[t, blank])
            # paso 2: repetir el ultimo token
            if prefix:
                last = prefix[-1]
                nb[1] = np.logaddexp(nb[1], pnb + log_probs[t, last])
            # paso 3: extender con un token nuevo
            top = np.argsort(log_probs[t])[::-1][:beam_size + 1]
            for c in top:
                c = int(c)
                if c == blank:
                    continue
                np_prefix = prefix + (c,)
                nn = new[np_prefix]
                if prefix and c == prefix[-1]:
                    base = pb  # repetir requiere blank previo
                else:
                    base = p_tot
                nn[1] = np.logaddexp(nn[1], base + log_probs[t, c] + lm_score(prefix, c))
        beams = dict(sorted(new.items(),
                            key=lambda kv: -np.logaddexp(kv[1][0], kv[1][1]))[:beam_size])
        beams = {k: (v[0], v[1]) for k, v in beams.items()}

    best = max(beams.items(), key=lambda kv: np.logaddexp(kv[1][0], kv[1][1]))[0]
    return vocab.decode_ctc(list(best))
