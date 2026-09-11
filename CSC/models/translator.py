"""capa 6: traduccion glosa->ingles con T5 y fallback a plantillas, separada del CTC"""
from pathlib import Path

import torch


class SimpleTokenizer:
    """tokenizador word-level minimo, solo smoke test"""

    PAD, EOS, UNK = 0, 1, 2

    def __init__(self, texts):
        words = sorted({w for t in texts for w in t.lower().split()})
        self.vocab = {"<pad>": self.PAD, "</s>": self.EOS, "<unk>": self.UNK}
        for w in words:
            self.vocab.setdefault(w, len(self.vocab))
        self.inv = {i: w for w, i in self.vocab.items()}
        self.pad_token_id = self.PAD
        self.eos_token_id = self.EOS

    def __len__(self):
        return len(self.vocab)

    def encode(self, text, max_len=64):
        ids = [self.vocab.get(w, self.UNK) for w in text.lower().split()][: max_len - 1]
        return ids + [self.EOS]

    def batch(self, texts, max_len=64):
        seqs = [self.encode(t, max_len) for t in texts]
        T = max(len(s) for s in seqs)
        ids = torch.full((len(seqs), T), self.PAD, dtype=torch.long)
        attn = torch.zeros(len(seqs), T, dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s)
            attn[i, :len(s)] = 1
        return ids, attn

    def decode(self, ids):
        words = [self.inv.get(int(i), "") for i in ids
                 if int(i) not in (self.PAD, self.EOS)]
        return " ".join(w for w in words if w)


class Translator:
    def __init__(self, model, tokenizer, device="cpu", source_prefix="", is_hf=True):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.source_prefix = source_prefix
        self.is_hf = is_hf

    # construccion
    @classmethod
    def from_pretrained(cls, name_or_dir, source_prefix="", device="cpu"):
        from transformers import AutoTokenizer, T5ForConditionalGeneration
        tok = AutoTokenizer.from_pretrained(name_or_dir)
        model = T5ForConditionalGeneration.from_pretrained(name_or_dir)
        return cls(model, tok, device, source_prefix, is_hf=True)

    @classmethod
    def tiny(cls, corpus_texts, device="cpu", source_prefix=""):
        """T5 diminuto aleatorio, solo smoke test sin descargas"""
        from transformers import T5Config, T5ForConditionalGeneration
        tok = SimpleTokenizer(corpus_texts)
        cfg = T5Config(vocab_size=len(tok), d_model=32, d_kv=8, d_ff=64, num_layers=2,
                       num_heads=2, decoder_start_token_id=tok.pad_token_id,
                       pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)
        return cls(T5ForConditionalGeneration(cfg), tok, device, source_prefix, is_hf=False)

    def save(self, out_dir):
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        if self.is_hf:
            self.model.save_pretrained(out)
            self.tokenizer.save_pretrained(out)

    # tensores
    def make_batch(self, sources, targets=None, max_src=64, max_tgt=64):
        sources = [self.source_prefix + s for s in sources]
        if self.is_hf:
            enc = self.tokenizer(sources, padding=True, truncation=True,
                                 max_length=max_src, return_tensors="pt")
            batch = {"input_ids": enc.input_ids.to(self.device),
                     "attention_mask": enc.attention_mask.to(self.device)}
            if targets is not None:
                lab = self.tokenizer(text_target=targets, padding=True, truncation=True,
                                     max_length=max_tgt, return_tensors="pt").input_ids
                lab[lab == self.tokenizer.pad_token_id] = -100
                batch["labels"] = lab.to(self.device)
        else:
            ids, attn = self.tokenizer.batch(sources, max_src)
            batch = {"input_ids": ids.to(self.device),
                     "attention_mask": attn.to(self.device)}
            if targets is not None:
                lab, _ = self.tokenizer.batch(targets, max_tgt)
                lab = lab.clone()
                lab[lab == self.tokenizer.pad_token_id] = -100
                batch["labels"] = lab.to(self.device)
        return batch

    def train_step(self, sources, targets, optimizer, max_src=64, max_tgt=64):
        self.model.train()
        batch = self.make_batch(sources, targets, max_src, max_tgt)
        out = self.model(**batch)
        optimizer.zero_grad()
        out.loss.backward()
        optimizer.step()
        return float(out.loss.item())

    @torch.no_grad()
    def eval_loss(self, sources, targets, max_src=64, max_tgt=64):
        self.model.eval()
        out = self.model(**self.make_batch(sources, targets, max_src, max_tgt))
        return float(out.loss.item())

    @torch.no_grad()
    def translate(self, gloss_sequences, max_len=64):
        self.model.eval()
        batch = self.make_batch(list(gloss_sequences))
        gen = self.model.generate(input_ids=batch["input_ids"],
                                  attention_mask=batch["attention_mask"],
                                  max_length=max_len, num_beams=2)
        if self.is_hf:
            return [self.tokenizer.decode(g, skip_special_tokens=True) for g in gen]
        return [self.tokenizer.decode(g) for g in gen]


def translate_glosses(gloss_sequence, vocab, translator=None):
    """capa 6: T5 si hay, plantillas si no; devuelve (ingles, fuente)"""
    from synthetic.english_templates import template_translate
    if translator is not None:
        try:
            out = translator.translate([gloss_sequence])[0].strip()
            if out:
                return out, "t5"
        except Exception as e:
            print(f"AVISO: T5 fallo ({e}); usando plantillas.")
    return template_translate(gloss_sequence, vocab), "plantillas"
