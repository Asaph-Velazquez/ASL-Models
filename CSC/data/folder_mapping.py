"""mapeo de carpetas del dataset a glosas"""
from pathlib import Path


def _files(folder, exts):
    return sorted(p for p in folder.iterdir()
                  if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."))


def _files_recursive(folder, exts):
    """recursivo: soporta subcarpetas de variantes (ej PLACE/V1)"""
    return sorted(p for p in folder.rglob("*")
                  if p.is_file() and p.suffix.lower() in exts and not p.name.startswith("."))


def scan_dataset(data_cfg):
    """escanea el dataset; devuelve (samples, warnings)"""
    root = Path(data_cfg["paths"]["dataset_root"])
    fcfg = data_cfg["folders"]
    video_exts = {e.lower() for e in fcfg["video_exts"]}
    image_exts = {e.lower() for e in fcfg["image_exts"]}
    letter_prefix = fcfg["letter_prefix"]
    number_prefix = fcfg["number_prefix"]
    blank_folder = fcfg["blank_folder"]

    from data.vocab import Vocab
    vocab = Vocab(data_cfg["paths"]["vocab_csv"], special_classes=fcfg["special_classes"])

    samples, warnings = [], []
    isolated = root / fcfg["isolated_dir"]
    letters_root = root / fcfg["letters_dir"]
    numbers_root = root / fcfg["numbers_dir"]

    if not isolated.is_dir():
        warnings.append(f"No existe {isolated}; no hay datos aislados que mapear.")
        return samples, warnings

    skip_names = {Path(fcfg["letters_dir"]).name, Path(fcfg["numbers_dir"]).name}

    # paso 1: videos aislados, carpeta = glosa
    for folder in sorted(p for p in isolated.iterdir() if p.is_dir()):
        if folder.name in skip_names:
            continue
        gloss = folder.name
        vids = _files_recursive(folder, video_exts)
        if gloss not in vocab and gloss not in vocab.special_classes:
            warnings.append(f"Carpeta '{folder}' no mapea a ninguna glosa del vocabulario; "
                            f"EXCLUIDA ({len(vids)} videos).")
            continue
        if not vids:
            warnings.append(f"Carpeta '{folder}' sin videos con extensiones validas.")
        bad = [p.name for p in folder.rglob("*")
               if p.is_file() and not p.name.startswith(".")
               and p.suffix.lower() not in video_exts]
        if bad:
            warnings.append(f"'{folder.name}': archivos con extension no valida "
                            f"(¿typo tipo .mp5?): {bad[:5]}")
        for v in vids:
            samples.append({"gloss": gloss, "path": str(v), "kind": "video",
                            "split_hint": None, "source": "isolated"})

    # paso 2: letras, imagenes -> FS-<L>; Blank -> BLANK
    for split_dir, hint in ((fcfg["letters_train"], "train"), (fcfg["letters_test"], "test")):
        base = letters_root / split_dir
        if not base.is_dir():
            warnings.append(f"No existe {base} (letras {hint}).")
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            name = folder.name
            if name == blank_folder:
                gloss = "BLANK"
            else:
                gloss = f"{letter_prefix}{name.upper()}"
            if gloss not in vocab and gloss not in vocab.special_classes:
                warnings.append(f"Carpeta letras '{folder}' -> '{gloss}' no esta en el "
                                f"vocabulario; EXCLUIDA.")
                continue
            for img in _files(folder, image_exts):
                samples.append({"gloss": gloss, "path": str(img), "kind": "image",
                                "split_hint": hint, "source": "letters"})

    # paso 3: numeros, imagenes -> NUM-<d>
    for split_dir, hint in ((fcfg["numbers_train"], "train"), (fcfg["numbers_test"], "test")):
        base = numbers_root / split_dir
        if not base.is_dir():
            warnings.append(f"No existe {base} (numeros {hint}).")
            continue
        for folder in sorted(p for p in base.iterdir() if p.is_dir()):
            if folder.name == blank_folder:
                gloss = "BLANK"  # negativo "no hay mano"
            else:
                gloss = f"{number_prefix}{folder.name}"
            if gloss not in vocab and gloss not in vocab.special_classes:
                warnings.append(f"Carpeta numeros '{folder}' -> '{gloss}' no esta en el "
                                f"vocabulario; EXCLUIDA.")
                continue
            for img in _files(folder, image_exts):
                samples.append({"gloss": gloss, "path": str(img), "kind": "image",
                                "split_hint": hint, "source": "numbers"})

    return samples, warnings


def scan_phrases(data_cfg, vocab):
    """frases continuas; devuelve (phrases, invalid, warnings)"""
    import csv as _csv
    root = Path(data_cfg["paths"]["dataset_root"])
    pcfg = data_cfg["phrases"]
    pdir = root / pcfg["dir"]
    video_exts = {e.lower() for e in data_cfg["folders"]["video_exts"]}
    phrases, invalid, warnings = [], [], []

    def check_sequence(seq, origin):
        tokens = seq.split()
        if not tokens:
            return "secuencia vacia"
        for t in tokens:
            if t != t.upper():
                return f"token '{t}' no esta en MAYUSCULAS ({origin})"
            if t not in vocab:
                return (f"token '{t}' no existe en el vocabulario ({origin}); nombres en "
                        f"lenguaje natural o espanol NO son validos")
        return None

    if not pdir.is_dir():
        warnings.append(f"No existe {pdir}; sin frases reales para la Fase 3.")
        return phrases, invalid, warnings

    if pcfg["phrase_label_source"] == "annotations":
        ann = root / pcfg["annotations_csv"]
        if not ann.is_file():
            warnings.append(f"phrase_label_source=annotations pero no existe {ann}; "
                            f"cayendo a modo filename.")
        else:
            with open(ann, newline="", encoding="utf-8-sig") as f:
                for r in _csv.DictReader(f):
                    vp = root / r["video_path"] if not Path(r["video_path"]).is_absolute() \
                        else Path(r["video_path"])
                    if not vp.is_file():
                        vp2 = pdir / r["video_path"]
                        vp = vp2 if vp2.is_file() else vp
                    seq = r["gloss_sequence"].strip()
                    err = check_sequence(seq, "annotations.csv")
                    if not vp.is_file():
                        invalid.append({"path": str(vp), "reason": "video no existe"})
                    elif err:
                        invalid.append({"path": str(vp), "reason": err})
                    else:
                        phrases.append({"path": str(vp), "gloss_sequence": seq,
                                        "signer_id": (r.get("signer_id") or "").strip() or None})
            return phrases, invalid, warnings

    # modo filename: el nombre del archivo ES la secuencia de glosas
    for v in _files(pdir, video_exts):
        seq = v.stem.strip()
        err = check_sequence(seq, "nombre de archivo")
        if err:
            invalid.append({"path": str(v), "reason": err})
        else:
            phrases.append({"path": str(v), "gloss_sequence": seq, "signer_id": None})
    return phrases, invalid, warnings


def scan_how2sign(data_cfg, vocab):
    """frases how2sign del manifiesto mapeado; splits oficiales, no se recalculan"""
    import csv as _csv
    h2s_cfg = data_cfg.get("how2sign") or {}
    items, warnings = [], []
    if not h2s_cfg.get("enabled"):
        return items, warnings
    man = Path(h2s_cfg["manifest"])
    if not man.is_file():
        warnings.append(f"How2Sign habilitado pero no existe {man}; corre "
                        f"`docker compose run --rm how2sign` para descargar y mapear.")
        return items, warnings
    n_missing, n_bad = 0, 0
    with open(man, newline="", encoding="utf-8-sig") as f:
        for r in _csv.DictReader(f):
            vp = Path(r["video_path"].strip())
            seq = r["gloss_sequence"].strip()
            if not vp.is_file():
                n_missing += 1
                continue
            if any(t not in vocab for t in seq.split()):
                n_bad += 1  # el vocabulario pudo cambiar despues del filtrado
                continue
            items.append({"path": str(vp), "gloss_sequence": seq,
                          "signer_id": r["signer_id"].strip(),
                          "split": r["split"].strip()})
    if n_missing:
        warnings.append(f"How2Sign: {n_missing} clips del manifiesto no estan en disco.")
    if n_bad:
        warnings.append(f"How2Sign: {n_bad} secuencias con tokens fuera del vocabulario "
                        f"actual (re-corre el servicio how2sign para refiltrar).")
    return items, warnings
