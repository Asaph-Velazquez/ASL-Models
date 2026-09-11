"""mapea los clips how2sign descargados a manifiesto propio + anexo a signer_manifest.csv (idempotente)"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.vocab import Vocab
from utils.config import load_config


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def collect(cfg, vocab):
    root = Path(cfg["root"])
    rows, n_no_clip, n_bad_vocab = [], 0, 0
    for split in cfg["splits"]:
        fcsv = root / f"filtered_{split}.csv"
        if not fcsv.is_file():
            print(f"AVISO [map] no existe {fcsv}; corre antes download_how2sign.py")
            continue
        for r in read_csv(fcsv):
            clip = root / "video_clips" / split / f"{r['sentence_name']}.mp4"
            if not clip.is_file():
                n_no_clip += 1
                continue
            bad = [t for t in r["gloss_sequence"].split() if t not in vocab]
            if bad:  # el vocabulario pudo cambiar tras el filtrado
                n_bad_vocab += 1
                continue
            rows.append({"video_path": str(clip), "gloss_sequence": r["gloss_sequence"],
                         "signer_id": f"h2s_{r['video_id']}", "split": split,
                         "sentence": r["sentence"]})
    return rows, n_no_clip, n_bad_vocab


def append_signer_manifest(manifest_path, rows):
    """anexa filas sin duplicar rutas ya presentes"""
    path = Path(manifest_path)
    existing = set()
    if path.is_file():
        existing = {r["video_path"].strip() for r in read_csv(path)}
    new = [r for r in rows if r["video_path"] not in existing]
    if not new:
        return 0
    if path.is_file() and path.stat().st_size and not path.read_bytes().endswith(b"\n"):
        with open(path, "a", encoding="utf-8") as f:  # evita pegar la 1a fila a la ultima linea
            f.write("\n")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for r in new:
            w.writerow([r["gloss_sequence"], r["video_path"], r["signer_id"]])
    return len(new)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/how2sign.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    root = Path(cfg["root"])
    vocab = Vocab(data_cfg["paths"]["vocab_csv"],
                  special_classes=data_cfg["folders"]["special_classes"])

    rows, n_no_clip, n_bad_vocab = collect(cfg, vocab)
    if n_no_clip:
        print(f"AVISO [map] {n_no_clip} oraciones filtradas sin clip en disco "
              f"(descarga incompleta); se omiten.")
    if n_bad_vocab:
        print(f"AVISO [map] {n_bad_vocab} secuencias con tokens ya fuera del "
              f"vocabulario actual; se omiten (re-corre download para refiltrar).")
    if not rows:
        raise SystemExit("[map] 0 clips mapeables; nada que escribir.")

    man = root / "how2sign_manifest.csv"
    with open(man, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["video_path", "gloss_sequence",
                                          "signer_id", "split", "sentence"])
        w.writeheader()
        w.writerows(rows)
    by_split = {}
    for r in rows:
        by_split[r["split"]] = by_split.get(r["split"], 0) + 1
    print(f"[map] {man}: {len(rows)} clips ({by_split})")

    n_new = append_signer_manifest(cfg["signer_manifest"], rows)
    print(f"[map] signer_manifest.csv: {n_new} filas nuevas anexadas "
          f"({len(rows) - n_new} ya estaban).")

    corpus = root / "parallel_corpus_how2sign.csv"
    with open(corpus, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gloss_sequence", "english"])
        for r in rows:
            w.writerow([r["gloss_sequence"], r["sentence"]])
    print(f"[map] {corpus}: corpus paralelo real opcional para la Fase 4.")
    print("[map] listo. Siguiente paso: docker compose run --rm preprocess")


if __name__ == "__main__":
    main()
