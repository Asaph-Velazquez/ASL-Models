"""descarga how2sign: csv -> filtro por vocabulario -> zip -> extraccion selectiva (retoma si se corta)"""
import argparse
import csv
import shutil
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.how2sign import filter_split
from data.vocab import Vocab
from utils.config import load_config


def gdrive_download(file_id, out_path):
    import gdown
    out_path.parent.mkdir(parents=True, exist_ok=True)
    got = gdown.download(id=file_id, output=str(out_path), quiet=False)
    if not got or not out_path.is_file():
        raise IOError(f"gdown no pudo bajar id={file_id} -> {out_path} "
                      f"(¿cuota de Drive? bajalo a mano y re-corre).")
    return out_path


def write_filtered(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sentence_name", "video_id",
                                          "gloss_sequence", "sentence"])
        w.writeheader()
        w.writerows(rows)


def extract_clips(zip_path, accepted, out_dir):
    """extrae del zip solo los clips aceptados, por nombre base"""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_ok, n_missing = 0, 0
    with zipfile.ZipFile(zip_path) as z:
        by_base = {Path(n).name: n for n in z.namelist()
                   if n.lower().endswith(".mp4")}
        for row in accepted:
            base = f"{row['sentence_name']}.mp4"
            member = by_base.get(base)
            dest = out_dir / base
            if dest.is_file() and dest.stat().st_size > 0:
                n_ok += 1
                continue
            if not member:
                n_missing += 1
                continue
            with z.open(member) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst, 1024 * 1024)
            n_ok += 1
    return n_ok, n_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/how2sign.yaml")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="por defecto los de la config (train val test)")
    ap.add_argument("--csv-only", action="store_true",
                    help="solo CSVs + filtrado; sin bajar los zips de video")
    ap.add_argument("--keep-zip", action="store_true",
                    help="no borrar el zip tras extraer (para reintentos)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_cfg = cfg["data"]
    root = Path(cfg["root"])
    splits = args.splits or cfg["splits"]
    keep_zip = args.keep_zip or cfg.get("keep_zip", False)
    vocab = Vocab(data_cfg["paths"]["vocab_csv"],
                  special_classes=data_cfg["folders"]["special_classes"])
    print(f"[how2sign] vocabulario: {vocab.N} glosas; root: {root}; splits: {splits}")
    print("[how2sign] licencia CC BY-NC 4.0 (solo uso no comercial, citar el paper).")

    for split in splits:
        print(f"\n===== split {split} =====")
        csv_path = root / "csv" / f"how2sign_realigned_{split}.csv"
        if not csv_path.is_file():
            gdrive_download(cfg["drive"]["csv"][split], csv_path)
        else:
            print(f"[how2sign] {csv_path} ya existe; no se vuelve a bajar.")

        accepted, stats = filter_split(csv_path, vocab, cfg["filter"])
        write_filtered(root / f"filtered_{split}.csv", accepted)
        print(f"[how2sign] {split}: {stats['accepted']}/{stats['total']} oraciones "
              f"aceptadas por el filtro de vocabulario -> filtered_{split}.csv")
        if not accepted:
            print(f"AVISO [how2sign] {split}: 0 oraciones aceptadas; baja "
                  f"filter.min_coverage en configs/how2sign.yaml si quieres mas "
                  f"(mas volumen = pseudo-glosas mas ruidosas).")
            continue
        if args.csv_only:
            continue

        zip_path = root / "zips" / f"{split}_rgb_front_clips.zip"
        out_dir = root / "video_clips" / split
        pending = [r for r in accepted
                   if not (out_dir / f"{r['sentence_name']}.mp4").is_file()]
        if not pending:
            print(f"[how2sign] {split}: los {len(accepted)} clips ya estan extraidos.")
            continue
        if not zip_path.is_file():
            print(f"[how2sign] bajando clips {split} (GRANDE; train ~31G)...")
            gdrive_download(cfg["drive"]["clips_front"][split], zip_path)
        n_ok, n_missing = extract_clips(zip_path, accepted, out_dir)
        print(f"[how2sign] {split}: {n_ok} clips extraidos en {out_dir} "
              f"({n_missing} no estaban en el zip).")
        if not keep_zip:
            zip_path.unlink()
            print(f"[how2sign] {zip_path} borrado (usa --keep-zip para conservarlo).")

    print("\n[how2sign] listo. Siguiente paso: "
          "python data/map_how2sign_manifest.py --config configs/how2sign.yaml")


if __name__ == "__main__":
    main()
