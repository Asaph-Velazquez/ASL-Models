"""validador de vocabulario, manifiestos, carpetas y frases"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.folder_mapping import scan_dataset, scan_how2sign, scan_phrases
from data.splits import load_signer_manifest
from data.vocab import Vocab
from utils.config import load_config


def validate(data_cfg):
    report = {"vocab": {}, "manifest": {}, "dataset": {}, "phrases": {}, "warnings": []}
    paths = data_cfg["paths"]

    # paso 1: vocabulario
    vocab = Vocab(paths["vocab_csv"], special_classes=data_cfg["folders"]["special_classes"])
    vp = vocab.validate()
    report["vocab"] = {"n_classes": vocab.N, "head_classes": vocab.num_head_classes,
                       "problems": vp}
    print(f"[vocab] {vocab.N} glosas en {paths['vocab_csv']} "
          f"(+{vocab.num_head_classes - vocab.N} clases especiales).")
    for p in vp:
        print(f"  PROBLEMA: {p}")

    # paso 2: signer_manifest.csv
    mproblems = []
    root = Path(paths["dataset_root"])
    manifest_file = paths["signer_manifest"]
    signer_map = load_signer_manifest(manifest_file, root)
    if not Path(manifest_file).is_file():
        mproblems.append(f"No existe {manifest_file}; el split caera a por-video (con aviso).")
    else:
        with open(manifest_file, newline="", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        video_exts = {e.lower() for e in data_cfg["folders"]["video_exts"]}
        for r in rows:
            raw = r["video_path"].strip()
            if Path(raw).suffix.lower() not in video_exts:
                mproblems.append(f"Extension no valida (¿typo tipo .mp5?): {raw}")
            # la columna gloss puede ser una glosa o una secuencia (frases/how2sign)
            bad = [t for t in r["gloss"].strip().split()
                   if t not in vocab and t not in vocab.special_classes]
            if bad:
                mproblems.append(f"Glosa(s) {bad} de '{r['gloss']}' del manifiesto "
                                 f"no estan en el vocabulario.")
        missing = [p for p in signer_map if not Path(p).is_file()]
        for p in missing[:20]:
            mproblems.append(f"Ruta del manifiesto no existe en disco: {p}")
        if len(missing) > 20:
            mproblems.append(f"... y {len(missing) - 20} rutas faltantes mas.")
        n_signers = len(set(signer_map.values()))
        report["manifest"].update({"rows": len(rows), "signers": n_signers,
                                   "missing_paths": len(missing)})
        print(f"[manifest] {len(rows)} filas, {n_signers} firmantes, "
              f"{len(missing)} rutas faltantes.")
    report["manifest"]["problems"] = mproblems
    for p in mproblems:
        print(f"  PROBLEMA: {p}")

    # paso 3: estructura del dataset
    samples, dwarn = scan_dataset(data_cfg)
    by_kind = {"video": 0, "image": 0}
    for s in samples:
        by_kind[s["kind"]] += 1
    covered = {s["gloss"] for s in samples}
    uncovered = [g for g in vocab.glosses if g not in covered]
    report["dataset"] = {"samples": len(samples), "videos": by_kind["video"],
                         "images": by_kind["image"], "classes_with_data": len(covered),
                         "classes_without_data": uncovered, "warnings": dwarn}
    print(f"[dataset] {by_kind['video']} videos + {by_kind['image']} imagenes mapeados; "
          f"{len(covered)}/{vocab.num_head_classes} clases con datos.")
    if uncovered:
        print(f"  AVISO: glosas sin ningun dato: {uncovered}")
    for w in dwarn:
        print(f"  AVISO: {w}")

    # paso 4: frases para el ctc
    phrases, invalid, pwarn = scan_phrases(data_cfg, vocab)
    report["phrases"] = {"valid": len(phrases), "invalid": invalid, "warnings": pwarn}
    print(f"[frases] {len(phrases)} validas, {len(invalid)} RECHAZADAS.")
    for iv in invalid:
        print(f"  RECHAZADA: {Path(iv['path']).name} -> {iv['reason']}")
    for w in pwarn:
        print(f"  AVISO: {w}")

    # paso 5: how2sign (fuente extra opcional)
    h2s_items, hwarn = scan_how2sign(data_cfg, vocab)
    report["how2sign"] = {"clips": len(h2s_items), "warnings": hwarn}
    if data_cfg.get("how2sign", {}).get("enabled"):
        by_split = {}
        for it in h2s_items:
            by_split[it["split"]] = by_split.get(it["split"], 0) + 1
        print(f"[how2sign] {len(h2s_items)} clips mapeados {by_split or ''}")
        for w in hwarn:
            print(f"  AVISO: {w}")

    report["warnings"] = dwarn + pwarn
    return report, vocab, samples, phrases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--strict", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", cfg)

    report, vocab, samples, phrases = validate(data_cfg)

    rdir = Path(data_cfg["paths"]["reports_dir"])
    rdir.mkdir(parents=True, exist_ok=True)
    out = rdir / "validation_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nReporte guardado en {out}")

    any_problem = (report["vocab"]["problems"] or report["manifest"].get("problems")
                   or report["phrases"]["invalid"] or report["warnings"])
    if args.strict and any_problem:
        print("MODO ESTRICTO: hay problemas; saliendo con error.")
        sys.exit(1)
    if not samples:
        print("ERROR: no se mapeo ningun dato utilizable.")
        sys.exit(2)
    print("Validacion terminada.")


if __name__ == "__main__":
    main()
