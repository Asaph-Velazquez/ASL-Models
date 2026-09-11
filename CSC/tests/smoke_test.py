"""SMOKE TEST: corre todo el pipeline con datos falsos en CPU, sin descargas ni entrenamiento real."""
import os
import shutil
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.trace import TraceRecorder

WORKDIR = Path(os.environ.get("SMOKE_WORKDIR", "outputs/smoke")).resolve()

DUMMY_VOCAB = [
    # canonical_gloss, english, functional_class, notes
    ("I", "I", "pronoun", ""),
    ("WANT", "Want", "verb", ""),
    ("TOWEL", "Towel", "object", ""),
    ("ROOM", "Room", "place", ""),
    ("PLEASE", "Please", "courtesy", ""),
    ("NOW", "Now", "time", ""),
    ("FS-A", "Letter A", "letter", "confusion con B"),
    ("FS-B", "Letter B", "letter", "confusion con A"),
    ("FS-J", "Letter J", "letter", "DINAMICA - video + imagen estatica"),
    ("NUM-1", "1", "number", ""),
    ("NUM-2", "2", "number", ""),
]
CONTENT = ["I", "WANT", "TOWEL", "ROOM", "PLEASE", "NOW"]
SIGNERS = ["s01", "s02", "s03", "s04"]


def make_fake_video(path):
    """mp4 real si hay cv2 (en Docker); si no, archivo stub (el preprocess es simulado)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 25, (64, 64))
        for _ in range(10):
            vw.write(np.random.randint(0, 255, (64, 64, 3), np.uint8))
        vw.release()
    except Exception:
        path.write_bytes(b"\x00\x00\x00\x18ftypmp42fake-smoke-video")


def make_fake_png(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(np.random.randint(0, 255, (64, 64, 3), np.uint8)).save(path)
    except Exception:
        path.write_bytes(b"\x89PNG\r\n\x1a\nfake")


def build_dummy_tree():
    root = WORKDIR / "dataset"
    ais = root / "Aisladas"
    manifest_rows = []
    for g in CONTENT:
        for i, s in enumerate(SIGNERS, 1):
            p = ais / g / f"{i:02d}.mp4"
            make_fake_video(p)
            manifest_rows.append(f"{g},dataset/Aisladas/{g}/{i:02d}.mp4,{s}")
    for i, s in enumerate(SIGNERS[:2], 1):  # FS-J dinamica en VIDEO (§4.2)
        p = ais / "FS-J" / f"{i:02d}.mp4"
        make_fake_video(p)
        manifest_rows.append(f"FS-J,dataset/Aisladas/FS-J/{i:02d}.mp4,{s}")
    for split, n in (("Train_Alphabet", 2), ("Test_Alphabet", 1)):
        for letter in ("A", "B", "J", "Blank"):  # J estatica -> misma glosa FS-J (§4.2)
            for k in range(n):
                make_fake_png(ais / "letras" / split / letter / f"img{k}.png")
    for split, n in (("Train_Nums", 2), ("Test_Nums", 1)):
        for d in ("1", "2"):
            for k in range(n):
                make_fake_png(ais / "numbers" / split / d / f"img{k}.png")
    make_fake_video(root / "Frases" / "I WANT TOWEL.mp4")       # valida
    make_fake_video(root / "Frases" / "I WANT TOWEL NOW.mp4")   # valida
    make_fake_video(root / "Frases" / "te quiero.mp4")          # DEBE ser rechazada

    # How2Sign simulado: manifiesto como el de map_how2sign_manifest.py + clips falsos
    h2s = WORKDIR / "external" / "how2sign"
    h2s_rows = [("I WANT TOWEL", "train", "h2s_vidA", "clipA"),
                ("ROOM PLEASE NOW", "train", "h2s_vidA", "clipB"),
                ("WANT TOWEL PLEASE", "val", "h2s_vidB", "clipC")]
    lines = ["video_path,gloss_sequence,signer_id,split,sentence"]
    for seq, split, signer, name in h2s_rows:
        clip = h2s / "video_clips" / split / f"{name}.mp4"
        make_fake_video(clip)
        lines.append(f"{clip},{seq},{signer},{split},{seq.lower()}")
    (h2s / "how2sign_manifest.csv").write_text("\n".join(lines) + "\n")

    (WORKDIR / "vocab.csv").write_text(
        "canonical_gloss,english,functional_class,notes\n" +
        "\n".join(",".join(r) for r in DUMMY_VOCAB) + "\n")
    (WORKDIR / "signer_manifest.csv").write_text(
        "gloss,video_path,signer_id\n" + "\n".join(manifest_rows) + "\n")
    return root


def write_configs():
    w = str(WORKDIR)
    cdir = WORKDIR / "configs"
    cdir.mkdir(parents=True, exist_ok=True)
    data = {
        "paths": {"dataset_root": f"{w}/dataset", "vocab_csv": f"{w}/vocab.csv",
                  "signer_manifest": f"{w}/signer_manifest.csv",
                  "cache_dir": f"{w}/outputs/cache", "reports_dir": f"{w}/reports",
                  "outputs_dir": f"{w}/outputs"},
        "folders": {"isolated_dir": "Aisladas", "letters_dir": "Aisladas/letras",
                    "numbers_dir": "Aisladas/numbers", "letters_train": "Train_Alphabet",
                    "letters_test": "Test_Alphabet", "numbers_train": "Train_Nums",
                    "numbers_test": "Test_Nums", "letter_prefix": "FS-",
                    "number_prefix": "NUM-", "blank_folder": "Blank",
                    "special_classes": ["BLANK"],
                    "video_exts": [".mp4", ".mov", ".avi", ".mkv", ".webm"],
                    "image_exts": [".png", ".jpg", ".jpeg"]},
        "phrases": {"dir": "Frases", "phrase_label_source": "filename",
                    "annotations_csv": "Frases/annotations.csv"},
        "how2sign": {"enabled": True, "root": f"{w}/external/how2sign",
                     "manifest": f"{w}/external/how2sign/how2sign_manifest.csv"},
        "splits": {"mode": "signer_independent", "test_fraction": 0.25, "seed": 7},
        "static_to_sequence": {"n_frames": 8, "jitter_std": 0.01},
        "landmarks": {"pose_upper_indices": [0, 11, 12, 13, 14, 15, 16, 23, 24],
                      "hand_points": 21, "min_detection_confidence": 0.5,
                      "min_tracking_confidence": 0.5},
        "video_filters": {"enabled": True, "target_fps": 25, "low_luma_threshold": 60,
                          "blur_threshold": 60.0, "noise_threshold": 12.0,
                          "min_detection_rate": 0.5, "hand_crop_margin": 0.35},
        "normalization": {"hand_relative_classes": ["letter", "number"],
                          "interpolate_missing": True},
        "trace": {"enabled": True, "max_sample_values": 6},
    }
    model = {"spatial": {"type": "mlp", "hidden_dims": [64, 64], "dropout": 0.1},
             "temporal": {"type": "bilstm", "hidden_dim": 64, "num_layers": 1,
                          "dropout": 0.0}}
    train_base = {"epochs": 1, "batch_size": 8, "lr": 1e-3, "weight_decay": 1e-4,
                  "max_seq_len": 64, "early_stop_patience": 3, "num_workers": 0}
    cfgs = {
        "data.yaml": data,
        "phase1.yaml": {"data_config": f"{cdir}/data.yaml", "phase": "phase1", "seed": 7,
                        "deterministic": True, "sources": ["letters", "numbers"],
                        "model": model, "train": train_base, "init_from": None,
                        "checkpoint_dir": f"{w}/checkpoints/phase1",
                        "log_dir": f"{w}/reports/tensorboard/phase1", "trace": True},
        "phase2.yaml": {"data_config": f"{cdir}/data.yaml", "phase": "phase2", "seed": 7,
                        "deterministic": True, "sources": ["videos", "letters", "numbers"],
                        "model": model, "train": train_base,
                        "init_from": f"{w}/checkpoints/phase1/best.pt",
                        "freeze_spatial_epochs": 0,
                        "checkpoint_dir": f"{w}/checkpoints/phase2",
                        "log_dir": f"{w}/reports/tensorboard/phase2", "trace": False},
        "synthetic.yaml": {"data_config": f"{cdir}/data.yaml",
                           "clips_manifest": f"{w}/outputs/cache/clips_manifest.csv",
                           "out_dir": f"{w}/outputs/synthetic", "n_train": 8, "n_val": 3,
                           "n_test": 3, "seed": 7,
                           "concat": {"transition_frames_min": 2,
                                      "transition_frames_max": 4,
                                      "time_warp": [0.9, 1.1], "trim_rest_ratio": 0.15},
                           "augment": {"jitter": 0.01, "frame_drop_p": 0.05}},
        "phase3.yaml": {"data_config": f"{cdir}/data.yaml", "phase": "phase3", "seed": 7,
                        "deterministic": True, "synthetic_dir": f"{w}/outputs/synthetic",
                        "model": model,
                        "train": {**train_base, "batch_size": 4, "max_seq_len": 256,
                                  "real_weight": 2.0, "synthetic_only_epochs": 0,
                                  "how2sign_weight": 0.5, "how2sign_in_val": False},
                        "init_from": f"{w}/checkpoints/phase2/best.pt",
                        "checkpoint_dir": f"{w}/checkpoints/phase3",
                        "log_dir": f"{w}/reports/tensorboard/phase3",
                        "decode": {"beam_size": 4, "lm_ngram": 2, "lm_alpha": 0.3,
                                   "lm_beta": 0.6},
                        "trace": True},
        "phase4.yaml": {"data_config": f"{cdir}/data.yaml", "phase": "phase4", "seed": 7,
                        "deterministic": True,
                        "parallel_corpus_dir": f"{w}/outputs/synthetic",
                        "model_name": "(no se descarga en smoke)",
                        "source_prefix": "translate ASL gloss to English: ",
                        "train": {"epochs": 1, "batch_size": 4, "lr": 1e-3,
                                  "weight_decay": 0.01, "max_source_len": 32,
                                  "max_target_len": 32, "early_stop_patience": 2,
                                  "num_workers": 0},
                        "checkpoint_dir": f"{w}/checkpoints/phase4",
                        "log_dir": f"{w}/reports/tensorboard/phase4", "trace": False},
        "eval.yaml": {"data_config": f"{cdir}/data.yaml", "seed": 7,
                      "checkpoints": {"isolated": f"{w}/checkpoints/phase2/best.pt",
                                      "ctc": f"{w}/checkpoints/phase3/best.pt",
                                      "translator": f"{w}/checkpoints/phase4"},
                      "synthetic_dir": f"{w}/outputs/synthetic",
                      "phase3_config": f"{cdir}/phase3.yaml",
                      "phase4_config": f"{cdir}/phase4.yaml",
                      "reports_dir": f"{w}/reports", "qualitative_n": 10,
                      "skeleton_animation_n": 1, "batch_size": 4, "trace": False},
    }
    for name, c in cfgs.items():
        (cdir / name).write_text(yaml.safe_dump(c, sort_keys=False, allow_unicode=True))
    return cdir


def main():
    t0 = time.time()
    if WORKDIR.exists():
        shutil.rmtree(WORKDIR)
    WORKDIR.mkdir(parents=True)
    tracer = TraceRecorder(enabled=True, reports_dir=WORKDIR / "reports")
    ok = []

    print(f"=== SMOKE TEST (workdir: {WORKDIR}) ===\n")

    with tracer.stage("Datos dummy", "vocabulario + carpetas §4 con archivos falsos") as st:
        build_dummy_tree()
        cdir = write_configs()
        st.metric("estructura", "Aisladas/ + letras/ + numbers/ + Frases/")
    ok.append("datos dummy")

    # 1) Validacion
    from utils.config import load_config
    from data.validate_manifest import validate
    data_cfg = load_config(cdir / "data.yaml")
    with tracer.stage("Validacion", "vocabulario, manifiesto, carpetas y frases") as st:
        report, vocab, samples, phrases = validate(data_cfg)
        assert not report["vocab"]["problems"], report["vocab"]["problems"]
        rejected = [Path(i["path"]).name for i in report["phrases"]["invalid"]]
        assert "te quiero.mp4" in rejected, "el validador NO rechazo 'te quiero.mp4'"
        assert report["phrases"]["valid"] == 2
        st.metric("muestras_mapeadas", len(samples))
        st.metric("frases_rechazadas", rejected)
    ok.append("validacion (rechaza nombres en lenguaje natural)")

    # 1-bis) Puente How2Sign: ingles -> pseudo-glosas (sin descargas)
    from data.how2sign import build_gloss_mapper, map_sentence
    with tracer.stage("How2Sign", "ingles -> pseudo-glosas del vocabulario") as st:
        w2g, stop = build_gloss_mapper(vocab)
        seq, info = map_sentence("I want a towel now!", w2g, stop)
        assert seq == "I WANT TOWEL NOW", f"mapeo inesperado: {seq}"
        seq_num, _ = map_sentence("I want one towel", w2g, stop)
        assert seq_num == "I WANT NUM-1 TOWEL", f"numeros mal mapeados: {seq_num}"
        rej, _ = map_sentence("the weather is beautiful today", w2g, stop)
        assert rej is None, "una oracion sin cobertura NO debe aceptarse"
        st.metric("ejemplo", f"'I want a towel now!' -> {seq}")
    ok.append("puente How2Sign (pseudo-glosas + filtro de cobertura)")

    # 2) Preprocesamiento SIMULADO (landmarks falsos, sin MediaPipe)
    from data.preprocess import run_preprocess
    run_preprocess(cdir / "data.yaml", fake=True, tracer=tracer)
    cache = WORKDIR / "outputs/cache"
    assert (cache / "clips_manifest.csv").is_file() and (cache / "layout.json").is_file()
    from data.datasets import read_manifest
    rows = read_manifest(cache / "clips_manifest.csv")
    splits = {r["split"] for r in rows}
    assert splits == {"train", "test"}, f"splits incompletos: {splits}"
    fsj = {r["kind"] for r in rows if r["gloss"] == "FS-J"}
    assert fsj == {"video", "image"}, f"FS-J debe tener AMBAS fuentes (§4.2): {fsj}"
    hrows = read_manifest(cache / "how2sign_phrases_manifest.csv")
    assert len(hrows) == 3, f"esperaba 3 frases How2Sign en cache: {len(hrows)}"
    assert {r["split"] for r in hrows} == {"train", "val"}, "splits How2Sign no respetados"
    ok.append("preprocesamiento simulado + splits + FS-J ambas fuentes + How2Sign en cache")

    # 3) Fase 1: forward encoder + softmax + 1 epoca (letras/numeros)
    from train.common import train_classification
    s1 = train_classification(load_config(cdir / "phase1.yaml"), "phase1")
    ok.append(f"fase 1 (clasificacion imagenes, top1={s1['best_val_top1']:.2f})")

    # 4) Fase 2: transferencia del encoder + vocabulario completo
    s2 = train_classification(load_config(cdir / "phase2.yaml"), "phase2")
    assert s2["num_classes"] == len(DUMMY_VOCAB) + 1, "N debe ser len(vocab)+BLANK"
    ok.append("fase 2 (transferencia encoder, N=len(vocab)+especiales)")

    # 5) Generacion sintetica minima + corpus paralelo
    from synthetic.generator import run_generator
    with tracer.stage("Sintetico", "concatenacion de clips + corpus paralelo") as st:
        srows = run_generator(cdir / "synthetic.yaml")
        assert len(srows) >= 6, f"muy pocas frases sinteticas: {len(srows)}"
        assert (WORKDIR / "outputs/synthetic/parallel_corpus_train.csv").is_file()
        st.metric("frases_sinteticas", len(srows))
        st.metric("ejemplo", f"[{srows[0]['gloss_sequence']}] -> {srows[0]['english']}")
    ok.append("generacion sintetica + corpus paralelo")

    # 6) Fase 3: forward + perdida CTC (1 epoca) + decodificacion greedy,
    #    mezclando sintetico + frases reales + How2Sign (peso debil)
    import json
    from train.train_phase3 import train_phase3
    train_phase3(load_config(cdir / "phase3.yaml"))
    assert (WORKDIR / "checkpoints/phase3/best.pt").is_file()
    summary = json.loads((WORKDIR / "checkpoints/phase3/summary.json").read_text())
    assert summary["how2sign_items"] == 2, \
        f"la Fase 3 debia entrenar con 2 clips How2Sign: {summary['how2sign_items']}"
    ok.append("fase 3 (CTC sintetico + real + How2Sign con peso debil)")

    # 7) 1 paso de T5 diminuto (aleatorio, SIN descargas) + generacion
    from models.translator import Translator
    from train.train_phase4 import load_corpus, train_phase4
    pairs = load_corpus(WORKDIR / "outputs/synthetic", "train")
    tiny = Translator.tiny([s for s, _ in pairs] + [t for _, t in pairs])
    with tracer.stage("Capa 6", "1 paso de entrenamiento T5 + generacion") as st:
        train_phase4(load_config(cdir / "phase4.yaml"), translator=tiny)
        out = tiny.translate([pairs[0][0]])[0]
        st.metric("t5_genero", out or "(vacio; esperable con pesos aleatorios)")
    ok.append("fase 4 (T5 tiny sin descargas)")

    # 8) Metricas + reportes (WER, BLEU, matriz de confusion, esqueleto, tabla)
    from eval.metrics import run_eval
    rep = run_eval(cdir / "eval.yaml")
    assert "aislado" in rep and "continuo" in rep and "traduccion" in rep
    assert (WORKDIR / "reports/confusion_matrix.png").is_file()
    assert (WORKDIR / "reports/qualitative_table.csv").is_file()
    ok.append("evaluacion (WER/BLEU/matriz/tabla cualitativa)")

    # 9) Demo end-to-end con traza
    from demo.infer import run_demo
    args = Namespace(video=str(WORKDIR / "dataset/Frases/I WANT TOWEL.mp4"),
                     config=str(cdir / "data.yaml"),
                     phase3_config=str(cdir / "phase3.yaml"),
                     ctc_checkpoint=str(WORKDIR / "checkpoints/phase3/best.pt"),
                     translator_dir=str(WORKDIR / "checkpoints/phase4/best"),
                     synthetic_dir=str(WORKDIR / "outputs/synthetic"),
                     out=str(WORKDIR / "outputs/demo"), fake_landmarks=True)
    glosas, ingles = run_demo(args)
    assert ingles is not None
    ok.append("demo (video -> glosas -> ingles, con traza)")

    tracer.save("pipeline_trace")
    print("\n=== SMOKE TEST OK ===")
    for i, item in enumerate(ok, 1):
        print(f"  {i}. {item}")
    print(f"\nTiempo total: {time.time() - t0:.1f}s")
    print(f"Traza por capa: {WORKDIR}/reports/pipeline_trace.json / .txt")
    print("La estructura encaja. Siguiente paso: colocar el dataset real y correr "
          "`docker compose run --rm preprocess`.")


if __name__ == "__main__":
    main()
