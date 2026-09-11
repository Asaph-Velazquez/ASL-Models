import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.config import load_config
from utils.trace import TraceRecorder


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--phase3-config", default="configs/phase3.yaml")
    ap.add_argument("--ctc-checkpoint", default="checkpoints/phase3/best.pt")
    ap.add_argument("--translator-dir", default="checkpoints/phase4/best")
    ap.add_argument("--synthetic-dir", default="outputs/synthetic")
    ap.add_argument("--out", default="outputs/demo")
    ap.add_argument("--fake-landmarks", action="store_true",
                    help="landmarks falsos (solo smoke test)")
    args = ap.parse_args()
    run_demo(args)


def run_demo(args, translator=None):
    import torch
    from data.normalize import load_layout, normalize_clip
    from data.preprocess import (LandmarkExtractor, build_layout, fake_landmarks,
                                 process_video)
    from data.vocab import Vocab
    from data.datasets import read_manifest
    from models.ctc_decoder import GlossNGram, beam_search_decode
    from models.heads import CTCModel
    from models.translator import Translator, translate_glosses
    from train.common import get_device

    cfg = load_config(args.config)
    data_cfg = cfg.get("data", cfg)
    device = get_device()
    out_dir = Path(args.out) / Path(args.video).stem.replace(" ", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    tracer = TraceRecorder(enabled=True, reports_dir=out_dir)  # demo SIEMPRE con traza

    try:
        layout = load_layout(data_cfg["paths"]["cache_dir"])
    except FileNotFoundError:
        layout = build_layout(data_cfg["landmarks"]["pose_upper_indices"])

    # Capa 0-bis + Capa 0
    with tracer.stage("Capa 0", "video -> landmarks (MediaPipe manos+torso, sin cara)") as st:
        if args.fake_landmarks:
            arr = fake_landmarks(layout, T=60)
            st.note("MODO SMOKE: landmarks falsos, sin MediaPipe")
            log = {"filtros": [], "tasa_deteccion_manos": 1.0}
        else:
            extractor = LandmarkExtractor(data_cfg["landmarks"], layout, static=False)
            arr, log = process_video(args.video, extractor, layout,
                                     data_cfg["video_filters"])
            extractor.close()
        st.io(inp=None, out=arr)
        st.metric("frames", len(arr))
        st.metric("tasa_deteccion_manos", log["tasa_deteccion_manos"])
        st.metric("filtros_aplicados (Capa 0-bis)", log["filtros"])

    # Capa 1
    with tracer.stage("Capa 1", "normalizacion relativa al cuerpo + pos/vel/acel") as st:
        feats, info = normalize_clip(arr, layout, "body",
                                     data_cfg["normalization"]["interpolate_missing"])
        st.io(inp=arr, out=feats)
        st.metric("pct_frames_interpolados", round(info["pct_interpolado"], 2))
        st.metric("camino_normalizacion", info["camino"])

    # Capas 2-4B
    vocab = Vocab(data_cfg["paths"]["vocab_csv"],
                  special_classes=data_cfg["folders"]["special_classes"])
    ckpt = Path(args.ctc_checkpoint)
    if not ckpt.is_file():
        print(f"ERROR: falta el checkpoint CTC {ckpt}. Entrena la Fase 3 primero.")
        if not args.fake_landmarks:
            sys.exit(1)
        # smoke: modelo aleatorio para verificar el encadenamiento
        p3 = load_config(args.phase3_config)
        model = CTCModel(p3["model"], feats.shape[1], vocab.num_ctc_classes).to(device)
        dcfg = p3["decode"]
        print("MODO SMOKE: usando modelo CTC aleatorio para verificar el encadenamiento.")
    else:
        state = torch.load(ckpt, map_location=device)
        model = CTCModel(state["model_cfg"], state["input_dim"],
                         state["num_ctc_classes"]).to(device)
        model.load_state_dict(state["model"])
        dcfg = state.get("decode_cfg", {"beam_size": 8, "lm_ngram": 2,
                                        "lm_alpha": 0.3, "lm_beta": 0.6})
    model.eval()

    x = torch.from_numpy(feats[None]).float().to(device)
    with tracer.stage("Capas 2-3", "encoder espacial + temporal") as st:
        with torch.no_grad():
            h = model.encoder(x)
        st.io(inp=x, out=h)
    with tracer.stage("Capa 4B", "logits CTC por frame") as st:
        with torch.no_grad():
            logits = model.head(h)
        st.io(inp=h, out=logits)

    # Capa 5
    with tracer.stage("Capa 5", "beam search CTC + n-grama de glosas") as st:
        lm = None
        syn = Path(args.synthetic_dir) / "synthetic_manifest_train.csv"
        if syn.is_file():
            lm = GlossNGram.from_sequences(
                [r["gloss_sequence"] for r in read_manifest(syn)], order=dcfg["lm_ngram"])
            st.note("n-grama de glosas cargado del corpus sintetico")
        logp = torch.log_softmax(logits, dim=-1)[0].cpu().numpy()
        glosas = beam_search_decode(logp, vocab, beam_size=dcfg["beam_size"], lm=lm,
                                    lm_alpha=dcfg["lm_alpha"], lm_beta=dcfg["lm_beta"])
        st.io(inp=logits, out=None)
        st.metric("SALIDA 1 - glosas", glosas or "(vacio)")

    # Capa 6
    with tracer.stage("Capa 6", "traduccion glosa->ingles (T5 con fallback a plantillas)") as st:
        if translator is None and (Path(args.translator_dir) / "config.json").is_file():
            try:
                translator = Translator.from_pretrained(args.translator_dir, device=device)
            except Exception as e:
                st.note(f"T5 no cargado ({e}); plantillas")
        ingles, fuente = translate_glosses(glosas, vocab, translator) if glosas \
            else ("", "plantillas")
        st.metric("SALIDA 2 - ingles", ingles or "(vacio)")
        st.metric("fuente", fuente)

    (out_dir / "resultado.txt").write_text(
        f"video: {args.video}\nglosas: {glosas}\ningles: {ingles}\n")
    tracer.save("pipeline_trace")

    if not args.fake_landmarks:
        try:
            render_side_by_side(args.video, arr, layout, glosas, ingles,
                                out_dir / "lado_a_lado.mp4", data_cfg)
            print(f"Video lado a lado -> {out_dir / 'lado_a_lado.mp4'}")
        except Exception as e:
            print(f"AVISO: no se pudo renderizar el video ({e}).")

    print(f"\n=== RESULTADO ===\nGLOSAS: {glosas}\nINGLES: {ingles}\n(detalle en {out_dir})")
    return glosas, ingles


def render_side_by_side(video_path, landmarks, layout, glosas, ingles, out_path, data_cfg):
    """Original | esqueleto, con las salidas sobreimpresas."""
    import cv2
    from data.preprocess import read_video_frames
    frames, _, _ = read_video_frames(video_path, data_cfg["video_filters"]["target_fps"])
    T = min(len(frames), len(landmarks))
    h, w = frames[0].shape[:2]
    scale = 480 / h
    w2, h2 = int(w * scale), 480
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                         data_cfg["video_filters"]["target_fps"], (w2 * 2, h2))
    for t in range(T):
        left = cv2.resize(frames[t], (w2, h2))
        right = np.full((h2, w2, 3), 255, np.uint8)
        for name, color in (("lh", (0, 0, 255)), ("rh", (255, 0, 0)), ("pose", (0, 160, 0))):
            a, b = layout["offsets"][name]
            pts = landmarks[t, a:b].reshape(-1, 3)
            for p in pts:
                if p[0] > 0 or p[1] > 0:
                    cv2.circle(right, (int(p[0] * w2), int(p[1] * h2)), 3, color, -1)
        canvas = np.concatenate([left, right], axis=1)
        cv2.putText(canvas, f"GLOSAS: {glosas}"[:90], (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        cv2.putText(canvas, f"EN: {ingles}"[:90], (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
        vw.write(canvas)
    vw.release()


if __name__ == "__main__":
    main()
