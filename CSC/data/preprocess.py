"""capas 0-bis/0: filtrado de video + landmarks mediapipe (sin cara) -> cache .npy"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.folder_mapping import scan_dataset, scan_how2sign, scan_phrases
from data.splits import (assign_phrase_splits, assign_splits, load_signer_manifest)
from data.vocab import Vocab
from utils.config import load_config
from utils.trace import TraceRecorder

HAND_POINTS = 21


def build_layout(pose_upper_indices):
    P = len(pose_upper_indices)
    lh, rh, pose = HAND_POINTS * 3, HAND_POINTS * 3, P * 3
    return {
        "pose_upper_indices": list(pose_upper_indices),
        "hand_points": HAND_POINTS,
        "offsets": {"lh": [0, lh], "rh": [lh, lh + rh],
                    "pose": [lh + rh, lh + rh + pose],
                    "mask": [lh + rh + pose, lh + rh + pose + 3]},
        "D": lh + rh + pose + 3,
        "uses_face": False,
        "nota": "Sin cara: se pierden marcadores no manuales; salida = secuencia lexica de glosas.",
    }


# capa 0-bis: calidad y filtros automaticos

def assess_frame_quality(gray, cfg):
    import cv2
    luma = float(gray.mean())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    noise = float(np.std(gray.astype(np.float32) - cv2.GaussianBlur(gray, (5, 5), 0)))
    filters = []
    if luma < cfg["low_luma_threshold"]:
        filters.append("clahe_brillo")
    if blur < cfg["blur_threshold"]:
        filters.append("nitidez")
    if noise > cfg["noise_threshold"]:
        filters.append("denoise")
    return {"luma": luma, "blur_var": blur, "noise": noise}, filters


def apply_filters(frame, filters):
    import cv2
    out = frame
    if "denoise" in filters:
        out = cv2.bilateralFilter(out, 5, 50, 50)
    if "clahe_brillo" in filters:
        ycrcb = cv2.cvtColor(out, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        out = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    if "nitidez" in filters:
        blurred = cv2.GaussianBlur(out, (0, 0), 3)
        out = cv2.addWeighted(out, 1.5, blurred, -0.5, 0)
    return out


def read_video_frames(path, target_fps):
    """lee el video corrigiendo fps al objetivo"""
    import cv2
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise IOError(f"No se pudo abrir el video: {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise IOError(f"Video sin frames: {path}")
    if src_fps and abs(src_fps - target_fps) > 1:
        n_out = max(1, int(round(len(frames) * target_fps / src_fps)))
        idx = np.linspace(0, len(frames) - 1, n_out).round().astype(int)
        frames = [frames[i] for i in idx]
        fps_fixed = True
    else:
        fps_fixed = False
    return frames, src_fps, fps_fixed


def crop_to_hands(frames, landmarks_seq, layout, margin):
    """zoom a la region de manos usando lo ya detectado"""
    import cv2
    pts = []
    a_m = layout["offsets"]["mask"][0]
    for arr in landmarks_seq:
        for name in ("lh", "rh", "pose"):
            a, b = layout["offsets"][name]
            block = arr[a:b].reshape(-1, 3)
            valid = block[(block[:, 0] > 0) & (block[:, 1] > 0)]
            pts.extend(valid[:, :2].tolist())
    if not pts:
        return frames, False
    pts = np.array(pts)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    dx, dy = (x1 - x0) * margin, (y1 - y0) * margin
    x0, y0 = max(0.0, x0 - dx), max(0.0, y0 - dy)
    x1, y1 = min(1.0, x1 + dx), min(1.0, y1 + dy)
    if (x1 - x0) < 0.05 or (y1 - y0) < 0.05 or ((x1 - x0) > 0.9 and (y1 - y0) > 0.9):
        return frames, False
    h, w = frames[0].shape[:2]
    ax0, ay0, ax1, ay1 = int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)
    out = [cv2.resize(f[ay0:ay1, ax0:ax1], (w, h)) for f in frames]
    return out, True


# capa 0: mediapipe holistic (manos + torso, sin cara)

class LandmarkExtractor:
    def __init__(self, lm_cfg, layout, static=False):
        import mediapipe as mp
        self.layout = layout
        self.holistic = mp.solutions.holistic.Holistic(
            static_image_mode=static,
            model_complexity=1,
            refine_face_landmarks=False,   # la cara no se usa
            min_detection_confidence=lm_cfg["min_detection_confidence"],
            min_tracking_confidence=lm_cfg["min_tracking_confidence"],
        )

    def close(self):
        self.holistic.close()

    def extract_frame(self, frame_bgr):
        import cv2
        res = self.holistic.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        D = self.layout["D"]
        out = np.zeros(D, dtype=np.float32)
        off = self.layout["offsets"]

        def fill(lms, a, indices=None):
            pts = lms.landmark
            sel = indices if indices is not None else range(len(pts))
            for j, i in enumerate(sel):
                out[a + 3 * j: a + 3 * j + 3] = (pts[i].x, pts[i].y, pts[i].z)

        m = off["mask"][0]
        if res.left_hand_landmarks:
            fill(res.left_hand_landmarks, off["lh"][0])
            out[m] = 1.0
        if res.right_hand_landmarks:
            fill(res.right_hand_landmarks, off["rh"][0])
            out[m + 1] = 1.0
        if res.pose_landmarks:
            fill(res.pose_landmarks, off["pose"][0], self.layout["pose_upper_indices"])
            out[m + 2] = 1.0
        return out


def detection_rate(seq, layout):
    m = layout["offsets"]["mask"][0]
    arr = np.stack(seq)
    hands = np.maximum(arr[:, m], arr[:, m + 1])
    return float(hands.mean())


def process_video(path, extractor, layout, vf_cfg):
    """video -> [T, D] con filtrado adaptativo"""
    frames, src_fps, fps_fixed = read_video_frames(path, vf_cfg["target_fps"])
    log = {"path": str(path), "src_fps": src_fps, "fps_corregido": fps_fixed,
           "n_frames": len(frames), "filtros": [], "reintentos": 0}

    import cv2
    gray = cv2.cvtColor(frames[len(frames) // 2], cv2.COLOR_BGR2GRAY)
    quality, filters = assess_frame_quality(gray, vf_cfg)
    log["calidad"] = quality
    if vf_cfg["enabled"] and filters:
        frames = [apply_filters(f, filters) for f in frames]
        log["filtros"] = filters

    seq = [extractor.extract_frame(f) for f in frames]
    rate = detection_rate(seq, layout)

    if vf_cfg["enabled"] and rate < vf_cfg["min_detection_rate"]:
        # reintento: filtros faltantes + zoom a manos
        log["reintentos"] += 1
        extra = [f for f in ("clahe_brillo", "nitidez") if f not in log["filtros"]]
        if extra:
            frames = [apply_filters(f, extra) for f in frames]
            log["filtros"] += extra
        frames2, cropped = crop_to_hands(frames, seq, layout, vf_cfg["hand_crop_margin"])
        if cropped:
            log["filtros"].append("zoom_manos")
        seq2 = [extractor.extract_frame(f) for f in frames2]
        if detection_rate(seq2, layout) > rate:
            seq = seq2
        rate = max(rate, detection_rate(seq, layout))

    log["tasa_deteccion_manos"] = round(rate, 3)
    return np.stack(seq).astype(np.float32), log


def process_image(path, extractor, layout):
    """imagen estatica -> [1, D]; la expansion a mini-secuencia ocurre al cargar"""
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        raise IOError(f"No se pudo leer la imagen: {path}")
    return extractor.extract_frame(img)[None, :].astype(np.float32)


def fake_landmarks(layout, T=None, rng=None):
    """landmarks falsos plausibles, solo smoke test"""
    rng = rng or np.random.default_rng(0)
    T = T or int(rng.integers(12, 40))
    D = layout["D"]
    arr = np.cumsum(rng.normal(0, 0.01, (T, D)).astype(np.float32), axis=0) + 0.5
    m = layout["offsets"]["mask"][0]
    arr[:, m:m + 3] = 1.0
    return arr


# orquestacion

def cache_path(cache_dir, sub, src_path, dataset_root):
    src = Path(src_path)
    try:
        rel = src.relative_to(dataset_root)
    except ValueError:
        rel = Path(src.name)
    return Path(cache_dir) / sub / rel.with_suffix(".npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--fake", action="store_true", help="landmarks falsos (smoke test)")
    ap.add_argument("--limit", type=int, default=0, help="procesar solo N muestras (prueba rapida)")
    args = ap.parse_args()
    run_preprocess(args.config, fake=args.fake, limit=args.limit)


def run_preprocess(config_path, fake=False, limit=0, tracer=None):
    cfg = load_config(config_path)
    data_cfg = cfg.get("data", cfg)
    paths = data_cfg["paths"]
    cache_dir = Path(paths["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    tracer = tracer or TraceRecorder(enabled=data_cfg.get("trace", {}).get("enabled", False),
                                     reports_dir=paths["reports_dir"])

    layout = build_layout(data_cfg["landmarks"]["pose_upper_indices"])
    (cache_dir / "layout.json").write_text(json.dumps(layout, indent=2, ensure_ascii=False))

    vocab = Vocab(paths["vocab_csv"], special_classes=data_cfg["folders"]["special_classes"])
    samples, warns = scan_dataset(data_cfg)
    signer_map = load_signer_manifest(paths["signer_manifest"], paths["dataset_root"])
    samples, swarns = assign_splits(samples, signer_map,
                                    data_cfg["splits"]["test_fraction"],
                                    data_cfg["splits"]["seed"])
    phrases, invalid, pwarns = scan_phrases(data_cfg, vocab)
    for p in phrases:  # signer_id: annotations.csv manda, si no signer_manifest
        if not p.get("signer_id"):
            p["signer_id"] = signer_map.get(p["path"])
    phrases, pswarns = assign_phrase_splits(phrases, data_cfg["splits"]["test_fraction"],
                                            data_cfg["splits"]["seed"])
    h2s_items, hwarns = scan_how2sign(data_cfg, vocab)
    for w in warns + swarns + pwarns + pswarns + hwarns:
        print(f"AVISO: {w}")
    if invalid:
        print(f"AVISO: {len(invalid)} frases rechazadas por el validador (no se procesan).")

    if limit:
        samples = samples[:limit]
        phrases = phrases[:limit]
        h2s_items = h2s_items[:limit]

    extractor_v = extractor_i = None
    if not fake:
        extractor_v = LandmarkExtractor(data_cfg["landmarks"], layout, static=False)
        extractor_i = LandmarkExtractor(data_cfg["landmarks"], layout, static=True)
    rng = np.random.default_rng(data_cfg["splits"]["seed"])

    logs, rows, n_done, n_fail, t0 = [], [], 0, 0, time.time()
    det_rates = []
    for s in samples:
        out = cache_path(cache_dir, "landmarks", s["path"], paths["dataset_root"])
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            if out.exists():
                arr = np.load(out)
            elif fake:
                arr = fake_landmarks(layout, T=1 if s["kind"] == "image" else None, rng=rng)
                np.save(out, arr)
            elif s["kind"] == "video":
                arr, log = process_video(s["path"], extractor_v, layout,
                                         data_cfg["video_filters"])
                det_rates.append(log["tasa_deteccion_manos"])
                logs.append(log)
                np.save(out, arr)
            else:
                arr = process_image(s["path"], extractor_i, layout)
                np.save(out, arr)
            rows.append({"gloss": s["gloss"], "npy_path": str(out),
                         "signer_id": s.get("signer_id") or "",
                         "split": s["split"], "kind": s["kind"], "source": s["source"]})
            n_done += 1
        except Exception as e:
            n_fail += 1
            print(f"FALLO {s['path']}: {e}")
        if n_done and n_done % 200 == 0:
            print(f"  ... {n_done}/{len(samples)} muestras ({time.time() - t0:.0f}s)")

    with open(cache_dir / "clips_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["gloss", "npy_path", "signer_id", "split",
                                          "kind", "source"])
        w.writeheader()
        w.writerows(rows)

    prow = []
    for p in phrases:
        out = cache_path(cache_dir, "phrases", p["path"], paths["dataset_root"])
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not out.exists():
                if fake:
                    arr = fake_landmarks(layout, T=int(rng.integers(40, 90)), rng=rng)
                else:
                    arr, log = process_video(p["path"], extractor_v, layout,
                                             data_cfg["video_filters"])
                    logs.append(log)
                np.save(out, arr)
            prow.append({"npy_path": str(out), "gloss_sequence": p["gloss_sequence"],
                         "signer_id": p.get("signer_id") or "", "split": p["split"]})
        except Exception as e:
            n_fail += 1
            print(f"FALLO frase {p['path']}: {e}")

    with open(cache_dir / "phrases_manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["npy_path", "gloss_sequence", "signer_id", "split"])
        w.writeheader()
        w.writerows(prow)

    hrow = []
    for it in h2s_items:
        out = cache_path(cache_dir, "how2sign", it["path"], paths["dataset_root"])
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            if not out.exists():
                if fake:
                    arr = fake_landmarks(layout, T=int(rng.integers(40, 90)), rng=rng)
                else:
                    arr, log = process_video(it["path"], extractor_v, layout,
                                             data_cfg["video_filters"])
                    logs.append(log)
                np.save(out, arr)
            hrow.append({"npy_path": str(out), "gloss_sequence": it["gloss_sequence"],
                         "signer_id": it["signer_id"], "split": it["split"]})
        except Exception as e:
            n_fail += 1
            print(f"FALLO How2Sign {it['path']}: {e}")
        if len(hrow) and len(hrow) % 200 == 0:
            print(f"  ... How2Sign {len(hrow)}/{len(h2s_items)} ({time.time() - t0:.0f}s)")

    if h2s_items or (cache_dir / "how2sign_phrases_manifest.csv").is_file():
        with open(cache_dir / "how2sign_phrases_manifest.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["npy_path", "gloss_sequence",
                                              "signer_id", "split"])
            w.writeheader()
            w.writerows(hrow)

    (cache_dir / "preprocess_log.json").write_text(
        json.dumps(logs, indent=2, ensure_ascii=False))

    with tracer.stage("Capa 0", "videos/imagenes -> landmarks .npy (manos+torso, sin cara)") as st:
        if rows:
            sample_arr = np.load(rows[0]["npy_path"])
            st.io(inp=None, out=sample_arr)
        st.metric("muestras_procesadas", n_done)
        st.metric("fallos", n_fail)
        st.metric("frases_procesadas", len(prow))
        st.metric("frases_how2sign", len(hrow))
        if det_rates:
            st.metric("tasa_deteccion_manos_media", round(float(np.mean(det_rates)), 3))
        st.note(f"layout D={layout['D']} -> {cache_dir}/layout.json; "
                f"manifiestos: clips_manifest.csv ({len(rows)}), phrases_manifest.csv "
                f"({len(prow)}), how2sign_phrases_manifest.csv ({len(hrow)})")

    if extractor_v:
        extractor_v.close()
        extractor_i.close()
    print(f"Preprocesamiento: {n_done} muestras + {len(prow)} frases + {len(hrow)} "
          f"How2Sign -> {cache_dir} ({n_fail} fallos, {time.time() - t0:.0f}s)")
    return cache_dir


if __name__ == "__main__":
    main()
