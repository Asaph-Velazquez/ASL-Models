"""
compare (SVM vs DTW)
"""
import argparse
import csv
import json
import os
import re
import sys
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.decomposition import KernelPCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

# ===================== CONFIGURACION =====================
CSV_FILE = "datos_30.csv"
DELIMITER = None
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_JOBS = -1
DEFAULT_OUTPUT_DIR = "models"
DEFAULT_RESULTS_DIR = "results"
PREFERRED_HAND_ORDER = ["Left", "Right"]
MODEL_FILENAMES = {
    "svm": "model_svm.joblib",
    "dtw": "model_dtw.joblib",
}


def percentile25(x, axis=0):
    return np.percentile(x, 25, axis=axis)


def percentile75(x, axis=0):
    return np.percentile(x, 75, axis=axis)


AGGREGATION_FUNCTIONS = [
    np.mean,
    np.std,
    np.min,
    np.max,
    percentile25,
    percentile75,
]
# =========================================================

KPCA_FEATURE_RE = re.compile(r"^kpca_(\d+)$", re.IGNORECASE)
LANDMARK_FEATURE_RE = re.compile(r"^l(\d+)_(x|y|z)$", re.IGNORECASE)
LANDMARK_AXES = ("x", "y", "z")
LANDMARK_COUNT = 21


def normalize_hand_slot(value, hand_column):
    """Normalize the hand key so it stays stable across rows and videos."""
    if pd.isna(value):
        return None

    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None

    if hand_column == "handedness":
        return text[:1].upper() + text[1:].lower()

    if hand_column == "hand_index":
        try:
            return str(int(float(text)))
        except ValueError:
            return text

    return text


def sort_hand_slots(values):
    """Sort hands placing Left/Right first and the rest later."""

    def key(slot):
        if slot == "Left":
            return (0, 0, slot)
        if slot == "Right":
            return (1, 0, slot)
        try:
            return (2, int(slot), slot)
        except (TypeError, ValueError):
            return (3, str(slot).lower(), str(slot))

    return sorted(values, key=key)


def detect_delimiter(csv_file, default=","):
    """Detect the real CSV delimiter without relying on a fixed constant."""
    with open(csv_file, "r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        if not sample:
            return default
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
            return dialect.delimiter
        except csv.Error:
            return default


def sort_feature_columns(columns):
    """Order feature columns by semantic structure."""
    kpca_cols = []
    landmark_cols = []

    for col in columns:
        match_kpca = KPCA_FEATURE_RE.match(col)
        if match_kpca:
            kpca_cols.append((int(match_kpca.group(1)), col))
            continue

        match_landmark = LANDMARK_FEATURE_RE.match(col)
        if match_landmark:
            landmark_idx = int(match_landmark.group(1))
            axis = match_landmark.group(2).lower()
            axis_order = LANDMARK_AXES.index(axis)
            landmark_cols.append((landmark_idx, axis_order, col))

    if kpca_cols:
        ordered = [col for _, col in sorted(kpca_cols, key=lambda item: item[0])]
        return ordered, "kpca"

    if landmark_cols:
        ordered = [col for _, _, col in sorted(landmark_cols, key=lambda item: (item[0], item[1]))]
        expected = [f"l{i}_{axis}" for i in range(LANDMARK_COUNT) for axis in LANDMARK_AXES]
        missing = [c for c in expected if c not in columns]
        if missing:
            raise ValueError(
                "Faltan columnas de landmarks requeridas: "
                f"{missing[:10]}{'...' if len(missing) > 10 else ''}"
            )
        return ordered, "landmarks"

    raise ValueError(
        "No se encontraron columnas de features. Se esperaban columnas kpca_* "
        "o landmarks l0_x...l20_z."
    )


def load_data(csv_file, delimiter=None, require_labels=True):
    """Load the CSV and return sequences and labels grouped by video."""
    if delimiter is None:
        delimiter = detect_delimiter(csv_file)
    print(f"DEBUG: Cargando {csv_file} con delimitador '{delimiter}'...", flush=True)

    if not os.path.exists(csv_file):
        sys.exit(f"ERROR: No existe {csv_file}")

    df = pd.read_csv(csv_file, delimiter=delimiter, encoding="utf-8-sig")
    df.columns = [str(c).strip() for c in df.columns]

    if "glosa" in df.columns:
        df["glosa"] = df["glosa"].astype(str).str.strip()
    elif require_labels:
        raise ValueError("Columna 'glosa' faltante para entrenamiento/evaluacion.")
    else:
        df["glosa"] = "__unknown__"

    if "video" in df.columns:
        df["video"] = df["video"].astype(str).str.strip()
    else:
        df["video"] = os.path.basename(csv_file)

    if "handedness" in df.columns:
        handedness = df["handedness"].astype("string").str.strip().str.title()
        df["handedness"] = handedness.where(~handedness.str.lower().isin(["", "nan", "none"]))

    for numeric_col in ["frame", "hand_index"]:
        if numeric_col not in df.columns:
            raise ValueError(f"Columna '{numeric_col}' faltante. Columnas: {list(df.columns)}")
        df[numeric_col] = pd.to_numeric(df[numeric_col], errors="coerce")
        if df[numeric_col].isna().any():
            raise ValueError(f"Columna '{numeric_col}' contiene valores no numericos.")
        df[numeric_col] = df[numeric_col].astype(int)

    print(f"DEBUG: {len(df)} filas, {df['video'].nunique()} videos unicos", flush=True)
    print(f"DEBUG: Glosas unicas: {df['glosa'].nunique()} - {df['glosa'].unique()}", flush=True)

    feature_cols, feature_mode = sort_feature_columns(df.columns)
    print(f"DEBUG: {len(feature_cols)} columnas de features ({feature_mode})", flush=True)

    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    if df[feature_cols].isna().any().any():
        raise ValueError("El CSV contiene valores vacios o no numericos en las columnas de features.")

    hand_column = "handedness" if "handedness" in df.columns and df["handedness"].notna().any() else "hand_index"
    if hand_column == "handedness":
        hand_order = PREFERRED_HAND_ORDER[:]
        extra_hands = [
            value
            for value in sort_hand_slots(
                [normalize_hand_slot(v, hand_column) for v in df[hand_column].dropna().unique()]
            )
            if value not in hand_order
        ]
        hand_order.extend(extra_hands)
    else:
        hand_order = sort_hand_slots(
            [normalize_hand_slot(v, hand_column) for v in df[hand_column].dropna().unique()]
        )

    hand_order = [value for value in hand_order if value is not None]
    if not hand_order:
        raise ValueError("No se detectaron manos validas en el CSV.")
    print(f"DEBUG: Manos detectadas ({hand_column}): {hand_order}", flush=True)

    sequences = {}
    labels = {}

    for video_id, group in df.groupby("video"):
        labels[video_id] = group["glosa"].iloc[0]

        hands = {}
        for raw_hand_value, hand_group in group.groupby(hand_column):
            hand_slot = normalize_hand_slot(raw_hand_value, hand_column)
            if hand_slot is None:
                continue
            hand_group = hand_group.sort_values("frame")
            seq = hand_group[feature_cols].values.astype(np.float32)
            hands[hand_slot] = seq

        sequences[video_id] = hands

        if len(sequences) <= 3:
            print(
                f"  VIDEO: {video_id} - frames: {len(group['frame'].unique())}, manos: {list(hands.keys())}",
                flush=True,
            )

    print(f"DEBUG: Total videos cargados: {len(sequences)}", flush=True)
    print(f"DEBUG: Total glosas: {len(set(labels.values()))}", flush=True)

    return sequences, labels, feature_cols, feature_mode, hand_column, hand_order


def extract_aggregated_features(hands_dict, hand_order):
    """Extract a statistic vector for SVM."""
    if not hands_dict:
        raise ValueError("Video sin manos")

    sample_hand = next(iter(hands_dict.values()))
    n_components = sample_hand.shape[1]
    n_funcs = len(AGGREGATION_FUNCTIONS)

    features = []
    for hand_slot in hand_order:
        if hand_slot in hands_dict:
            seq = hands_dict[hand_slot]
            for func in AGGREGATION_FUNCTIONS:
                features.extend(func(seq, axis=0))
        else:
            features.extend([0.0] * (n_funcs * n_components))
    return np.array(features, dtype=np.float32)


def prepare_svm_data(sequences, hand_order):
    """Convert all sequences into a matrix for SVM."""
    return np.array([extract_aggregated_features(seq, hand_order) for seq in sequences.values()])


class DTWClassifier:
    def __init__(self, hand_order=None):
        self.train_sequences = []
        self.train_labels = []
        self.le = None
        self.n_features = None
        self.hand_order = list(hand_order) if hand_order is not None else None

    def fit(self, sequences_dict, labels_dict, hand_order=None):
        if hand_order is not None:
            self.hand_order = list(hand_order)
        elif self.hand_order is None:
            inferred = []
            for hands in sequences_dict.values():
                inferred.extend(hands.keys())
            self.hand_order = sort_hand_slots(set(inferred))

        if not self.hand_order:
            raise ValueError("No se definio el orden de manos para DTW")

        self.le = LabelEncoder()
        self.train_labels = self.le.fit_transform(list(labels_dict.values()))
        video_ids = list(labels_dict.keys())
        self.train_sequences = [self._flatten_hands(sequences_dict[vid]) for vid in video_ids]
        self.n_features = self.train_sequences[0].shape[1]

    def _flatten_hands(self, hands_dict):
        """Combine the configured hands with zero padding if needed."""
        if not self.hand_order:
            raise ValueError("No se definio el orden de manos")

        present_sequences = [hands_dict[idx] for idx in self.hand_order if idx in hands_dict]
        if not present_sequences:
            raise ValueError("Video sin manos")

        n_feat = present_sequences[0].shape[1]
        min_len = min(seq.shape[0] for seq in present_sequences)

        flattened_hands = []
        for hand_slot in self.hand_order:
            seq = hands_dict.get(hand_slot)
            if seq is None:
                seq = np.zeros((min_len, n_feat), dtype=np.float32)
            else:
                seq = seq[:min_len]
            flattened_hands.append(seq)

        combined = np.hstack(flattened_hands)
        return combined.astype(np.float32)

    def predict(self, sequences_dict):
        preds = []
        for hands in sequences_dict.values():
            seq = self._flatten_hands(hands)
            dists = []
            for train_seq in self.train_sequences:
                try:
                    d, _ = fastdtw(seq, train_seq, dist=euclidean)
                    dists.append(d)
                except ValueError:
                    max_cols = max(seq.shape[1], train_seq.shape[1])
                    s1 = np.pad(seq, ((0, 0), (0, max_cols - seq.shape[1])), mode="constant")
                    s2 = np.pad(train_seq, ((0, 0), (0, max_cols - train_seq.shape[1])), mode="constant")
                    d, _ = fastdtw(s1, s2, dist=euclidean)
                    dists.append(d)
            preds.append(self.train_labels[np.argmin(dists)])
        return np.array(preds)


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def ensure_results_dir(results_dir):
    os.makedirs(results_dir, exist_ok=True)
    return results_dir


def get_model_path(output_dir, model_name):
    return os.path.join(output_dir, MODEL_FILENAMES[model_name])


def get_results_paths(results_dir, stem):
    ensure_results_dir(results_dir)
    return {
        "json": os.path.join(results_dir, f"{stem}.json"),
        "txt": os.path.join(results_dir, f"{stem}.txt"),
    }


def write_metrics_report(results_dir, stem, payload, text_lines):
    paths = get_results_paths(results_dir, stem)
    with open(paths["json"], "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(paths["txt"], "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines).rstrip() + "\n")
    return paths


def save_svm_artifact(model, label_encoder, feature_cols, feature_mode, hand_column, hand_order, output_dir):
    ensure_output_dir(output_dir)
    path = get_model_path(output_dir, "svm")
    joblib.dump(
        {
            "model": model,
            "le": label_encoder,
            "agg_funcs": AGGREGATION_FUNCTIONS,
            "feature_mode": feature_mode,
            "feature_cols": feature_cols,
            "n_features": len(feature_cols),
            "hand_column": hand_column,
            "hand_order": hand_order,
        },
        path,
    )
    return path


def save_dtw_artifact(model, hand_column, hand_order, output_dir):
    ensure_output_dir(output_dir)
    path = get_model_path(output_dir, "dtw")
    joblib.dump(
        {
            "model": model,
            "le": model.le,
            "hand_column": hand_column,
            "hand_order": hand_order,
        },
        path,
    )
    return path


def run_inference(model_name, input_csv, output_dir):
    model_path = get_model_path(output_dir, model_name)
    if not os.path.exists(model_path):
        sys.exit(f"ERROR: No existe modelo guardado en {model_path}")

    artifact = joblib.load(model_path)
    sequences, _, feature_cols, feature_mode, hand_column, _ = load_data(
        input_csv, delimiter=DELIMITER, require_labels=False
    )
    if not sequences:
        sys.exit("No hay secuencias en el CSV de entrada.")

    saved_feature_mode = artifact.get("feature_mode")
    saved_feature_cols = artifact.get("feature_cols")
    if saved_feature_cols and list(saved_feature_cols) != list(feature_cols):
        sys.exit(
            "ERROR: El CSV de entrada no coincide con el esquema usado para entrenar el modelo. "
            f"Esperado: {saved_feature_mode} ({len(saved_feature_cols)} features). "
            f"Recibido: {len(feature_cols)} features."
        )

    saved_hand_column = artifact.get("hand_column", hand_column)
    saved_hand_order = artifact.get("hand_order") or sort_hand_slots(
        {slot for hands in sequences.values() for slot in hands.keys()}
    )
    if saved_hand_column != hand_column:
        sys.exit(
            "ERROR: El CSV de entrada no coincide con la columna usada para agrupar las manos. "
            f"Esperado: {saved_hand_column}. Recibido: {hand_column}."
        )

    video_ids = list(sequences.keys())

    if model_name == "svm":
        X_input = prepare_svm_data(sequences, saved_hand_order)
        predictions = artifact["model"].predict(X_input)
        probabilities = None
        if hasattr(artifact["model"], "predict_proba"):
            probabilities = artifact["model"].predict_proba(X_input)
    else:
        predictions = artifact["model"].predict(sequences)
        probabilities = None

    predicted_labels = artifact["le"].inverse_transform(predictions)
    print()
    print("### PREDICCIONES ###")
    for idx, video_id in enumerate(video_ids):
        line = f"{video_id}: {predicted_labels[idx]}"
        if probabilities is not None:
            confidence = float(np.max(probabilities[idx]))
            line += f" (confidence={confidence:.4f})"
        print(line)


def train_svm(X_train, y_train):
    """Train SVM with a small hyperparameter search."""
    max_components = max(1, min(200, X_train.shape[1] - 1))
    candidate_components = sorted({c for c in [50, 100, 150, 200, max_components] if 1 <= c <= X_train.shape[1] - 1})
    if not candidate_components:
        candidate_components = [1]

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("kpca", KernelPCA(kernel="rbf", n_components=max_components)),
            ("svm", SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)),
        ]
    )
    param_grid = {
        "kpca__n_components": candidate_components,
        "kpca__gamma": ["scale", "auto", 0.01, 0.1],
        "svm__C": [0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.01, 0.1],
    }
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring="f1_macro", n_jobs=N_JOBS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid.fit(X_train, y_train)
    return grid.best_estimator_


def compare_models(sequences, labels_dict, feature_cols, feature_mode, hand_column, hand_order, output_dir, results_dir):
    """Compare SVM and DTW on the same split."""
    print()
    print("### COMPARACION SVM vs DTW ###")

    le = LabelEncoder()
    y_all = le.fit_transform(list(labels_dict.values()))
    video_ids = list(labels_dict.keys())

    print("Numero de secuencias:", len(sequences))
    print("IDs:", list(sequences.keys()))
    ids_train, ids_test = train_test_split(
        video_ids, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )
    print(f"DEBUG: Train={len(ids_train)} videos, Test={len(ids_test)} videos", flush=True)

    print("DEBUG: Extrayendo caracteristicas SVM...", flush=True)
    X_all = prepare_svm_data(sequences, hand_order)
    train_idx = [video_ids.index(v) for v in ids_train]
    test_idx = [video_ids.index(v) for v in ids_test]
    X_train_svm = X_all[train_idx]
    X_test_svm = X_all[test_idx]
    y_train_svm = le.transform([labels_dict[v] for v in ids_train])
    y_test_svm = le.transform([labels_dict[v] for v in ids_test])

    train_seq_dtw = {v: sequences[v] for v in ids_train}
    test_seq_dtw = {v: sequences[v] for v in ids_test}
    train_lab_dtw = {v: labels_dict[v] for v in ids_train}
    test_lab_dtw = [labels_dict[v] for v in ids_test]

    print("DEBUG: Entrenando SVM...", flush=True)
    svm_model = train_svm(X_train_svm, y_train_svm)
    svm_pred = svm_model.predict(X_test_svm)
    svm_acc = accuracy_score(y_test_svm, svm_pred)
    svm_f1 = f1_score(y_test_svm, svm_pred, average="macro")

    print("DEBUG: Entrenando DTW...", flush=True)
    dtw_model = DTWClassifier(hand_order=hand_order)
    dtw_model.fit(train_seq_dtw, train_lab_dtw, hand_order=hand_order)
    print("DEBUG: Prediciendo con DTW (puede tardar)...", flush=True)
    dtw_pred = dtw_model.predict(test_seq_dtw)
    dtw_acc = accuracy_score(le.transform(test_lab_dtw), dtw_pred)
    dtw_f1 = f1_score(le.transform(test_lab_dtw), dtw_pred, average="macro")

    print()
    print("=" * 60)
    print(" COMPARACION SVM vs DTW (same test set)")
    print("=" * 60)
    print(f"{'Metric':<20} {'SVM':>12} {'DTW':>12}")
    print("-" * 46)
    print(f"{'Accuracy':<20} {svm_acc:>12.4f} {dtw_acc:>12.4f}")
    print(f"{'F1 Macro':<20} {svm_f1:>12.4f} {dtw_f1:>12.4f}")
    print("=" * 60)

    print()
    print(">>> Report SVM:")
    label_ids = list(range(len(le.classes_)))
    print(classification_report(y_test_svm, svm_pred, labels=label_ids, target_names=le.classes_, zero_division=0))
    print("Confusion matrix SVM:")
    print(confusion_matrix(y_test_svm, svm_pred, labels=label_ids))

    print()
    print(">>> Report DTW:")
    label_ids = list(range(len(le.classes_)))
    print(classification_report(le.transform(test_lab_dtw), dtw_pred, labels=label_ids, target_names=le.classes_, zero_division=0))
    print("Confusion matrix DTW:")
    print(confusion_matrix(le.transform(test_lab_dtw), dtw_pred, labels=label_ids))

    svm_path = save_svm_artifact(svm_model, le, feature_cols, feature_mode, hand_column, hand_order, output_dir)
    dtw_path = save_dtw_artifact(dtw_model, hand_column, hand_order, output_dir)

    payload = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "mode": "compare",
        "feature_mode": feature_mode,
        "feature_count": len(feature_cols),
        "hand_column": hand_column,
        "hand_order": hand_order,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "train_videos": len(ids_train),
            "test_videos": len(ids_test),
        },
        "svm": {
            "accuracy": float(svm_acc),
            "f1_macro": float(svm_f1),
            "classification_report": classification_report(
                y_test_svm, svm_pred, labels=label_ids, target_names=le.classes_, zero_division=0, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(y_test_svm, svm_pred, labels=label_ids).tolist(),
        },
        "dtw": {
            "accuracy": float(dtw_acc),
            "f1_macro": float(dtw_f1),
            "classification_report": classification_report(
                le.transform(test_lab_dtw), dtw_pred, labels=label_ids, target_names=le.classes_, zero_division=0, output_dict=True
            ),
            "confusion_matrix": confusion_matrix(le.transform(test_lab_dtw), dtw_pred, labels=label_ids).tolist(),
        },
        "artifacts": {
            "svm": svm_path,
            "dtw": dtw_path,
        },
    }
    text_lines = [
        "COMPARACION SVM vs DTW",
        f"timestamp_utc: {payload['timestamp_utc']}",
        f"feature_mode: {feature_mode}",
        f"hand_column: {hand_column}",
        f"hand_order: {', '.join(hand_order)}",
        f"svm_accuracy: {svm_acc:.4f}",
        f"svm_f1_macro: {svm_f1:.4f}",
        f"dtw_accuracy: {dtw_acc:.4f}",
        f"dtw_f1_macro: {dtw_f1:.4f}",
        f"svm_artifact: {svm_path}",
        f"dtw_artifact: {dtw_path}",
    ]
    report_paths = write_metrics_report(results_dir, 'latest_compare', payload, text_lines)
    print()
    print(f"Modelos guardados: {svm_path}, {dtw_path}")
    print(f"Metricas guardadas: {report_paths['json']}, {report_paths['txt']}")


def main():
    parser = argparse.ArgumentParser(description="Clasificador de senas SVM/DTW")
    parser.add_argument("mode", nargs="?", default="compare", choices=["train", "evaluate", "infer", "compare"])
    parser.add_argument("--model", choices=["svm", "dtw"], required=False, default=None)
    parser.add_argument("--input", help="CSV para inferencia")
    parser.add_argument("--csv", default=CSV_FILE, help="CSV de datos")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directorio para guardar/cargar modelos")
    parser.add_argument("--results-dir", default=DEFAULT_RESULTS_DIR, help="Directorio para guardar metricas")
    args = parser.parse_args()

    if args.mode in ["train", "evaluate", "infer"] and args.model is None:
        parser.error("--model es obligatorio para train/evaluate/infer")

    print(f"DEBUG: modo={args.mode}, modelo={args.model}, output_dir={args.output_dir}", flush=True)

    if args.mode == "infer":
        if not args.input:
            parser.error("--input es obligatorio para infer")
        run_inference(args.model, args.input, args.output_dir)
        return

    sequences, labels_dict, feature_cols, feature_mode, hand_column, hand_order = load_data(args.csv, delimiter=DELIMITER)
    if not sequences:
        sys.exit("No hay datos. Revisa el CSV.")

    if args.mode == "compare":
        compare_models(sequences, labels_dict, feature_cols, feature_mode, hand_column, hand_order, args.output_dir, args.results_dir)
        return

    le = LabelEncoder()
    y_all = le.fit_transform(list(labels_dict.values()))
    video_ids = list(labels_dict.keys())
    ids_train, ids_test = train_test_split(
        video_ids, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )

    if args.model == "svm":
        X_all = prepare_svm_data(sequences, hand_order)
        train_idx = [video_ids.index(v) for v in ids_train]
        test_idx = [video_ids.index(v) for v in ids_test]
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
        y_train = le.transform([labels_dict[v] for v in ids_train])
        y_test = le.transform([labels_dict[v] for v in ids_test])

        if args.mode == "train":
            model = train_svm(X_train, y_train)
            path = save_svm_artifact(model, le, feature_cols, feature_mode, hand_column, hand_order, args.output_dir)
            payload = {
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "mode": "train",
                "model": "svm",
                "feature_mode": feature_mode,
                "feature_count": len(feature_cols),
                "hand_column": hand_column,
                "hand_order": hand_order,
                "artifacts": {"svm": path},
            }
            report_paths = write_metrics_report(
                args.results_dir,
                'latest_train_svm',
                payload,
                [
                    'ENTRENAMIENTO SVM',
                    f"timestamp_utc: {payload['timestamp_utc']}",
                    f"feature_mode: {feature_mode}",
                    f"hand_column: {hand_column}",
                    f"hand_order: {', '.join(hand_order)}",
                    f"svm_artifact: {path}",
                ],
            )
            print(f"Modelo SVM guardado en {path}")
            print(f"Metricas guardadas: {report_paths['json']}, {report_paths['txt']}")

        elif args.mode == "evaluate":
            model = train_svm(X_train, y_train)
            pred = model.predict(X_test)
            acc = accuracy_score(y_test, pred)
            f1 = f1_score(y_test, pred, average='macro')
            label_ids = list(range(len(le.classes_)))
            report = classification_report(y_test, pred, labels=label_ids, target_names=le.classes_, zero_division=0, output_dict=True)
            cm = confusion_matrix(y_test, pred, labels=label_ids)
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 macro: {f1:.4f}")
            print(classification_report(y_test, pred, labels=label_ids, target_names=le.classes_, zero_division=0))
            payload = {
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "mode": "evaluate",
                "model": "svm",
                "feature_mode": feature_mode,
                "feature_count": len(feature_cols),
                "hand_column": hand_column,
                "hand_order": hand_order,
                "split": {
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                    "train_videos": len(ids_train),
                    "test_videos": len(ids_test),
                },
                "metrics": {
                    "accuracy": float(acc),
                    "f1_macro": float(f1),
                    "classification_report": report,
                    "confusion_matrix": cm.tolist(),
                },
            }
            text_lines = [
                "EVALUACION SVM",
                f"timestamp_utc: {payload['timestamp_utc']}",
                f"feature_mode: {feature_mode}",
                f"hand_column: {hand_column}",
                f"hand_order: {', '.join(hand_order)}",
                f"accuracy: {acc:.4f}",
                f"f1_macro: {f1:.4f}",
            ]
            report_paths = write_metrics_report(args.results_dir, 'latest_evaluate_svm', payload, text_lines)
            print(f"Metricas guardadas: {report_paths['json']}, {report_paths['txt']}")

    elif args.model == "dtw":
        train_seq = {v: sequences[v] for v in ids_train}
        test_seq = {v: sequences[v] for v in ids_test}
        train_lab = {v: labels_dict[v] for v in ids_train}
        test_lab = [labels_dict[v] for v in ids_test]

        if args.mode == "train":
            dtw = DTWClassifier(hand_order=hand_order)
            dtw.fit(train_seq, train_lab, hand_order=hand_order)
            path = save_dtw_artifact(dtw, hand_column, hand_order, args.output_dir)
            payload = {
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "mode": "train",
                "model": "dtw",
                "feature_mode": feature_mode,
                "feature_count": len(feature_cols),
                "hand_column": hand_column,
                "hand_order": hand_order,
                "artifacts": {"dtw": path},
            }
            report_paths = write_metrics_report(
                args.results_dir,
                'latest_train_dtw',
                payload,
                [
                    'ENTRENAMIENTO DTW',
                    f"timestamp_utc: {payload['timestamp_utc']}",
                    f"feature_mode: {feature_mode}",
                    f"hand_column: {hand_column}",
                    f"hand_order: {', '.join(hand_order)}",
                    f"dtw_artifact: {path}",
                ],
            )
            print(f"Modelo DTW guardado en {path}")
            print(f"Metricas guardadas: {report_paths['json']}, {report_paths['txt']}")

        elif args.mode == "evaluate":
            dtw = DTWClassifier(hand_order=hand_order)
            dtw.fit(train_seq, train_lab, hand_order=hand_order)
            pred = dtw.predict(test_seq)
            y_true = le.transform(test_lab)
            acc = accuracy_score(y_true, pred)
            f1 = f1_score(y_true, pred, average='macro')
            label_ids = list(range(len(le.classes_)))
            report = classification_report(y_true, pred, labels=label_ids, target_names=le.classes_, zero_division=0, output_dict=True)
            cm = confusion_matrix(y_true, pred, labels=label_ids)
            print(f"Accuracy: {acc:.4f}")
            print(f"F1 macro: {f1:.4f}")
            print(classification_report(y_true, pred, labels=label_ids, target_names=le.classes_, zero_division=0))
            payload = {
                "timestamp_utc": datetime.utcnow().isoformat() + "Z",
                "mode": "evaluate",
                "model": "dtw",
                "feature_mode": feature_mode,
                "feature_count": len(feature_cols),
                "hand_column": hand_column,
                "hand_order": hand_order,
                "split": {
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                    "train_videos": len(ids_train),
                    "test_videos": len(ids_test),
                },
                "metrics": {
                    "accuracy": float(acc),
                    "f1_macro": float(f1),
                    "classification_report": report,
                    "confusion_matrix": cm.tolist(),
                },
            }
            text_lines = [
                "EVALUACION DTW",
                f"timestamp_utc: {payload['timestamp_utc']}",
                f"feature_mode: {feature_mode}",
                f"hand_column: {hand_column}",
                f"hand_order: {', '.join(hand_order)}",
                f"accuracy: {acc:.4f}",
                f"f1_macro: {f1:.4f}",
            ]
            report_paths = write_metrics_report(args.results_dir, 'latest_evaluate_dtw', payload, text_lines)
            print(f"Metricas guardadas: {report_paths['json']}, {report_paths['txt']}")


if __name__ == "__main__":
    main()
