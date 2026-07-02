
"""
compare (SVM vs DTW)
"""
import argparse
import os
import sys
import warnings

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

# ===================== CONFIGURACIÓN =====================
CSV_FILE = "datos_30.csv"     
DELIMITER = ','                  # Separador 
TEST_SIZE = 0.2                   # 20% para test
RANDOM_STATE = 42
N_JOBS = -1                       # Usar todos los núcleos
DEFAULT_OUTPUT_DIR = "models"
HAND_INDICES = [0, 1, 2]
MODEL_FILENAMES = {
    'svm': 'model_svm.joblib',
    'dtw': 'model_dtw.joblib',
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

def load_data(csv_file, delimiter=',', require_labels=True):
    """Carga el CSV y devuelve diccionarios de secuencias y etiquetas."""
    print(f"DEBUG: Cargando {csv_file} con delimitador '{delimiter}'...", flush=True)
    
    if not os.path.exists(csv_file):
        sys.exit(f"ERROR: No existe {csv_file}")
    
    # Cargar CSV
    df = pd.read_csv(csv_file, delimiter=delimiter)
    
    # Limpiar nombres de columnas y valores
    df.columns = df.columns.str.strip()
    if 'glosa' in df.columns:
        df['glosa'] = df['glosa'].str.strip()
    elif require_labels:
        raise ValueError("Columna 'glosa' faltante para entrenamiento/evaluación.")
    else:
        df['glosa'] = '__unknown__'
    if 'video' in df.columns:
        df['video'] = df['video'].str.strip()
    else:
        df['video'] = os.path.basename(csv_file)
    
    print(f"DEBUG: {len(df)} filas, {df['video'].nunique()} videos únicos", flush=True)
    print(f"DEBUG: Glosas únicas: {df['glosa'].nunique()} - {df['glosa'].unique()}", flush=True)
    
    # Verificar columnas
    required_cols = ['glosa', 'video', 'frame', 'hand_index']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Columna '{col}' faltante. Columnas: {list(df.columns)}")
    
    # Identificar columnas KPCA
    kpca_cols = [c for c in df.columns if c.startswith('kpca_')]
    print(f"DEBUG: {len(kpca_cols)} columnas kpca", flush=True)
    
    # ⚡ CORRECCIÓN: Inicializar diccionarios FUERA del bucle
    sequences = {}
    labels = {}
    
    # Agrupar por video
    for video_id, group in df.groupby('video'):
        # Guardar etiqueta (toma la primera, todas son iguales para un mismo video)
        labels[video_id] = group['glosa'].iloc[0]
        
        # Procesar manos
        hands = {}
        for hand_idx, hand_group in group.groupby('hand_index'):
            hand_group = hand_group.sort_values('frame')
            seq = hand_group[kpca_cols].values.astype(np.float32)
            hands[hand_idx] = seq
        
        sequences[video_id] = hands
        
        # Debug: imprimir solo los primeros 3 videos
        if len(sequences) <= 3:
            print(
                f"  VIDEO: {video_id} - frames: {len(group['frame'].unique())}, manos: {list(hands.keys())}",
                flush=True,
            )
    
    print(f"DEBUG: Total videos cargados: {len(sequences)}", flush=True)
    print(f"DEBUG: Total glosas: {len(set(labels.values()))}", flush=True)
    
    return sequences, labels, kpca_cols

def extract_aggregated_features(hands_dict):
    """Extrae vector de estadísticos para SVM."""
    sample_hand = next(iter(hands_dict.values()))
    n_components = sample_hand.shape[1]
    n_funcs = len(AGGREGATION_FUNCTIONS)
    
    features = []
    for hand_idx in HAND_INDICES:
        if hand_idx in hands_dict:
            seq = hands_dict[hand_idx]
            for func in AGGREGATION_FUNCTIONS:
                features.extend(func(seq, axis=0))
        else:
            features.extend([0.0] * (n_funcs * n_components))
    return np.array(features, dtype=np.float32)

def prepare_svm_data(sequences):
    """Convierte todas las secuencias en matriz para SVM."""
    return np.array([extract_aggregated_features(seq) for seq in sequences.values()])

class DTWClassifier:
    def __init__(self):
        self.train_sequences = []
        self.train_labels = []
        self.le = None
        self.n_features = None  # Se establecerá en fit
    
    def fit(self, sequences_dict, labels_dict):
        self.le = LabelEncoder()
        self.train_labels = self.le.fit_transform(list(labels_dict.values()))
        video_ids = list(labels_dict.keys())
        self.train_sequences = [self._flatten_hands(sequences_dict[vid]) for vid in video_ids]
        # Guardar dimensionalidad para inferencia
        self.n_features = self.train_sequences[0].shape[1]
    
    def _flatten_hands(self, hands_dict):
        """Combina las manos configuradas con padding de ceros si es necesario."""
        present_sequences = [hands_dict[idx] for idx in HAND_INDICES if idx in hands_dict]
        if not present_sequences:
            raise ValueError("Video sin manos")

        n_feat = present_sequences[0].shape[1]
        min_len = min(seq.shape[0] for seq in present_sequences)

        flattened_hands = []
        for hand_idx in HAND_INDICES:
            seq = hands_dict.get(hand_idx)
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
                    # Si aún hay diferencia, rellenar con ceros
                    max_cols = max(seq.shape[1], train_seq.shape[1])
                    s1 = np.pad(seq, ((0,0),(0,max_cols-seq.shape[1])), mode='constant')
                    s2 = np.pad(train_seq, ((0,0),(0,max_cols-train_seq.shape[1])), mode='constant')
                    d, _ = fastdtw(s1, s2, dist=euclidean)
                    dists.append(d)
            preds.append(self.train_labels[np.argmin(dists)])
        return np.array(preds)


def ensure_output_dir(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_model_path(output_dir, model_name):
    return os.path.join(output_dir, MODEL_FILENAMES[model_name])


def save_svm_artifact(model, label_encoder, kpca_cols, output_dir):
    ensure_output_dir(output_dir)
    path = get_model_path(output_dir, 'svm')
    joblib.dump({
        'model': model,
        'le': label_encoder,
        'agg_funcs': AGGREGATION_FUNCTIONS,
        'n_kpca': len(kpca_cols),
        'hand_indices': HAND_INDICES,
    }, path)
    return path


def save_dtw_artifact(model, output_dir):
    ensure_output_dir(output_dir)
    path = get_model_path(output_dir, 'dtw')
    joblib.dump({
        'model': model,
        'le': model.le,
        'hand_indices': HAND_INDICES,
    }, path)
    return path


def run_inference(model_name, input_csv, output_dir):
    model_path = get_model_path(output_dir, model_name)
    if not os.path.exists(model_path):
        sys.exit(f"ERROR: No existe modelo guardado en {model_path}")

    artifact = joblib.load(model_path)
    sequences, _, _ = load_data(input_csv, delimiter=DELIMITER, require_labels=False)
    if not sequences:
        sys.exit("No hay secuencias en el CSV de entrada.")

    video_ids = list(sequences.keys())

    if model_name == 'svm':
        X_input = prepare_svm_data(sequences)
        predictions = artifact['model'].predict(X_input)
        probabilities = None
        if hasattr(artifact['model'], 'predict_proba'):
            probabilities = artifact['model'].predict_proba(X_input)
    else:
        predictions = artifact['model'].predict(sequences)
        probabilities = None

    predicted_labels = artifact['le'].inverse_transform(predictions)
    print("\n### PREDICCIONES ###")
    for idx, video_id in enumerate(video_ids):
        line = f"{video_id}: {predicted_labels[idx]}"
        if probabilities is not None:
            confidence = float(np.max(probabilities[idx]))
            line += f" (confidence={confidence:.4f})"
        print(line)


def train_svm(X_train, y_train):
    """Entrena SVM con búsqueda de hiperparámetros."""
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('kpca', KernelPCA(kernel='rbf', n_components=min(200, X_train.shape[1]-1))),
        ('svm', SVC(kernel='rbf', probability=True, random_state=RANDOM_STATE))
    ])
    param_grid = {
        'kpca__n_components': [50, 100, 150, 200],
        'kpca__gamma': ['scale', 'auto', 0.01, 0.1],
        'svm__C': [0.1, 1, 10, 100],
        'svm__gamma': ['scale', 'auto', 0.01, 0.1]
    }
    grid = GridSearchCV(pipe, param_grid, cv=3, scoring='f1_macro', n_jobs=N_JOBS)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid.fit(X_train, y_train)
    return grid.best_estimator_

def compare_models(sequences, labels_dict, kpca_cols, output_dir):
    """Compara SVM y DTW con el mismo split de datos."""
    print("\n### COMPARACIÓN SVM vs DTW ###", flush=True)
    
    le = LabelEncoder()
    y_all = le.fit_transform(list(labels_dict.values()))
    video_ids = list(labels_dict.keys())
    
    # División única
    print("Número de secuencias:", len(sequences))
    print("IDs:", list(sequences.keys()))
    ids_train, ids_test = train_test_split(
        video_ids, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )
    print(f"DEBUG: Train={len(ids_train)} videos, Test={len(ids_test)} videos", flush=True)
    
    # Preparar SVM
    print("DEBUG: Extrayendo características SVM...", flush=True)
    X_all = prepare_svm_data(sequences)
    train_idx = [video_ids.index(v) for v in ids_train]
    test_idx = [video_ids.index(v) for v in ids_test]
    X_train_svm = X_all[train_idx]
    X_test_svm = X_all[test_idx]
    y_train_svm = le.transform([labels_dict[v] for v in ids_train])
    y_test_svm = le.transform([labels_dict[v] for v in ids_test])
    
    # Preparar DTW
    train_seq_dtw = {v: sequences[v] for v in ids_train}
    test_seq_dtw = {v: sequences[v] for v in ids_test}
    train_lab_dtw = {v: labels_dict[v] for v in ids_train}
    test_lab_dtw = [labels_dict[v] for v in ids_test]
    
    # Entrenar SVM
    print("DEBUG: Entrenando SVM...", flush=True)
    svm_model = train_svm(X_train_svm, y_train_svm)
    svm_pred = svm_model.predict(X_test_svm)
    svm_acc = accuracy_score(y_test_svm, svm_pred)
    svm_f1 = f1_score(y_test_svm, svm_pred, average='macro')
    
    # Entrenar DTW
    print("DEBUG: Entrenando DTW...", flush=True)
    dtw_model = DTWClassifier()
    dtw_model.fit(train_seq_dtw, train_lab_dtw)
    print("DEBUG: Prediciendo con DTW (puede tardar)...", flush=True)
    dtw_pred = dtw_model.predict(test_seq_dtw)
    dtw_acc = accuracy_score(le.transform(test_lab_dtw), dtw_pred)
    dtw_f1 = f1_score(le.transform(test_lab_dtw), dtw_pred, average='macro')
    
    
    print("\n" + "="*60)
    print(" COMPARACIÓN SVM vs DTW (mismo test set)")
    print("="*60)
    print(f"{'Métrica':<20} {'SVM':<15} {'DTW':<15}")
    print("-"*50)
    print(f"{'Accuracy':<20} {svm_acc:<15.4f} {dtw_acc:<15.4f}")
    print(f"{'F1 Macro':<20} {svm_f1:<15.4f} {dtw_f1:<15.4f}")
    print("="*60)
    
    print("\n>>> Informe SVM:")
    print(classification_report(y_test_svm, svm_pred, target_names=le.classes_, zero_division=0))
    print("Matriz de confusión SVM:")
    print(confusion_matrix(y_test_svm, svm_pred))
    
    print("\n>>> Informe DTW:")
    print(classification_report(le.transform(test_lab_dtw), dtw_pred, 
                               target_names=le.classes_, zero_division=0))
    print("Matriz de confusión DTW:")
    print(confusion_matrix(le.transform(test_lab_dtw), dtw_pred))
    
    # Guardar modelos
    svm_path = save_svm_artifact(svm_model, le, kpca_cols, output_dir)
    dtw_path = save_dtw_artifact(dtw_model, output_dir)
    print(f"\nModelos guardados: {svm_path}, {dtw_path}")

def main():
    parser = argparse.ArgumentParser(description="Clasificador de señas SVM/DTW")
    parser.add_argument('mode', nargs='?', default='compare', choices=['train', 'evaluate', 'infer', 'compare'])
    parser.add_argument('--model', choices=['svm', 'dtw'], required=False, default=None)
    parser.add_argument('--input', help='CSV para inferencia')
    parser.add_argument('--csv', default=CSV_FILE, help='CSV de datos')
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR, help='Directorio para guardar/cargar modelos')
    args = parser.parse_args()
    
    # Validación
    if args.mode in ['train', 'evaluate', 'infer'] and args.model is None:
        parser.error("--model es obligatorio para train/evaluate/infer")
    
    print(f"DEBUG: modo={args.mode}, modelo={args.model}, output_dir={args.output_dir}", flush=True)

    if args.mode == 'infer':
        if not args.input:
            parser.error("--input es obligatorio para infer")
        run_inference(args.model, args.input, args.output_dir)
        return

    sequences, labels_dict, kpca_cols = load_data(args.csv, delimiter=DELIMITER)
    if not sequences:
        sys.exit("No hay datos. Revisa el CSV.")

    if args.mode == 'compare':
        compare_models(sequences, labels_dict, kpca_cols, args.output_dir)
        return

    le = LabelEncoder()
    y_all = le.fit_transform(list(labels_dict.values()))
    video_ids = list(labels_dict.keys())
    ids_train, ids_test = train_test_split(
        video_ids, test_size=TEST_SIZE, stratify=y_all, random_state=RANDOM_STATE
    )
    
    if args.model == 'svm':
        X_all = prepare_svm_data(sequences)
        train_idx = [video_ids.index(v) for v in ids_train]
        test_idx = [video_ids.index(v) for v in ids_test]
        X_train = X_all[train_idx]
        X_test = X_all[test_idx]
        y_train = le.transform([labels_dict[v] for v in ids_train])
        y_test = le.transform([labels_dict[v] for v in ids_test])
        
        if args.mode == 'train':
            model = train_svm(X_train, y_train)
            path = save_svm_artifact(model, le, kpca_cols, args.output_dir)
            print(f"Modelo SVM guardado en {path}")
        
        elif args.mode == 'evaluate':
            model = train_svm(X_train, y_train)
            pred = model.predict(X_test)
            print(f"Accuracy: {accuracy_score(y_test, pred):.4f}")
            print(f"F1 macro: {f1_score(y_test, pred, average='macro'):.4f}")
            print(classification_report(y_test, pred, target_names=le.classes_))
    
    elif args.model == 'dtw':
        train_seq = {v: sequences[v] for v in ids_train}
        test_seq = {v: sequences[v] for v in ids_test}
        train_lab = {v: labels_dict[v] for v in ids_train}
        test_lab = [labels_dict[v] for v in ids_test]
        
        if args.mode == 'train':
            dtw = DTWClassifier()
            dtw.fit(train_seq, train_lab)
            path = save_dtw_artifact(dtw, args.output_dir)
            print(f"Modelo DTW guardado en {path}")
        
        elif args.mode == 'evaluate':
            dtw = DTWClassifier()
            dtw.fit(train_seq, train_lab)
            pred = dtw.predict(test_seq)
            y_true = le.transform(test_lab)
            print(f"Accuracy: {accuracy_score(y_true, pred):.4f}")
            print(f"F1 macro: {f1_score(y_true, pred, average='macro'):.4f}")
            print(classification_report(y_true, pred, target_names=le.classes_))

if __name__ == '__main__':
    main()
