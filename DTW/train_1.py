
"""
compare (SVM vs DTW)
"""
import numpy as np
import pandas as pd
import os
import sys
import argparse
import warnings
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import KernelPCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import joblib

# ===================== CONFIGURACIÓN =====================
CSV_FILE = "datos_30.csv"     
DELIMITER = ','                  # Separador 
TEST_SIZE = 0.2                   # 20% para test
RANDOM_STATE = 42
N_JOBS = -1                       # Usar todos los núcleos


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

def load_data(csv_file, delimiter=','):
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
    if 'video' in df.columns:
        df['video'] = df['video'].str.strip()
    
    print(f"DEBUG: {len(df)} filas, {df['video'].nunique()} videos únicos", flush=True)
    print(f"DEBUG: Glosas únicas: {df['glosa'].nunique()} - {df['glosa'].unique()}", flush=True)
    
    # Verificar columnas
    required_cols = ['glosa', 'video', 'frame', 'hand_index', 'handedness']
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
            print(f"  VIDEO: {video_id} - frames: {len(hand_group)}, manos: {list(hands.keys())}", flush=True)
    
    print(f"DEBUG: Total videos cargados: {len(sequences)}", flush=True)
    print(f"DEBUG: Total glosas: {len(set(labels.values()))}", flush=True)
    
    return sequences, labels, kpca_cols

def extract_aggregated_features(hands_dict):
    """Extrae vector de estadísticos para SVM."""
    sample_hand = next(iter(hands_dict.values()))
    n_components = sample_hand.shape[1]
    n_funcs = len(AGGREGATION_FUNCTIONS)
    
    features = []
    for hand_idx in [0, 1]:  # Orden fijo
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
        """Combina ambas manos con padding de ceros si es necesario."""
        # Obtener dimensionalidad de las manos presentes
        seq0 = hands_dict.get(0)
        seq1 = hands_dict.get(1)
        
        # Si no hay info de features aún, detectar
        if seq0 is not None:
            n_feat = seq0.shape[1]
            n_frames = seq0.shape[0]
        elif seq1 is not None:
            n_feat = seq1.shape[1]
            n_frames = seq1.shape[0]
        else:
            raise ValueError("Video sin manos")
        
        # Crear mano faltante con ceros
        if seq0 is None:
            seq0 = np.zeros((n_frames, n_feat), dtype=np.float32)
        if seq1 is None:
            seq1 = np.zeros((n_frames, n_feat), dtype=np.float32)
        
        # Igualar frames
        min_len = min(seq0.shape[0], seq1.shape[0])
        combined = np.hstack([seq0[:min_len], seq1[:min_len]])
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

def compare_models(sequences, labels_dict, kpca_cols):
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
    joblib.dump({
        'model': svm_model,
        'le': le,
        'agg_funcs': AGGREGATION_FUNCTIONS,
        'n_kpca': len(kpca_cols)
    }, 'model_svm.joblib')
    joblib.dump({
        'model': dtw_model,
        'le': le
    }, 'model_dtw.joblib')
    print("\n Modelos guardados: model_svm.joblib, model_dtw.joblib")

def main():
    parser = argparse.ArgumentParser(description="Clasificador de señas SVM/DTW")
    parser.add_argument('mode', choices=['train', 'evaluate', 'infer', 'compare'])
    parser.add_argument('--model', choices=['svm', 'dtw'], required=False, default=None)
    parser.add_argument('--input', help='CSV para inferencia')
    parser.add_argument('--csv', default=CSV_FILE, help='CSV de datos')
    args = parser.parse_args()
    
    # Validación
    if args.mode in ['train', 'evaluate', 'infer'] and args.model is None:
        parser.error("--model es obligatorio para train/evaluate/infer")
    
    print(f"DEBUG: modo={args.mode}, modelo={args.model}", flush=True)
    
    
    sequences, labels_dict, kpca_cols = load_data(args.csv, delimiter=None)
    if not sequences:
        sys.exit("No hay datos. Revisa el CSV.")
    
    # Modo comparación
    if args.mode == 'compare':
        compare_models(sequences, labels_dict, kpca_cols)
        return
    
    # Resto de modos (train, evaluate, infer)
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
            joblib.dump({
                'model': model, 'le': le,
                'agg_funcs': AGGREGATION_FUNCTIONS,
                'n_kpca': len(kpca_cols)
            }, 'model_svm.joblib')
            print(" Modelo SVM guardado")
        
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
            joblib.dump({'model': dtw, 'le': dtw.le}, 'model_dtw.joblib')
            print("Modelo DTW guardado")
        
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