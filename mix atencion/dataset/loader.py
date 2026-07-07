# dataset/loader.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import joblib
import yaml
import logging
import re
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

KPCA_FEATURE_RE = re.compile(r"^kpca_(\d+)$", re.IGNORECASE)
LANDMARK_FEATURE_RE = re.compile(r"^l(\d+)_(x|y|z)$", re.IGNORECASE)
LANDMARK_AXES = ("x", "y", "z")
LANDMARK_COUNT = 21

class SignDataLoader:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_path = self.config['data']['raw_path']
        self.max_seq_len = self.config['data']['max_sequence_length']
        self.random_state = self.config['data']['random_state']
        
        self.label_encoder = LabelEncoder()
        self.data = None
        
    def load_and_preprocess(self):
        """Carga y preprocesa los datos"""
        logger.info(f"Cargando datos desde {self.data_path}")
        df = pd.read_csv(self.data_path)
        df.columns = [str(c).strip() for c in df.columns]
        
        # Crear identificador único por video
        df['video_id'] = df['video'] + '_' + df['hand_index'].astype(str)
        
        # Codificar glosas
        df['label_encoded'] = self.label_encoder.fit_transform(df['glosa'])

        self.feature_cols = self._detect_feature_columns(df.columns)
        self.feature_dim = len(self.feature_cols)
        
        self.data = df
        logger.info(
            f"Datos cargados: {len(df)} frames, {len(self.label_encoder.classes_)} clases, "
            f"{self.feature_dim} features"
        )
        
        return df

    def _detect_feature_columns(self, columns):
        kpca_cols = []
        landmark_cols = []

        for col in columns:
            kpca_match = KPCA_FEATURE_RE.match(col)
            if kpca_match:
                kpca_cols.append((int(kpca_match.group(1)), col))
                continue

            landmark_match = LANDMARK_FEATURE_RE.match(col)
            if landmark_match:
                idx = int(landmark_match.group(1))
                axis = landmark_match.group(2).lower()
                axis_order = LANDMARK_AXES.index(axis)
                landmark_cols.append((idx, axis_order, col))

        if kpca_cols:
            return [col for _, col in sorted(kpca_cols, key=lambda item: item[0])]

        if landmark_cols:
            expected = [f"l{i}_{axis}" for i in range(LANDMARK_COUNT) for axis in LANDMARK_AXES]
            missing = [c for c in expected if c not in columns]
            if missing:
                raise ValueError(
                    f"Faltan columnas de landmarks requeridas: {missing[:10]}"
                    f"{'...' if len(missing) > 10 else ''}"
                )
            return [col for _, _, col in sorted(landmark_cols, key=lambda item: (item[0], item[1]))]

        raise ValueError(
            "No se encontraron columnas de features. Se esperaban columnas kpca_* "
            "o landmarks l0_x...l20_z."
        )
        
    def create_sequences(self, df=None):
        """Crea secuencias temporales por video"""
        if df is None:
            df = self.data
            
        sequences = []
        labels = []
        video_ids = []
        
        if not hasattr(self, "feature_cols") or self.feature_cols is None:
            self.feature_cols = self._detect_feature_columns(df.columns)

        for video_id, group in df.groupby('video_id'):
            group = group.sort_values('frame')
            features = group[self.feature_cols].values
            label = group['label_encoded'].iloc[0]
            
            sequences.append(features)
            labels.append(label)
            video_ids.append(video_id)
        
        # Padding
        sequences_padded = pad_sequences(
            sequences,
            maxlen=self.max_seq_len,
            padding='post',
            dtype='float32',
            value=0.0
        )
        
        logger.info(f"Secuencias creadas: {len(sequences_padded)} videos")
        
        return np.array(sequences_padded), np.array(labels), video_ids, self.feature_dim
    
    def split_data(self, X, y, video_ids=None):
        """Divide datos en train/val/test"""
        test_size = self.config['data']['test_size']
        val_size = self.config['data']['val_size']
        
        #test
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=test_size, 
            stratify=y, random_state=self.random_state
        )
        
        # train
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, 
            test_size=val_ratio,
            stratify=y_train_val, 
            random_state=self.random_state
        )
        
        logger.info(f"División completada:")
        logger.info(f"  Train: {len(X_train)} muestras")
        logger.info(f"  Val: {len(X_val)} muestras")
        logger.info(f"  Test: {len(X_test)} muestras")
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test
        }
    
    def save_processed_data(self, data_split):
        """Guarda datos procesados"""
        save_path = Path(self.config['data']['processed_path'])
        save_path.mkdir(parents=True, exist_ok=True)
        
        for key, value in data_split.items():
            np.save(save_path / f"{key}.npy", value)
        
        # Guardar label encoder
        joblib.dump(self.label_encoder, save_path / 'label_encoder.pkl')
        
        logger.info(f"Datos procesados guardados en {save_path}")
