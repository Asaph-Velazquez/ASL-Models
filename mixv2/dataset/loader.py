# dataset/loader.py
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignDataLoader:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.data_path = self.config['data']['raw_path']
        self.max_seq_len = self.config['data']['max_sequence_length']
        self.random_state = self.config['data']['random_state']

        self.label_encoder = LabelEncoder()
        self.data = None
        self.feature_cols = None

    def _infer_feature_columns(self, df):
        """Detecta columnas de features numericas ignorando metadatos."""
        metadata_cols = {'glosa', 'video', 'frame', 'hand_index', 'handedness'}
        candidate_cols = [c for c in df.columns if c not in metadata_cols]

        if not candidate_cols:
            raise ValueError("No se encontraron columnas de features en el CSV.")

        feature_df = df[candidate_cols].apply(pd.to_numeric, errors='coerce')
        if feature_df.isna().any().any():
            bad_cols = feature_df.columns[feature_df.isna().any()].tolist()
            raise ValueError(
                "Hay columnas de features con valores no numericos o vacios: "
                f"{bad_cols}"
            )

        return candidate_cols

    def load_and_preprocess(self):
        """Carga y preprocesa los datos."""
        logger.info(f"Cargando datos desde {self.data_path}")
        df = pd.read_csv(self.data_path)

        required_cols = {'glosa', 'video', 'frame'}
        missing_required = required_cols - set(df.columns)
        if missing_required:
            raise ValueError(
                f"El CSV no contiene las columnas requeridas: {sorted(missing_required)}"
            )

        self.feature_cols = self._infer_feature_columns(df)

        if 'hand_index' in df.columns:
            df['video_id'] = df['video'].astype(str) + '_' + df['hand_index'].astype(str)
        else:
            df['video_id'] = df['video'].astype(str)

        df['group_id'] = df['video'].astype(str)
        df['label_encoded'] = self.label_encoder.fit_transform(df['glosa'])
        self.data = df

        logger.info(
            f"Datos cargados: {len(df)} frames, {len(self.label_encoder.classes_)} clases"
        )
        logger.info(f"Features detectadas: {len(self.feature_cols)}")

        return df

    def create_sequences(self, df=None):
        """Crea secuencias temporales por video."""
        if df is None:
            df = self.data

        if df is None:
            raise ValueError("Primero debes ejecutar load_and_preprocess().")

        if self.feature_cols is None:
            self.feature_cols = self._infer_feature_columns(df)

        sequences = []
        labels = []
        sample_ids = []
        group_ids = []

        for video_id, group in df.groupby('video_id'):
            sort_cols = ['frame']
            if 'hand_index' in group.columns:
                sort_cols.append('hand_index')
            group = group.sort_values(sort_cols)

            features = group[self.feature_cols].to_numpy(dtype=np.float32)
            label = group['label_encoded'].iloc[0]
            group_id = group['group_id'].iloc[0]

            sequences.append(features)
            labels.append(label)
            sample_ids.append(video_id)
            group_ids.append(group_id)

        sequences_padded = pad_sequences(
            sequences,
            maxlen=self.max_seq_len,
            padding='post',
            dtype='float32',
            value=0.0,
        )

        logger.info(f"Secuencias creadas: {len(sequences_padded)} videos")

        return np.array(sequences_padded), np.array(labels), sample_ids, group_ids

    def split_data(self, X, y, video_ids=None, group_ids=None):
        """Divide datos en train/val/test evitando fuga entre manos del mismo video."""
        test_size = self.config['data']['test_size']
        val_size = self.config['data']['val_size']

        if group_ids is None:
            group_ids = video_ids

        if group_ids is None:
            raise ValueError("Se requieren identificadores de grupo para dividir los datos.")

        group_ids = np.asarray(group_ids)
        y = np.asarray(y)

        group_to_label = {}
        for idx, gid in enumerate(group_ids):
            if gid not in group_to_label:
                group_to_label[gid] = y[idx]

        unique_groups = np.array(list(group_to_label.keys()))
        unique_labels = np.array([group_to_label[g] for g in unique_groups])

        groups_train_val, groups_test = train_test_split(
            unique_groups,
            test_size=test_size,
            stratify=unique_labels,
            random_state=self.random_state,
        )

        train_val_mask = np.isin(group_ids, groups_train_val)
        test_mask = np.isin(group_ids, groups_test)

        X_train_val = X[train_val_mask]
        y_train_val = y[train_val_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        groups_train_val = group_ids[train_val_mask]

        val_ratio = val_size / (1 - test_size)
        group_to_label_tv = {}
        for idx, gid in enumerate(groups_train_val):
            if gid not in group_to_label_tv:
                group_to_label_tv[gid] = y_train_val[idx]

        unique_groups_tv = np.array(list(group_to_label_tv.keys()))
        unique_labels_tv = np.array([group_to_label_tv[g] for g in unique_groups_tv])

        groups_train, groups_val = train_test_split(
            unique_groups_tv,
            test_size=val_ratio,
            stratify=unique_labels_tv,
            random_state=self.random_state,
        )

        train_mask = np.isin(group_ids, groups_train)
        val_mask = np.isin(group_ids, groups_val)

        X_train = X[train_mask]
        y_train = y[train_mask]
        X_val = X[val_mask]
        y_val = y[val_mask]

        logger.info("Division completada:")
        logger.info(f"  Train: {len(X_train)} muestras")
        logger.info(f"  Val: {len(X_val)} muestras")
        logger.info(f"  Test: {len(X_test)} muestras")

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
        }

    def save_processed_data(self, data_split):
        """Guarda datos procesados."""
        save_path = Path(self.config['data']['processed_path'])
        save_path.mkdir(parents=True, exist_ok=True)

        for key, value in data_split.items():
            np.save(save_path / f"{key}.npy", value)

        joblib.dump(self.label_encoder, save_path / 'label_encoder.pkl')

        logger.info(f"Datos procesados guardados en {save_path}")
