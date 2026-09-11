import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import random


class KPCA_Dataset(Dataset):
    def __init__(self, landmarks, labels, max_seq_len=100, augment=False):
        self.landmarks = landmarks
        self.labels = labels
        self.max_seq_len = max_seq_len
        self.augment = augment

        if len(landmarks) == 0:
            raise ValueError("No se recibieron secuencias para construir el dataset.")

        # Estadisticas globales sobre todas las secuencias
        all_data = np.concatenate([seq for seq in landmarks], axis=0)
        self.mean = np.mean(all_data, axis=0)
        self.std = np.std(all_data, axis=0)
        self.std[self.std < 1e-6] = 1.0

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        landmarks = self.landmarks[idx].copy()
        label = self.labels[idx]

        seq_len = landmarks.shape[0]

        # Truncar o padding
        if seq_len > self.max_seq_len:
            if self.augment:
                start = random.randint(0, seq_len - self.max_seq_len)
                landmarks = landmarks[start:start + self.max_seq_len]
            else:
                indices = np.linspace(0, seq_len - 1, self.max_seq_len, dtype=int)
                landmarks = landmarks[indices]
        elif seq_len < self.max_seq_len:
            pad_len = self.max_seq_len - seq_len
            landmarks = np.vstack([landmarks, np.zeros((pad_len, landmarks.shape[1]))])

        # Normalizacion
        landmarks = (landmarks - self.mean) / self.std

        if self.augment:
            noise = np.random.normal(0, 0.02, landmarks.shape)
            landmarks = landmarks + noise

            scale = 1.0 + np.random.uniform(-0.15, 0.15)
            landmarks = landmarks * scale

        return torch.FloatTensor(landmarks), torch.LongTensor([label])[0]


def _infer_feature_columns(df):
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


def load_data(csv_path, test_size=0.15, val_size=0.15, random_seed=42):
    """Carga y prepara los datos del CSV."""
    print("Cargando datos...")
    df = pd.read_csv(csv_path)

    required_cols = {'glosa', 'video', 'frame'}
    missing_required = required_cols - set(df.columns)
    if missing_required:
        raise ValueError(
            f"El CSV no contiene las columnas requeridas: {sorted(missing_required)}"
        )

    feature_cols = _infer_feature_columns(df)

    glosas = sorted(df['glosa'].dropna().unique())
    glosa_to_idx = {g: i for i, g in enumerate(glosas)}

    print(f"Glosas encontradas: {len(glosas)}")
    print(f"Total de filas: {len(df)}")
    print(f"Features detectadas: {len(feature_cols)}")

    landmarks_data = []
    labels = []

    for video_name, video_df in df.groupby('video', sort=False):
        sort_cols = ['frame']
        if 'hand_index' in video_df.columns:
            sort_cols.append('hand_index')
        video_df = video_df.sort_values(sort_cols)

        glosa = video_df['glosa'].iloc[0]
        glosa_idx = glosa_to_idx[glosa]

        sequence_data = video_df[feature_cols].to_numpy(dtype=np.float32)
        if len(sequence_data) == 0:
            continue

        landmarks_data.append(sequence_data)
        labels.append(glosa_idx)

    print(f"Videos procesados: {len(landmarks_data)}")

    if test_size + val_size >= 1.0:
        raise ValueError("test_size + val_size debe ser menor que 1.0")

    indices = np.arange(len(landmarks_data))
    labels_arr = np.array(labels)

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=(test_size + val_size),
        random_state=random_seed,
        stratify=labels_arr
    )

    temp_labels = labels_arr[temp_idx]
    test_ratio_in_temp = test_size / (test_size + val_size)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=test_ratio_in_temp,
        random_state=random_seed,
        stratify=temp_labels
    )

    print("\nDivision por video:")
    print(f"  Train: {len(train_idx)} videos")
    print(f"  Val: {len(val_idx)} videos")
    print(f"  Test: {len(test_idx)} videos")

    return (
        train_idx.tolist(),
        val_idx.tolist(),
        test_idx.tolist(),
        glosas,
        landmarks_data,
        labels,
    )
