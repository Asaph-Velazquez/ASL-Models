import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import random

class KPCA_Dataset(Dataset):
    def __init__(self, landmarks, labels, max_seq_len=100, augment=False):
        self.landmarks = landmarks
        self.labels = labels
        self.max_seq_len = max_seq_len
        self.augment = augment
        
        # Calcular estadísticas
        all_data = np.concatenate([l for l in landmarks], axis=0)
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
        
        # Normalización
        landmarks = (landmarks - self.mean) / self.std
        
        # Data augmentation
        if self.augment:
            # Ruido
            noise = np.random.normal(0, 0.02, landmarks.shape)
            landmarks = landmarks + noise
            
            # Escala
            scale = 1.0 + np.random.uniform(-0.15, 0.15)
            landmarks = landmarks * scale
        
        return torch.FloatTensor(landmarks), torch.LongTensor([label])[0]

def load_data(csv_path, test_size=0.15, val_size=0.15, random_seed=42):
    """Carga y prepara los datos del CSV"""
    print("Cargando datos...")
    df = pd.read_csv(csv_path)
    
    # Obtener glosas únicas
    glosas = sorted(df['glosa'].unique())
    glosa_to_idx = {g: i for i, g in enumerate(glosas)}
    
    print(f"Glosas encontradas: {len(glosas)}")
    print(f"Total de filas: {len(df)}")
    
    # Obtener videos únicos
    videos = df['video'].unique()
    print(f"Total de videos: {len(videos)}")
    
    # Procesar cada video
    landmarks_data = []
    labels = []
    video_info = []
    
    kpca_cols = [f'kpca_{i}' for i in range(150)]
    
    for video_name in videos:
        video_df = df[df['video'] == video_name].sort_values('frame')
        
        glosa = video_df['glosa'].iloc[0]
        glosa_idx = glosa_to_idx[glosa]
        
        # Extraer KPCA features
        kpca_data = video_df[kpca_cols].values
        
        if len(kpca_data) < 10:
            continue
        
        # Limitar frames
        if len(kpca_data) > 100:
            indices = np.linspace(0, len(kpca_data) - 1, 100, dtype=int)
            kpca_data = kpca_data[indices]
        
        landmarks_data.append(kpca_data)
        labels.append(glosa_idx)
        video_info.append({
            'video': video_name,
            'glosa': glosa,
            'frames': len(kpca_data)
        })
    
    print(f"Videos procesados: {len(landmarks_data)}")
    
    # Dividir por signante (mejor para evaluación realista)
    return split_by_signer(landmarks_data, labels, glosas, test_size, val_size, random_seed)

def split_by_signer(landmarks, labels, glosas, test_size, val_size, random_seed):
    """Divide por signante (asumiendo que el nombre del video contiene ID)"""
    import re
    
    # Extraer signante del nombre del video
    signer_indices = defaultdict(list)
    for idx, info in enumerate(landmarks):
        # Extraer número del video (asumiendo formato help_000001.mp4)
        numbers = re.findall(r'\d+', str(idx))
        signer_id = int(numbers[0]) % 10 if numbers else 0
        signer_indices[signer_id].append(idx)
    
    # Dividir signantes
    random.seed(random_seed)
    signers = list(signer_indices.keys())
    random.shuffle(signers)
    
    n_train = int(len(signers) * (1 - test_size - val_size))
    n_val = int(len(signers) * val_size)
    
    train_signers = signers[:n_train]
    val_signers = signers[n_train:n_train + n_val]
    test_signers = signers[n_train + n_val:]
    
    train_idx = [i for s in train_signers for i in signer_indices[s]]
    val_idx = [i for s in val_signers for i in signer_indices[s]]
    test_idx = [i for s in test_signers for i in signer_indices[s]]
    
    print(f"\nDivisión por signante:")
    print(f"  Train: {len(train_idx)} videos de {len(train_signers)} signantes")
    print(f"  Val: {len(val_idx)} videos de {len(val_signers)} signantes")
    print(f"  Test: {len(test_idx)} videos de {len(test_signers)} signantes")
    
    return (train_idx, val_idx, test_idx, glosas)