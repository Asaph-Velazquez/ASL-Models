import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from model import SignLanguageTransformer
from dataset import load_data, KPCA_Dataset
from config import *

def create_dataloaders(landmarks, labels, indices, batch_size=32, max_seq_len=100, augment=False):
    """Crea dataloader a partir de índices"""
    data = [landmarks[i] for i in indices]
    lbls = [labels[i] for i in indices]
    
    dataset = KPCA_Dataset(data, lbls, max_seq_len, augment)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=augment, num_workers=0)
    
    return loader

def train_epoch(model, loader, optimizer, criterion, device):
    """Entrena una época"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for landmarks, labels in tqdm(loader, desc="Training"):
        landmarks, labels = landmarks.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(landmarks)
        loss = criterion(outputs, labels)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    return total_loss / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, device):
    """Evalúa el modelo"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    criterion = nn.CrossEntropyLoss()
    
    for landmarks, labels in tqdm(loader, desc="Evaluating"):
        landmarks, labels = landmarks.to(device), labels.to(device)
        
        outputs = model(landmarks)
        loss = criterion(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    return total_loss / len(loader), correct / total, all_preds, all_labels

def plot_confusion_matrix(cm, glosas, save_path):
    """Genera y guarda la matriz de confusión"""
    plt.figure(figsize=(15, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
               xticklabels=glosas, yticklabels=glosas,
               annot_kws={'size': 10})
    plt.title('Matriz de Confusión - Test Set', fontsize=16)
    plt.xlabel('Predicción', fontsize=14)
    plt.ylabel('Real', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Matriz de confusión guardada en: {save_path}")

def plot_training_history(history, save_path):
    """Grafica el historial de entrenamiento"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', linewidth=2)
    axes[0].plot(history['val_loss'], label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Época')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Evolución de la Pérdida')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train Acc', linewidth=2)
    axes[1].plot(history['val_acc'], label='Val Acc', linewidth=2)
    axes[1].set_xlabel('Época')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Evolución de la Precisión')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Historial guardado en: {save_path}")

def main():
    # Crear directorios
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path('models').mkdir(parents=True, exist_ok=True)
    
    # Configurar dispositivo
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}")
    
    # 1. Cargar datos
    print("\n" + "="*60)
    print("CARGANDO DATOS")
    print("="*60)
    
    train_idx, val_idx, test_idx, glosas = load_data(
        DATA_PATH,
        test_size=TRAINING_CONFIG['test_size'],
        val_size=TRAINING_CONFIG['val_size'],
        random_seed=TRAINING_CONFIG['random_seed']
    )
    
    # 2. Crear dataloaders
    print("\n" + "="*60)
    print("CREANDO DATALOADERS")
    print("="*60)
    
    # Primero cargar todos los datos para obtener estadísticas
    df = pd.read_csv(DATA_PATH)
    kpca_cols = [f'kpca_{i}' for i in range(150)]
    videos = df['video'].unique()
    
    landmarks_data = []
    labels = []
    glosa_to_idx = {g: i for i, g in enumerate(glosas)}
    
    for video_name in videos:
        video_df = df[df['video'] == video_name].sort_values('frame')
        glosa = video_df['glosa'].iloc[0]
        glosa_idx = glosa_to_idx[glosa]
        kpca_data = video_df[kpca_cols].values
        
        if len(kpca_data) >= 10:
            if len(kpca_data) > 100:
                indices = np.linspace(0, len(kpca_data) - 1, 100, dtype=int)
                kpca_data = kpca_data[indices]
            landmarks_data.append(kpca_data)
            labels.append(glosa_idx)
    
    train_loader = create_dataloaders(
        landmarks_data, labels, train_idx,
        batch_size=TRAINING_CONFIG['batch_size'],
        max_seq_len=MODEL_CONFIG['max_seq_len'],
        augment=True
    )
    
    val_loader = create_dataloaders(
        landmarks_data, labels, val_idx,
        batch_size=TRAINING_CONFIG['batch_size'],
        max_seq_len=MODEL_CONFIG['max_seq_len'],
        augment=False
    )
    
    test_loader = create_dataloaders(
        landmarks_data, labels, test_idx,
        batch_size=TRAINING_CONFIG['batch_size'],
        max_seq_len=MODEL_CONFIG['max_seq_len'],
        augment=False
    )
    
    print(f"Train: {len(train_loader.dataset)} muestras")
    print(f"Val: {len(val_loader.dataset)} muestras")
    print(f"Test: {len(test_loader.dataset)} muestras")
    
    # 3. Crear modelo
    print("\n" + "="*60)
    print("CREANDO MODELO")
    print("="*60)
    
    model = SignLanguageTransformer(
        num_classes=len(glosas),
        d_model=MODEL_CONFIG['d_model'],
        nhead=MODEL_CONFIG['nhead'],
        num_layers=MODEL_CONFIG['num_layers'],
        dim_feedforward=MODEL_CONFIG['dim_feedforward'],
        dropout=MODEL_CONFIG['dropout'],
        max_seq_len=MODEL_CONFIG['max_seq_len'],
        feature_dim=MODEL_CONFIG['feature_dim']
    ).to(device)
    
    print(f"Total de parámetros: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Clases: {len(glosas)}")
    
    # 4. Configurar entrenamiento
    optimizer = optim.AdamW(
        model.parameters(),
        lr=TRAINING_CONFIG['learning_rate'],
        weight_decay=TRAINING_CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 5. Entrenar
    print("\n" + "="*60)
    print("INICIANDO ENTRENAMIENTO")
    print("="*60)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_acc = 0
    patience_counter = 0
    early_stopping_patience = 10
    
    for epoch in range(TRAINING_CONFIG['epochs']):
        print(f"\nÉpoca {epoch+1}/{TRAINING_CONFIG['epochs']}")
        print("-"*40)
        
        # Entrenar
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validar
        val_loss, val_acc, _, _ = evaluate(model, val_loader, device)
        
    
        scheduler.step(val_loss)
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Guardar mejor modelo
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'epoch': epoch,
                'config': MODEL_CONFIG
            }, MODEL_PATH)
            print(f"✓ Nuevo mejor modelo guardado (val_acc: {val_acc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{early_stopping_patience}")
        
        if patience_counter >= early_stopping_patience:
            print(f"Early stopping en época {epoch+1}")
            break
    
    # 6. Graficar historial
    print("\n" + "="*60)
    print("GENERANDO GRÁFICOS")
    print("="*60)
    
    plot_training_history(history, f"{OUTPUT_DIR}/training_history.png")
    
    # 7. Evaluar en test
    print("\n" + "="*60)
    print("EVALUACIÓN EN TEST SET")
    print("="*60)
    
    # Cargar mejor modelo
    checkpoint = torch.load(MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, preds, true_labels = evaluate(model, test_loader, device)
    
    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Accuracy (%): {test_acc*100:.2f}%")
    
    # 8. Matriz de confusión
    cm = confusion_matrix(true_labels, preds)
    plot_confusion_matrix(cm, glosas, f"{OUTPUT_DIR}/confusion_matrix.png")
    
    # 9. Reporte de clasificación detallado
    print("\n" + "="*60)
    print("REPORTE DE CLASIFICACIÓN")
    print("="*60)
    
    report = classification_report(true_labels, preds, target_names=glosas, digits=4)
    print(report)
    
    # 10. Métricas por clase
    print("\n" + "="*60)
    print("MÉTRICAS POR CLASE")
    print("="*60)
    
    class_metrics = {}
    for i, glosa in enumerate(glosas):
        total_class = np.sum(np.array(true_labels) == i)
        if total_class > 0:
            correct_class = np.sum((np.array(true_labels) == i) & (np.array(preds) == i))
            class_acc = correct_class / total_class
            class_metrics[glosa] = {
                'accuracy': class_acc,
                'total': int(total_class),
                'correct': int(correct_class),
                'errors': int(total_class - correct_class)
            }
            print(f"{glosa}:")
            print(f"  Accuracy: {class_acc:.4f} ({correct_class}/{total_class})")
            print(f"  Errores: {total_class - correct_class}")
    
    # 11. Guardar todos los resultados
    results = {
        'test_accuracy': test_acc,
        'test_loss': test_loss,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'class_metrics': class_metrics,
        'history': history,
        'glosas': glosas,
        'predictions': [int(p) for p in preds],
        'true_labels': [int(l) for l in true_labels],
        'model_config': MODEL_CONFIG,
        'training_config': TRAINING_CONFIG
    }
    
    with open(f"{OUTPUT_DIR}/results.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    # 12. Guardar matriz como CSV
    cm_df = pd.DataFrame(cm, index=glosas, columns=glosas)
    cm_df.to_csv(f"{OUTPUT_DIR}/confusion_matrix.csv")
    
    # 13. Resumen final
    print("\n" + "="*60)
    print("RESUMEN FINAL")
    print("="*60)
    print(f" Mejor Accuracy en Test: {test_acc*100:.2f}%")
    print(f" Modelo guardado en: {MODEL_PATH}")
    print(f"Matriz de confusión: {OUTPUT_DIR}/confusion_matrix.png")
    print(f"Resultados completos: {OUTPUT_DIR}/results.json")
    print("Archivos generados:")
    print(f"   - {OUTPUT_DIR}/training_history.png")
    print(f"   - {OUTPUT_DIR}/confusion_matrix.png")
    print(f"   - {OUTPUT_DIR}/confusion_matrix.csv")
    print(f"   - {OUTPUT_DIR}/results.json")
    print(f"   - {MODEL_PATH}")
    
    print("\n" + "="*60)
    print("ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
    print("="*60)

if __name__ == "__main__":
    main()