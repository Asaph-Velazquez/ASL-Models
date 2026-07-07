# Configuración del modelo
MODEL_CONFIG = {
    "d_model": 256,
    "nhead": 8,
    "num_layers": 4,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "max_seq_len": 100,
    "feature_dim": 150,  
}

# Configuración de entrenamiento
TRAINING_CONFIG = {
    "batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 30,
    "test_size": 0.15,
    "val_size": 0.15,
    "random_seed": 42,
}

# Rutas
DATA_PATH = "data/dataset_35.csv"
OUTPUT_DIR = "output"
MODEL_PATH = "models/best_model.pt"