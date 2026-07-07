# model/train.py
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
import numpy as np
import yaml
import logging
from pathlib import Path
from .architecture import create_lstm_attention_model, create_cnn_lstm_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignTrainer:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.history = None
        
    def build_model(self, input_shape, num_classes):
        """Construye el modelo según configuración"""
        model_type = self.config['model']['type']
        
        if model_type == 'lstm_attention':
            self.model = create_lstm_attention_model(
                input_shape, num_classes, self.config
            )
        elif model_type == 'cnn_lstm':
            self.model = create_cnn_lstm_model(
                input_shape, num_classes, self.config
            )
        else:
            raise ValueError(f"Model type {model_type} not supported")
        
        logger.info(f"Modelo {model_type} creado")
        return self.model
    
    def compile_model(self):
        """Compila el modelo"""
        lr = float(self.config['model']['learning_rate'])
        wd = float(self.config['model']['weight_decay'])
        
        optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        logger.info("Modelo compilado")
        
    def train(self, X_train, y_train, X_val, y_val):
        """Entrena el modelo"""
        batch_size = int(self.config['model']['batch_size'])
        epochs = int(self.config['model']['epochs'])
        patience = int(self.config['training']['early_stopping_patience'])
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=int(self.config['training']['reduce_lr_patience']),
                min_lr=float(self.config['training']['min_lr']),
                verbose=1
            ),
            ModelCheckpoint(
                filepath='model/checkpoints/best_model.weights.h5',
                monitor='val_accuracy',
                save_best_only=True,
                save_weights_only=True,
                verbose=1
            )
        ]
        
        # Entrenar
        logger.info("Iniciando entrenamiento...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("Entrenamiento completado")
        return self.history
    
    def save_model(self, path="model/best_model.keras"):
        """Guarda el modelo"""
        self.model.save(path)
        logger.info(f"Modelo guardado en {path}")
    
    def load_model(self, path="model/best_model.keras"):
        """Carga el modelo"""
        self.model = tf.keras.models.load_model(path)
        logger.info(f"Modelo cargado desde {path}")
        return self.model
