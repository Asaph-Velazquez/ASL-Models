# model/train.py
import logging
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
import yaml
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW

from .architecture import (
    create_cnn_lstm_model,
    create_lstm_attention_model,
    create_svm_model,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SignTrainer:
    def __init__(self, config_path="config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.scaler = None
        self.history = None

    def build_model(self, input_shape=None, num_classes=None, X_train=None, y_train=None):
        """Builds the model according to the config."""
        model_type = str(self.config["model"]["type"]).lower()

        if model_type == "svm":
            if X_train is None or y_train is None:
                raise ValueError("X_train and y_train are required to train SVM")
            self.model, self.scaler = create_svm_model(X_train, y_train)
        elif model_type == "lstm_attention":
            self.model = create_lstm_attention_model(input_shape, num_classes, self.config)
        elif model_type == "cnn_lstm":
            self.model = create_cnn_lstm_model(input_shape, num_classes, self.config)
        else:
            raise ValueError(f"Model type {model_type} not supported")

        logger.info(f"Model {model_type} created")
        return self.model

    def compile_model(self):
        """Compiles the model."""
        if str(self.config["model"]["type"]).lower() == "svm":
            logger.info("SVM does not require Keras compilation")
            return

        lr = self.config["model"]["learning_rate"]
        wd = self.config["model"]["weight_decay"]

        optimizer = AdamW(learning_rate=lr, weight_decay=wd)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()

        self.model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
        logger.info("Model compiled")

    def train(self, X_train, y_train, X_val, y_val):
        """Trains the model."""
        if str(self.config["model"]["type"]).lower() == "svm":
            logger.info("SVM was already trained during build_model")
            return None

        batch_size = self.config["model"]["batch_size"]
        epochs = self.config["model"]["epochs"]
        patience = self.config["training"]["early_stopping_patience"]

        checkpoint_dir = Path(self.config["training"]["checkpoint_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.5,
                patience=self.config["training"]["reduce_lr_patience"],
                min_lr=float(self.config["training"]["min_lr"]),
                verbose=1,
            ),
            ModelCheckpoint(
                filepath=str(checkpoint_dir / "best_model.h5"),
                monitor="val_accuracy",
                save_best_only=True,
                verbose=1,
            ),
        ]

        logger.info("Starting training...")
        self.history = self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
        )

        logger.info("Training completed")
        return self.history

    def save_model(self, path="model/best_model.keras"):
        """Saves the model."""
        model_type = str(self.config["model"]["type"]).lower()
        if model_type == "svm":
            payload = {
                "model": self.model,
                "scaler": self.scaler,
                "model_type": "svm",
            }
            joblib.dump(payload, path)
        else:
            self.model.save(path)

        logger.info(f"Model saved to {path}")

    def load_model(self, path="model/best_model.keras"):
        """Loads the model."""
        path = str(path)
        if path.endswith((".joblib", ".pkl")):
            payload = joblib.load(path)
            if isinstance(payload, dict):
                self.model = payload.get("model")
                self.scaler = payload.get("scaler")
            else:
                self.model = payload
        else:
            self.model = tf.keras.models.load_model(path)

        logger.info(f"Model loaded from {path}")
        return self.model
