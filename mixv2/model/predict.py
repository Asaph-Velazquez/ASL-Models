# model/predict.py
import logging
from pathlib import Path
from typing import Dict

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

logger = logging.getLogger(__name__)


class SignPredictor:
    def __init__(self, model_path_or_model, label_encoder_path):
        self.model_path = None
        self.is_svm = False
        self.scaler = None

        if isinstance(model_path_or_model, tf.keras.Model):
            self.model = model_path_or_model
        else:
            self.model_path = Path(model_path_or_model)
            self.is_svm = self.model_path.suffix.lower() in {".joblib", ".pkl"}

            if self.is_svm:
                payload = joblib.load(self.model_path)
                if isinstance(payload, dict):
                    self.model = payload.get("model")
                    self.scaler = payload.get("scaler")
                else:
                    self.model = payload
            else:
                self.model = tf.keras.models.load_model(self.model_path)

        self.label_encoder = joblib.load(label_encoder_path)
        self.classes = self.label_encoder.classes_

    def _prepare_input(self, X):
        if not self.is_svm:
            return X

        if self.scaler is None:
            raise ValueError("SVM scaler was not loaded")

        X_flat = X.reshape(X.shape[0], -1)
        return self.scaler.transform(X_flat)

    def predict(self, X):
        """Runs predictions."""
        X_prepared = self._prepare_input(X)
        if self.is_svm:
            return self.model.predict(X_prepared)
        return self.model.predict(X_prepared)

    def predict_class(self, X):
        """Returns the predicted class label."""
        preds = self.predict(X)
        if self.is_svm:
            return np.asarray(preds)
        return np.argmax(preds, axis=1)

    def evaluate(self, X_test, y_test) -> Dict:
        """Evaluates the model."""
        y_pred = self.predict_class(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average="weighted", zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average="macro", zero_division=0
        )

        metrics = {
            "accuracy": accuracy,
            "precision_weighted": precision,
            "recall_weighted": recall,
            "f1_weighted": f1,
            "precision_macro": precision_macro,
            "recall_macro": recall_macro,
            "f1_macro": f1_macro,
            "confusion_matrix": confusion_matrix(y_test, y_pred),
        }

        return metrics

    def plot_confusion_matrix(self, metrics, save_path=None):
        """Visualizes the confusion matrix."""
        cm = metrics["confusion_matrix"]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            xticklabels=self.classes,
            yticklabels=self.classes,
            cmap="Blues",
        )
        plt.title("Confusion Matrix")
        plt.ylabel("Real")
        plt.xlabel("Predicted")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
