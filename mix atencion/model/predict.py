# model/predict.py
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict

from .architecture import Attention

logger = logging.getLogger(__name__)


class SignPredictor:
    def __init__(self, model_path, label_encoder_path):
        self.model = tf.keras.models.load_model(
            model_path,
            custom_objects={"Attention": Attention},
        )
        self.label_encoder = joblib.load(label_encoder_path)
        self.classes = self.label_encoder.classes_

    def predict(self, X):
        """Realiza predicciones"""
        return self.model.predict(X)

    def predict_class(self, X):
        """Predice la clase con mayor probabilidad"""
        probs = self.predict(X)
        return np.argmax(probs, axis=1)

    def evaluate(self, X_test, y_test) -> Dict:
        """Evalua el modelo"""
        y_pred = self.predict_class(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro'
        )

        metrics = {
            'accuracy': accuracy,
            'precision_weighted': precision,
            'recall_weighted': recall,
            'f1_weighted': f1,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }

        return metrics

    def plot_confusion_matrix(self, metrics, save_path=None):
        """Visualiza matriz de confusion"""
        cm = metrics['confusion_matrix']

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            xticklabels=self.classes,
            yticklabels=self.classes,
            cmap='Blues'
        )
        plt.title('Matriz de Confusion')
        plt.ylabel('Real')
        plt.xlabel('Predicho')
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path)
        plt.close()
