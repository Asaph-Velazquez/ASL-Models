# train_and_evaluate.py
import json
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import logging
from pathlib import Path

from dataset.loader import SignDataLoader
from model.train import SignTrainer
from model.predict import SignPredictor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _format_metrics_report(results):
    history = results.get("history", {})
    best_train_acc = max(history.get("accuracy", [0.0])) if history.get("accuracy") else 0.0
    best_val_acc = max(history.get("val_accuracy", [0.0])) if history.get("val_accuracy") else 0.0
    min_train_loss = min(history.get("loss", [0.0])) if history.get("loss") else 0.0
    min_val_loss = min(history.get("val_loss", [0.0])) if history.get("val_loss") else 0.0

    lines = [
        "Mix Atencion - Metrics Report",
        "=" * 32,
        f"Accuracy: {results['accuracy']:.4f}",
        f"Precision weighted: {results['precision_weighted']:.4f}",
        f"Recall weighted: {results['recall_weighted']:.4f}",
        f"F1 weighted: {results['f1_weighted']:.4f}",
        f"Precision macro: {results['precision_macro']:.4f}",
        f"Recall macro: {results['recall_macro']:.4f}",
        f"F1 macro: {results['f1_macro']:.4f}",
        "",
        "Training summary",
        "-" * 16,
        f"Best train accuracy: {best_train_acc:.4f}",
        f"Best val accuracy: {best_val_acc:.4f}",
        f"Min train loss: {min_train_loss:.4f}",
        f"Min val loss: {min_val_loss:.4f}",
        "",
        f"Model path: {results['model_path']}",
    ]
    return "\n".join(lines) + "\n"


def main():
    output_dir = Path("output")
    model_dir = Path("model")
    checkpoint_dir = model_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Cargar datos
    loader = SignDataLoader()
    df = loader.load_and_preprocess()
    X, y, video_ids, feature_dim = loader.create_sequences()
    logger.info(f"Feature dimension detectada: {feature_dim}")

    # 2. Dividir datos
    data_split = loader.split_data(X, y, video_ids)
    loader.save_processed_data(data_split)

    # 3. Entrenar modelo
    trainer = SignTrainer()
    input_shape = (data_split['X_train'].shape[1], data_split['X_train'].shape[2])
    num_classes = len(np.unique(y))

    trainer.build_model(input_shape, num_classes)
    trainer.compile_model()

    history = trainer.train(
        data_split['X_train'], data_split['y_train'],
        data_split['X_val'], data_split['y_val']
    )

    model_path = model_dir / "best_model.keras"
    trainer.save_model(str(model_path))

    # 4. Evaluar
    predictor = SignPredictor(str(model_path), "data/processed/label_encoder.pkl")

    metrics = predictor.evaluate(
        data_split['X_test'],
        data_split['y_test']
    )

    # 5. Guardar reporte
    logger.info("\n" + "=" * 50)
    logger.info("REPORTE DE EVALUACION")
    logger.info("=" * 50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")

    # Guardar matriz de confusion
    predictor.plot_confusion_matrix(metrics, save_path=str(output_dir / "confusion_matrix.png"))

    results = {
        "accuracy": float(metrics["accuracy"]),
        "precision_weighted": float(metrics["precision_weighted"]),
        "recall_weighted": float(metrics["recall_weighted"]),
        "f1_weighted": float(metrics["f1_weighted"]),
        "precision_macro": float(metrics["precision_macro"]),
        "recall_macro": float(metrics["recall_macro"]),
        "f1_macro": float(metrics["f1_macro"]),
        "history": {
            key: [float(value) for value in values]
            for key, values in history.history.items()
        },
        "model_path": str(model_path),
    }
    with open(output_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    report_text = _format_metrics_report(results)
    with open(output_dir / "metrics_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)
    with open(output_dir / "metrics_report.md", "w", encoding="utf-8") as f:
        f.write("```text\n")
        f.write(report_text)
        f.write("```\n")

    logger.info("Evaluacion completada. Graficos guardados.")


if __name__ == "__main__":
    main()
