# train_and_evaluate.py
import json
import logging
import os
from pathlib import Path

import numpy as np
import yaml
from sklearn.metrics import classification_report

from dataset.loader import SignDataLoader
from model.predict import SignPredictor
from model.train import SignTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config" / "config.yaml"


def _to_jsonable(value):
    if isinstance(value, np.ndarray):
        return [_to_jsonable(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main():
    # 1. Load data
    loader = SignDataLoader(str(CONFIG_PATH))
    model_type = os.getenv("MODEL_TYPE_OVERRIDE", loader.config["model"]["type"])
    model_type = str(model_type).lower()
    artifact_root_env = os.getenv("ARTIFACT_ROOT")
    artifact_root = Path(artifact_root_env) if artifact_root_env else BASE_DIR / "artifacts" / model_type
    processed_dir = artifact_root / "processed"
    checkpoint_dir = artifact_root / "checkpoints"
    experiment_dir = BASE_DIR / "results" / model_type

    loader.config["data"]["processed_path"] = str(processed_dir)
    df = loader.load_and_preprocess()
    X, y, video_ids, group_ids = loader.create_sequences()

    # 2. Split data
    data_split = loader.split_data(X, y, video_ids, group_ids)
    loader.save_processed_data(data_split)

    # 3. Train model
    trainer = SignTrainer(str(CONFIG_PATH))
    model_type = os.getenv("MODEL_TYPE_OVERRIDE", trainer.config["model"]["type"])
    model_type = str(model_type).lower()
    trainer.config["model"]["type"] = model_type

    trainer.config["data"]["processed_path"] = str(processed_dir)
    trainer.config["training"]["checkpoint_dir"] = str(checkpoint_dir)

    processed_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    experiment_dir.mkdir(parents=True, exist_ok=True)

    input_shape = (data_split["X_train"].shape[1], data_split["X_train"].shape[2])
    num_classes = len(np.unique(y))

    if model_type == "svm":
        trainer.build_model(X_train=data_split["X_train"], y_train=data_split["y_train"])
    else:
        trainer.build_model(input_shape, num_classes)

    trainer.compile_model()
    history = trainer.train(
        data_split["X_train"],
        data_split["y_train"],
        data_split["X_val"],
        data_split["y_val"],
    )

    model_path = experiment_dir / ("best_model.joblib" if model_type == "svm" else "best_model.keras")
    trainer.save_model(str(model_path))

    # Save config used for the run
    with open(experiment_dir / "config_used.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(trainer.config, f, sort_keys=False, allow_unicode=True)

    # 4. Evaluate
    predictor_model = trainer.model if model_type != "svm" else model_path
    predictor = SignPredictor(predictor_model, processed_dir / "label_encoder.pkl")
    y_pred = predictor.predict_class(data_split["X_test"])
    metrics = predictor.evaluate(data_split["X_test"], data_split["y_test"])

    # 5. Save results
    logger.info("\n" + "=" * 50)
    logger.info("EVALUATION REPORT")
    logger.info("=" * 50)
    logger.info(f"Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"F1-Score (macro): {metrics['f1_macro']:.4f}")
    logger.info(f"F1-Score (weighted): {metrics['f1_weighted']:.4f}")

    report = classification_report(
        data_split["y_test"],
        y_pred,
        target_names=predictor.classes,
        zero_division=0,
    )

    (experiment_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    with open(experiment_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({k: _to_jsonable(v) for k, v in metrics.items()}, f, indent=2, ensure_ascii=False)

    if history is not None and hasattr(history, "history"):
        with open(experiment_dir / "history.json", "w", encoding="utf-8") as f:
            json.dump(_to_jsonable(history.history), f, indent=2, ensure_ascii=False)

    predictor.plot_confusion_matrix(metrics, save_path=experiment_dir / "confusion_matrix.png")

    logger.info(f"Results saved in {experiment_dir}")


if __name__ == "__main__":
    main()
