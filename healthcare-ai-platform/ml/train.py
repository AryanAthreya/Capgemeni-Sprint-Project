# stdlib
import logging
import os
import sys
from pathlib import Path

# third-party
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

logger = logging.getLogger(__name__)

CURATED_PATH = "data/curated/features.parquet"
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "model.pkl"
ENCODER_PATH = MODEL_DIR / "label_encoder.pkl"
CONFUSION_MATRIX_PATH = MODEL_DIR / "confusion_matrix.png"


def load_curated_data() -> tuple[np.ndarray, np.ndarray, LabelEncoder]:
    """
    Load the curated Parquet file and return encoded X, y arrays and the label encoder.

    Returns:
        Tuple of (X numpy array, y numpy array, fitted LabelEncoder).

    Raises:
        FileNotFoundError: If the curated Parquet file is missing.
    """
    parquet_path = Path(CURATED_PATH)
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Curated data not found at {parquet_path}. "
            "Run 'python ml/data_pipeline.py' first."
        )

    df = pq.read_table(parquet_path).to_pandas()
    logger.info("Loaded curated data: shape=%s", df.shape)

    X = df.drop(columns=["prognosis"]).values.astype(int)
    y_labels = df["prognosis"].values

    encoder = LabelEncoder()
    y = encoder.fit_transform(y_labels)

    print(f"[load_curated_data] X shape: {X.shape}")
    print(f"[load_curated_data] Classes: {len(encoder.classes_)}")
    return X, y, encoder


def train_model(
    X: np.ndarray, y: np.ndarray
) -> tuple[RandomForestClassifier, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Train a RandomForestClassifier and split data into train/test sets.

    Args:
        X: Feature matrix (n_samples, 132).
        y: Encoded target labels (n_samples,).

    Returns:
        Tuple of (trained model, X_train, X_test, y_train, y_test).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logger.info(
        "Train size: %d | Test size: %d", len(X_train), len(X_test)
    )
    print(f"[train_model] Training on {len(X_train)} samples...")

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    logger.info("RandomForest training complete.")
    return model, X_train, X_test, y_train, y_test


def evaluate_model(
    model: RandomForestClassifier,
    X_test: np.ndarray,
    y_test: np.ndarray,
    encoder: LabelEncoder,
) -> float:
    """
    Evaluate the model and save a confusion matrix PNG.

    Args:
        model: Trained RandomForestClassifier.
        X_test: Test feature matrix.
        y_test: Encoded test labels.
        encoder: Fitted LabelEncoder for decoding class names.

    Returns:
        Test accuracy as a float.
    """
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"\n[evaluate_model] Test Accuracy: {acc:.4f} ({acc * 100:.2f}%)")
    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred, target_names=encoder.classes_, zero_division=0
        )
    )

    # ─── Confusion matrix heatmap ──────────────────────────────────────────────
    cm = confusion_matrix(y_test, y_pred)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(20, 18))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=encoder.classes_,
        yticklabels=encoder.classes_,
        ax=ax,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Disease Prediction Random Forest", fontsize=14)
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved to: %s", CONFUSION_MATRIX_PATH)
    print(f"[evaluate_model] Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
    return acc


def save_model(model: RandomForestClassifier, encoder: LabelEncoder) -> None:
    """
    Serialize the trained model and label encoder to disk using joblib.

    Args:
        model: Trained RandomForestClassifier.
        encoder: Fitted LabelEncoder.
    """
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    logger.info("Model saved to: %s", MODEL_PATH)
    logger.info("Label encoder saved to: %s", ENCODER_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Label encoder saved to {ENCODER_PATH}")


def main() -> None:
    """
    Full training pipeline: load curated data → train → evaluate → save.
    Run with: python ml/train.py
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    print("=" * 60)
    print("HEALTHCARE ML TRAINING PIPELINE")
    print("=" * 60)

    X, y, encoder = load_curated_data()
    model, _, X_test, _, y_test = train_model(X, y)
    evaluate_model(model, X_test, y_test, encoder)
    save_model(model, encoder)

    print("\n" + "=" * 60)
    print("Model saved to models/model.pkl")
    print("Run inference with: from ml.predict import predict_disease")
    print("=" * 60)


if __name__ == "__main__":
    main()
