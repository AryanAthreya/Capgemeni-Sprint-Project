# stdlib
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Dict, Any

# third-party
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# local
from ml.feature_engineering import encode_symptoms, get_symptom_columns

logger = logging.getLogger(__name__)

_MODEL: RandomForestClassifier | None = None
_ENCODER: LabelEncoder | None = None
_SYMPTOM_COLUMNS: List[str] | None = None

# These paths can be overridden via config but default to the standard locations
_DEFAULT_MODEL_PATH = "models/model.pkl"
_DEFAULT_ENCODER_PATH = "models/label_encoder.pkl"
_DEFAULT_STAGED_CSV = "data/staged/Training_cleaned.csv"


def _get_symptom_columns() -> List[str]:
    """
    Load symptom columns from the staged CSV once and cache them.

    Returns:
        Ordered list of 132 symptom column names.
    """
    global _SYMPTOM_COLUMNS
    if _SYMPTOM_COLUMNS is None:
        import pandas as pd

        staged_path = Path(_DEFAULT_STAGED_CSV)
        if staged_path.exists():
            df = pd.read_csv(staged_path, nrows=1)
            _SYMPTOM_COLUMNS = get_symptom_columns(df)
        else:
            # Fallback: load columns from the raw training data if staged not available
            raw_path = Path("data/raw/Training.csv")
            if raw_path.exists():
                df = pd.read_csv(raw_path, nrows=1)
                _SYMPTOM_COLUMNS = get_symptom_columns(df)
            else:
                raise FileNotFoundError(
                    "Cannot find symptom column definitions. "
                    "Run 'python ml/data_pipeline.py' first."
                )
    return _SYMPTOM_COLUMNS


def load_model(
    model_path: str = _DEFAULT_MODEL_PATH,
    encoder_path: str = _DEFAULT_ENCODER_PATH,
) -> tuple[RandomForestClassifier, LabelEncoder]:
    """
    Load the trained model and label encoder from disk (cached after first load).

    Args:
        model_path: Path to model.pkl file.
        encoder_path: Path to label_encoder.pkl file.

    Returns:
        Tuple of (RandomForestClassifier, LabelEncoder).

    Raises:
        FileNotFoundError: If model or encoder files do not exist.
    """
    global _MODEL, _ENCODER

    if _MODEL is None or _ENCODER is None:
        mp = Path(model_path)
        ep = Path(encoder_path)

        if not mp.exists():
            raise FileNotFoundError(
                f"Model not found at '{mp}'. Run 'python ml/train.py' first."
            )
        if not ep.exists():
            raise FileNotFoundError(
                f"Encoder not found at '{ep}'. Run 'python ml/train.py' first."
            )

        _MODEL = joblib.load(mp)
        _ENCODER = joblib.load(ep)
        logger.info("ML model loaded from: %s", mp)
        logger.info("Label encoder loaded from: %s", ep)

    return _MODEL, _ENCODER


def _assign_risk_level(confidence: float) -> str:
    """
    Assign a risk level based on prediction confidence.

    Args:
        confidence: Probability score from model (0.0 – 1.0).

    Returns:
        Risk level string: "high", "medium", or "low".
    """
    if confidence >= 0.7:
        return "high"
    elif confidence >= 0.4:
        return "medium"
    return "low"


def predict_disease(
    symptom_list: List[str],
    top_k: int = 3,
    model_path: str = _DEFAULT_MODEL_PATH,
    encoder_path: str = _DEFAULT_ENCODER_PATH,
) -> List[Dict[str, Any]]:
    """
    Predict the most likely diseases given a list of symptom strings.

    Encodes the symptom list into a binary feature vector, runs predict_proba,
    and returns the top-k predictions sorted by confidence descending.

    Args:
        symptom_list: List of symptom strings (e.g. ["itching", "skin_rash"]).
        top_k: Number of top predictions to return (default 3).
        model_path: Path to the saved model file.
        encoder_path: Path to the saved label encoder file.

    Returns:
        List of prediction dicts, each containing:
            - disease (str): Disease name.
            - confidence (float): Probability score rounded to 4 decimal places.
            - risk_level (str): "high" / "medium" / "low".

    Raises:
        FileNotFoundError: If model files are missing.
        ValueError: If symptom_list is empty.
    """
    if not symptom_list:
        raise ValueError("symptom_list must contain at least one symptom string.")

    model, encoder = load_model(model_path, encoder_path)
    all_cols = _get_symptom_columns()

    feature_vector = encode_symptoms(symptom_list, all_cols)
    proba = model.predict_proba(feature_vector)[0]  # shape: (n_classes,)

    # Get indices of top-k probabilities
    top_indices = np.argsort(proba)[::-1][:top_k]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        confidence = float(round(proba[idx], 4))
        disease_name = encoder.inverse_transform([idx])[0]
        results.append(
            {
                "disease": disease_name,
                "confidence": confidence,
                "risk_level": _assign_risk_level(confidence),
            }
        )

    logger.info(
        "Predicted top-%d diseases for symptoms %s: %s",
        top_k,
        symptom_list,
        [r["disease"] for r in results],
    )
    return results
