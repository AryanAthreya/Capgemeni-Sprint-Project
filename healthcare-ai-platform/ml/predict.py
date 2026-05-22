# stdlib
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

# third-party
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Artifact paths  (all saved by train_model.py)
# ─────────────────────────────────────────────────────────────────────────────

_ROOT          = Path(__file__).resolve().parent.parent   # project root
_MODEL_PATH    = _ROOT / "models" / "model.pkl"
_ENCODER_PATH  = _ROOT / "models" / "label_encoder.pkl"
_SYM_COLS_PATH = _ROOT / "models" / "symptom_columns.pkl"
_SEV_MAP_PATH  = _ROOT / "models" / "severity_map.pkl"

# Module-level cache (loaded once per process)
_MODEL:    RandomForestClassifier | None = None
_ENCODER:  LabelEncoder            | None = None
_SYM_COLS: List[str]               | None = None
_SEV_MAP:  Dict[str, float]        | None = None


def _load_artifacts() -> None:
    """Load all model artifacts from models/ directory (idempotent)."""
    global _MODEL, _ENCODER, _SYM_COLS, _SEV_MAP

    if _MODEL is not None:
        return  # already loaded

    for path in [_MODEL_PATH, _ENCODER_PATH, _SYM_COLS_PATH, _SEV_MAP_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: '{path}'. "
                "Run 'python train_model.py' from the project root first."
            )

    _MODEL    = joblib.load(_MODEL_PATH)
    _ENCODER  = joblib.load(_ENCODER_PATH)
    _SYM_COLS = joblib.load(_SYM_COLS_PATH)
    _SEV_MAP  = joblib.load(_SEV_MAP_PATH)

    logger.info(
        "ML artifacts loaded | classes=%d | features=%d",
        len(_ENCODER.classes_),
        len(_SYM_COLS),
    )


def _normalize(symptom: str) -> str:
    """Lowercase, strip, replace whitespace/hyphens with underscores."""
    return re.sub(r"[\s\-]+", "_", symptom.strip().lower())


def _assign_risk_level(confidence: float) -> str:
    if confidence >= 0.7:
        return "high"
    elif confidence >= 0.4:
        return "medium"
    return "low"


def predict_disease(
    symptom_list: List[str],
    top_k: int = 3,
) -> List[Dict[str, Any]]:
    """
    Predict the most likely diseases from a list of symptom strings.

    Builds a severity-weighted feature vector (identical to train_model.py)
    and returns the top-k predictions sorted by confidence descending.

    Args:
        symptom_list: Symptom strings e.g. ["itching", "skin_rash", "fever"].
        top_k: Number of top predictions to return (default 3).

    Returns:
        List of dicts: [{disease, confidence, risk_level}]

    Raises:
        FileNotFoundError: If model artifacts are missing (run train_model.py).
        ValueError: If symptom_list is empty.
    """
    if not symptom_list:
        raise ValueError("symptom_list must contain at least one symptom.")

    _load_artifacts()

    # Build severity-weighted feature vector
    # Length = len(_SYM_COLS) + 1  (+1 for total_symptom_count)
    vector  = np.zeros(len(_SYM_COLS) + 1, dtype=float)
    matched = []

    for symptom in symptom_list:
        norm = _normalize(symptom)
        if norm in _SYM_COLS:
            idx          = _SYM_COLS.index(norm)
            weight       = _SEV_MAP.get(norm, 1.0)
            vector[idx]  = weight
            matched.append(norm)
        else:
            logger.warning("Symptom not recognised: '%s'", symptom)

    # total_symptom_count is the last feature
    vector[-1] = len(matched)

    if not matched:
        return [{"disease": "No recognised symptoms", "confidence": 0.0, "risk_level": "unknown"}]

    # Predict
    proba       = _MODEL.predict_proba([vector])[0]
    top_indices = np.argsort(proba)[::-1][:top_k]

    results: List[Dict[str, Any]] = []
    for idx in top_indices:
        conf = float(round(proba[idx], 4))
        results.append({
            "disease":    _ENCODER.classes_[idx],
            "confidence": conf,
            "risk_level": _assign_risk_level(conf),
        })

    logger.info(
        "Prediction | symptoms=%s | top=%s (%.2f%%)",
        symptom_list,
        results[0]["disease"],
        results[0]["confidence"] * 100,
    )
    return results
