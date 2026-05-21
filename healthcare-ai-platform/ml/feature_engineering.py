# stdlib
import logging
import re
from typing import List

# third-party
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def get_symptom_columns(df: pd.DataFrame) -> List[str]:
    """
    Return all 132 symptom column names from the training DataFrame.

    Args:
        df: DataFrame loaded from Training.csv (must contain 'prognosis' column).

    Returns:
        List of symptom column name strings in dataset order.
    """
    return [col for col in df.columns if col != "prognosis"]


def _normalize_symptom(symptom: str) -> str:
    """
    Normalize a symptom string for fuzzy matching.

    Converts spaces and hyphens to underscores, lowercases, and strips whitespace.

    Args:
        symptom: Raw symptom string (e.g. "high fever", "High_Fever").

    Returns:
        Normalized string (e.g. "high_fever").
    """
    return re.sub(r"[\s\-]+", "_", symptom.strip().lower())


def encode_symptoms(symptom_list: List[str], all_columns: List[str]) -> np.ndarray:
    """
    Encode a list of symptom strings into a binary numpy array for ML inference.

    Matching is case-insensitive and normalizes spaces/hyphens to underscores,
    so "high fever", "High Fever", and "high_fever" all match the column "high_fever".

    Args:
        symptom_list: List of symptom strings provided by the user or agent.
        all_columns: Ordered list of all 132 symptom column names from the dataset.

    Returns:
        Binary numpy array of shape (1, len(all_columns)) where 1 indicates
        the symptom is present and 0 indicates absent.

    Example:
        >>> cols = get_symptom_columns(df)
        >>> encode_symptoms(["itching", "skin rash"], cols)
        array([[1, 0, 1, 0, ...]])
    """
    # Build a normalized → original index map for fast lookup
    normalized_col_map: dict[str, int] = {
        _normalize_symptom(col): idx for idx, col in enumerate(all_columns)
    }

    vector = np.zeros((1, len(all_columns)), dtype=int)

    for symptom in symptom_list:
        normalized = _normalize_symptom(symptom)
        if normalized in normalized_col_map:
            idx = normalized_col_map[normalized]
            vector[0, idx] = 1
            logger.debug("Matched symptom '%s' → column index %d", symptom, idx)
        else:
            logger.warning("Symptom '%s' not found in known symptom columns.", symptom)

    matched_count = int(vector.sum())
    logger.info(
        "Encoded %d/%d symptoms from input list.", matched_count, len(symptom_list)
    )
    return vector


def extract_symptom_keywords(text: str, all_columns: List[str]) -> List[str]:
    """
    Extract symptom keywords from free-text by fuzzy matching against known column names.

    Useful when a user describes symptoms in natural language (e.g. from triage agent).

    Args:
        text: Free-text symptom description (e.g. "I have high fever and itching").
        all_columns: Ordered list of all 132 symptom column names.

    Returns:
        List of matched symptom column names.
    """
    normalized_text = _normalize_symptom(text)
    matched: List[str] = []

    for col in all_columns:
        normalized_col = _normalize_symptom(col)
        # Check if the normalized column name appears as a substring in the text
        if normalized_col in normalized_text:
            matched.append(col)

    logger.info("Extracted %d symptom keywords from text.", len(matched))
    return matched
