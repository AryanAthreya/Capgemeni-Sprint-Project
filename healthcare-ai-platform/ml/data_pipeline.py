# AZURE SETUP REQUIRED:
# 1. No Azure resources needed for this file — runs fully locally.
# LOCAL FALLBACK: Reads from data/raw/Training.csv and writes staged/curated outputs locally.

# stdlib
import logging
import os
import sys
from pathlib import Path

# third-party
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)

# ─── Expected columns ─────────────────────────────────────────────────────────
REQUIRED_TARGET_COLUMN = "prognosis"
EXPECTED_MIN_COLUMNS = 133  # 132 symptom columns + 1 prognosis


def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load Training.csv from the given path and validate its structure.

    Args:
        path: Absolute or relative path to Training.csv.

    Returns:
        Raw DataFrame with all columns intact.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Training data not found at: {csv_path}")

    df = pd.read_csv(csv_path)
    logger.info("Loaded raw data: shape=%s", df.shape)

    if REQUIRED_TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{REQUIRED_TARGET_COLUMN}' not found in dataset. "
            f"Available columns: {list(df.columns[:5])}..."
        )

    if df.shape[1] < EXPECTED_MIN_COLUMNS:
        raise ValueError(
            f"Expected at least {EXPECTED_MIN_COLUMNS} columns, got {df.shape[1]}."
        )

    print(f"[load_raw_data] Shape: {df.shape}")
    print(f"[load_raw_data] Columns (first 5): {list(df.columns[:5])}")
    print(f"[load_raw_data] Target classes: {df[REQUIRED_TARGET_COLUMN].nunique()}")
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicates and normalise the prognosis column.

    Args:
        df: Raw DataFrame loaded from Training.csv.

    Returns:
        Cleaned DataFrame.
    """
    original_len = len(df)
    df = df.drop_duplicates()
    removed = original_len - len(df)
    if removed:
        logger.info("Dropped %d duplicate rows.", removed)

    # Strip whitespace from string columns
    df[REQUIRED_TARGET_COLUMN] = df[REQUIRED_TARGET_COLUMN].str.strip()

    # Drop rows where prognosis is null
    df = df.dropna(subset=[REQUIRED_TARGET_COLUMN])

    # Fill any remaining NaN symptom values with 0
    symptom_cols = [c for c in df.columns if c != REQUIRED_TARGET_COLUMN]
    df[symptom_cols] = df[symptom_cols].fillna(0).astype(int)

    print(f"[clean_data] Shape after cleaning: {df.shape}")
    print(f"[clean_data] Class distribution (top 5):")
    print(df[REQUIRED_TARGET_COLUMN].value_counts().head(5).to_string())
    return df


def encode_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix X from target series y.

    Args:
        df: Cleaned DataFrame.

    Returns:
        Tuple of (X DataFrame with 132 symptom columns, y Series with prognosis labels).
    """
    symptom_cols = [c for c in df.columns if c != REQUIRED_TARGET_COLUMN]
    X = df[symptom_cols].astype(int)
    y = df[REQUIRED_TARGET_COLUMN]

    logger.info("Feature matrix: %s | Target series: %s", X.shape, y.shape)
    print(f"[encode_features] X shape: {X.shape}, y shape: {y.shape}")
    print(f"[encode_features] Unique diseases: {y.nunique()}")
    return X, y


def save_staged(df: pd.DataFrame, path: str) -> None:
    """
    Save the cleaned DataFrame as a CSV to the staged directory.

    Args:
        df: Cleaned DataFrame.
        path: Destination file path (e.g. data/staged/Training_cleaned.csv).
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Staged data saved to: %s", output_path)
    print(f"[save_staged] Saved {len(df)} rows to {output_path}")


def save_curated(X: pd.DataFrame, y: pd.Series, path: str) -> None:
    """
    Save the feature matrix and target as a Parquet file to the curated directory.

    Args:
        X: Feature matrix DataFrame.
        y: Target Series.
        path: Destination file path (e.g. data/curated/features.parquet).
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    curated_df = X.copy()
    curated_df[REQUIRED_TARGET_COLUMN] = y.values

    table = pa.Table.from_pandas(curated_df)
    pq.write_table(table, output_path)
    logger.info("Curated data saved to: %s", output_path)
    print(f"[save_curated] Saved parquet with shape {curated_df.shape} to {output_path}")


def run_pipeline(
    raw_path: str = "data/raw/Training.csv",
    staged_path: str = "data/staged/Training_cleaned.csv",
    curated_path: str = "data/curated/features.parquet",
) -> None:
    """
    End-to-end data pipeline: raw → staged → curated.

    Args:
        raw_path: Path to raw Training.csv.
        staged_path: Output path for cleaned CSV.
        curated_path: Output path for curated Parquet file.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("=" * 60)
    print("HEALTHCARE DATA PIPELINE -- RAW -> STAGED -> CURATED")
    print("=" * 60)

    df_raw = load_raw_data(raw_path)
    df_clean = clean_data(df_raw)
    save_staged(df_clean, staged_path)

    X, y = encode_features(df_clean)
    save_curated(X, y, curated_path)

    print("\n[Pipeline Complete]")
    print(f"  Staged  → {staged_path}")
    print(f"  Curated → {curated_path}")


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/raw/Training.csv"
    run_pipeline(raw_path=raw)
