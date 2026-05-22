"""
train_model.py — Standalone Disease Prediction Training Script
===============================================================
Mirrors the pipeline from ml/disease_prediction_pipeline.ipynb exactly.

Usage (run from project root):
    python train_model.py

Output saved to models/:
    model.pkl           — Trained RandomForestClassifier
    label_encoder.pkl   — LabelEncoder (int <-> disease name)
    symptom_columns.pkl — Ordered list of symptom feature names
    feature_names.pkl   — All feature names (symptoms + total_symptom_count)
    severity_map.pkl    — Symptom severity weight dict
    metadata.json       — Accuracy, classes, feature count
"""

import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Paths  (always relative to this script's location = project root)
# ─────────────────────────────────────────────────────────────────────────────

ROOT_DIR   = Path(__file__).resolve().parent
DATA_DIR   = ROOT_DIR / "ml"           # where the 4 CSVs live
MODELS_DIR = ROOT_DIR / "models"        # output directory

DATASET_CSV     = DATA_DIR / "dataset.csv"
DESCRIPTION_CSV = DATA_DIR / "symptom_Description.csv"
PRECAUTION_CSV  = DATA_DIR / "symptom_precaution.csv"
SEVERITY_CSV    = DATA_DIR / "Symptom-severity.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Load CSVs
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    """Load all 4 CSVs and return raw DataFrames."""
    for path in [DATASET_CSV, SEVERITY_CSV]:
        if not path.exists():
            log.error("Missing file: %s", path)
            sys.exit(1)

    df_main        = pd.read_csv(DATASET_CSV)
    df_description = pd.read_csv(DESCRIPTION_CSV) if DESCRIPTION_CSV.exists() else None
    df_precaution  = pd.read_csv(PRECAUTION_CSV)  if PRECAUTION_CSV.exists()  else None
    df_severity    = pd.read_csv(SEVERITY_CSV)

    log.info("Loaded dataset.csv        : %s", df_main.shape)
    log.info("Loaded Symptom-severity   : %s", df_severity.shape)
    return df_main, df_description, df_precaution, df_severity


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Preprocess  (matches notebook cell 13 exactly)
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df_main: pd.DataFrame):
    """
    Clean dataset.csv and convert to binary symptom matrix with 'prognosis' column.

    The raw dataset.csv is in WIDE FORMAT:
        Disease | Symptom_1 | Symptom_2 | ... | Symptom_17
    where each Symptom_N cell contains a symptom name string (or NaN).

    We convert it to BINARY FORMAT:
        prognosis | itching | skin_rash | ... (one column per unique symptom)
    where each cell is 1 (present) or 0 (absent).

    Each original row is preserved (4920 rows → 4920 rows), unlike crosstab
    which would incorrectly collapse all rows per disease into one.
    """
    df = df_main.copy()

    # Strip column name whitespace
    df.columns = df.columns.str.strip()

    # Identify symptom columns
    symptom_value_cols = [c for c in df.columns if c.startswith("Symptom_")]
    disease_col = "Disease"

    # Clean disease names (removes trailing spaces like "Diabetes ", "Hypertension ")
    df[disease_col] = df[disease_col].str.strip()

    # Drop unnamed/junk columns
    junk = [c for c in df.columns if "unnamed" in c.lower()]
    if junk:
        df.drop(columns=junk, inplace=True)
        log.info("Dropped junk columns: %s", junk)

    # ── Normalise every symptom cell ──────────────────────────────────────────
    def _norm(val):
        if pd.isnull(val):
            return None
        cleaned = str(val).strip().lower()
        import re as _re
        cleaned = _re.sub(r"\s+", "_", cleaned)  # all whitespace → _
        return cleaned if cleaned else None

    for col in symptom_value_cols:
        df[col] = df[col].apply(_norm)

    # ── Build binary matrix row-by-row ────────────────────────────────────────
    # Add an integer row index so we can pivot correctly
    df["_row"] = range(len(df))

    # Melt wide → long  (each symptom cell becomes a separate row)
    df_long = df.melt(
        id_vars=[disease_col, "_row"],
        value_vars=symptom_value_cols,
        var_name="_slot",
        value_name="symptom",
    )
    # Drop NaN / empty symptom slots
    df_long = df_long.dropna(subset=["symptom"])
    df_long = df_long[df_long["symptom"] != ""]

    # Pivot back to wide binary format: rows = original rows, cols = unique symptoms
    df_binary = (
        df_long
        .assign(val=1)
        .pivot_table(
            index=[disease_col, "_row"],
            columns="symptom",
            values="val",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
        .drop(columns=["_row"])
    )
    df_binary.columns.name = None

    # Rename disease column to 'prognosis' to match the rest of the pipeline
    df_binary.rename(columns={disease_col: "prognosis"}, inplace=True)

    # Remove duplicate rows
    before = len(df_binary)
    df_binary.drop_duplicates(inplace=True)
    removed = before - len(df_binary)
    if removed:
        log.info("Removed %d duplicate rows", removed)

    # Fill any leftover NaN with 0
    symptom_cols = [c for c in df_binary.columns if c != "prognosis"]
    df_binary[symptom_cols] = df_binary[symptom_cols].fillna(0).astype(int)

    log.info(
        "Preprocessed shape: %s  |  diseases: %d  |  symptoms: %d",
        df_binary.shape,
        df_binary["prognosis"].nunique(),
        len(symptom_cols),
    )
    return df_binary, symptom_cols



# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Build severity map  (matches notebook cell 15)
# ─────────────────────────────────────────────────────────────────────────────

def build_severity_map(df_severity: pd.DataFrame) -> dict:
    """
    Build symptom_name → severity_weight (1-7) dict from Symptom-severity.csv.
    Normalises names the same way as training data.
    """
    severity_map = {}
    weight_col = "weight" if "weight" in df_severity.columns else df_severity.columns[1]

    for _, row in df_severity.iterrows():
        symptom = str(row["Symptom"]).strip().lower().replace(" ", "_")
        try:
            severity_map[symptom] = float(row[weight_col])
        except (ValueError, TypeError):
            pass

    log.info("Severity map: %d symptoms", len(severity_map))
    return severity_map


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Build feature matrix  (matches notebook cell 15)
# ─────────────────────────────────────────────────────────────────────────────

def build_feature_matrix(df_clean: pd.DataFrame, symptom_cols: list, severity_map: dict):
    """
    Build the feature matrix X:
      - Start with binary symptom matrix
      - Multiply each column by its severity weight (if available)
      - Add one extra feature: total_symptom_count per row
    """
    X = df_clean[symptom_cols].copy().astype(float)

    if severity_map:
        matched = 0
        for col in symptom_cols:
            col_norm = col.strip().lower().replace(" ", "_")
            if col_norm in severity_map:
                X[col] = X[col] * severity_map[col_norm]
                matched += 1
        log.info("Severity weights applied to %d / %d symptoms", matched, len(symptom_cols))

    # Extra feature: symptom count
    X["total_symptom_count"] = df_clean[symptom_cols].sum(axis=1)

    feature_names = list(X.columns)
    log.info("Feature matrix shape: %s  (%d symptoms + 1 count)", X.shape, len(symptom_cols))
    return X, feature_names


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Encode labels
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(df_clean: pd.DataFrame):
    """LabelEncode the 'prognosis' column."""
    le = LabelEncoder()
    y  = le.fit_transform(df_clean["prognosis"].values)
    log.info("Label encoded: %d classes", len(le.classes_))
    return y, le


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Train
# ─────────────────────────────────────────────────────────────────────────────

def train(X, y):
    """Train/test split → RandomForest fit."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y,   # safe now: 4920 rows, ~120 per class
    )

    log.info("Train: %d  |  Test: %d", len(X_train), len(X_test))
    print(f"\n[train] Training on {len(X_train)} samples, {X_train.shape[1]} features...")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    log.info("Training complete — %d trees", model.n_estimators)
    return model, X_train, X_test, y_train, y_test


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_train, X_test, y_train, y_test, le):
    """Print accuracy, F1, and classification report."""
    y_pred      = model.predict(X_test)
    train_acc   = accuracy_score(y_train, model.predict(X_train))
    test_acc    = accuracy_score(y_test, y_pred)
    macro_f1    = f1_score(y_test, y_pred, average="macro",    zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print("\n" + "=" * 50)
    print("  MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"  Training Accuracy  : {train_acc * 100:.2f}%")
    print(f"  Test Accuracy      : {test_acc  * 100:.2f}%")
    print(f"  Macro F1 Score     : {macro_f1:.4f}")
    print(f"  Weighted F1 Score  : {weighted_f1:.4f}")
    print(f"  Train samples      : {X_train.shape[0]}")
    print(f"  Test  samples      : {X_test.shape[0]}")
    print("=" * 50)

    print("\nClassification Report:")
    print(
        classification_report(
            y_test, y_pred,
            labels=sorted(set(y_test)),          # only classes in test split
            target_names=le.classes_[sorted(set(y_test))],
            zero_division=0,
        )
    )

    return test_acc, macro_f1, weighted_f1


# ─────────────────────────────────────────────────────────────────────────────
# Step 8 — Save artifacts
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(model, le, symptom_cols, feature_names, severity_map,
                   test_acc, macro_f1):
    """Save all model artifacts to models/ directory."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Core model files
    joblib.dump(model,         MODELS_DIR / "model.pkl")
    joblib.dump(le,            MODELS_DIR / "label_encoder.pkl")
    joblib.dump(symptom_cols,  MODELS_DIR / "symptom_columns.pkl")
    joblib.dump(feature_names, MODELS_DIR / "feature_names.pkl")
    joblib.dump(severity_map,  MODELS_DIR / "severity_map.pkl")

    # Metadata JSON
    metadata = {
        "model_type"    : "RandomForestClassifier",
        "n_estimators"  : model.n_estimators,
        "n_features"    : int(model.n_features_in_),
        "n_classes"     : int(len(le.classes_)),
        "test_accuracy" : round(float(test_acc), 4),
        "macro_f1"      : round(float(macro_f1), 4),
        "symptom_cols"  : symptom_cols,
        "diseases"      : list(le.classes_),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[OK] All artifacts saved to models/")
    print(f"{'File':<35} {'Size':>8}")
    print("-" * 45)
    for fname in sorted(MODELS_DIR.iterdir()):
        if fname.is_file() and fname.name != ".gitkeep":
            print(f"  {fname.name:<33} {fname.stat().st_size / 1024:>6.1f} KB")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  DISEASE PREDICTION — TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load
    print("\n[1/7] Loading CSV files...")
    df_main, _, _, df_severity = load_data()

    # 2. Preprocess
    print("[2/7] Preprocessing data...")
    df_clean, symptom_cols = preprocess(df_main)

    # 3. Severity map
    print("[3/7] Building severity map...")
    severity_map = build_severity_map(df_severity)

    # 4. Feature matrix
    print("[4/7] Building feature matrix...")
    X, feature_names = build_feature_matrix(df_clean, symptom_cols, severity_map)

    # 5. Encode labels
    print("[5/7] Encoding labels...")
    y, le = encode_labels(df_clean)

    # 6. Train
    print("[6/7] Training Random Forest...")
    model, X_train, X_test, y_train, y_test = train(X.values, y)

    # 7. Evaluate
    print("[7/7] Evaluating model...")
    test_acc, macro_f1, _ = evaluate(model, X_train, X_test, y_train, y_test, le)

    # 8. Save
    save_artifacts(model, le, symptom_cols, feature_names, severity_map, test_acc, macro_f1)

    print("\n" + "=" * 60)
    print("  Done! Run predictions with:")
    print("    from ml_work.inference import get_prediction_for_api")
    print("=" * 60)


if __name__ == "__main__":
    main()
