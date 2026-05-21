# stdlib
import logging
import sys
from pathlib import Path

# third-party
import pandas as pd

logger = logging.getLogger(__name__)

PATIENTS_CSV_SOURCES = [
    "data/raw/patients.csv",
    "exports/healthcare_powerbi.csv",
]
OUTPUT_PATH = "exports/healthcare_powerbi.csv"


def load_curated_patients() -> pd.DataFrame:
    """
    Load patient records from the best available CSV source.

    Returns:
        DataFrame of patient records.

    Raises:
        FileNotFoundError: If no patient data CSV is found.
    """
    for path in PATIENTS_CSV_SOURCES:
        p = Path(path)
        if p.exists():
            df = pd.read_csv(p)
            logger.info("Loaded %d patient records from %s", len(df), path)
            print(f"[load_curated_patients] Loaded {len(df)} rows from {path}")
            return df

    raise FileNotFoundError(
        "No patient CSV found. Run: python pipeline/generate_patients.py first."
    )


def add_analytics_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich the patient DataFrame with computed analytics columns for Power BI.

    Adds:
        - symptom_count: number of symptoms per patient
        - age_group: bucketed age range string
        - month: calendar month extracted from timestamp
        - week: ISO week number extracted from timestamp

    Args:
        df: Raw patient DataFrame.

    Returns:
        Enriched DataFrame with added columns.
    """
    df = df.copy()

    # ─── Symptom count ────────────────────────────────────────────────────────
    if "symptoms" in df.columns:
        df["symptom_count"] = df["symptoms"].apply(
            lambda x: len(str(x).split("|")) if pd.notna(x) else 0
        )
    else:
        df["symptom_count"] = 0

    # ─── Age group ────────────────────────────────────────────────────────────
    def _age_group(age: int) -> str:
        """Return the age bracket string for a given age."""
        if pd.isna(age):
            return "Unknown"
        age = int(age)
        if age <= 30:
            return "18-30"
        elif age <= 50:
            return "31-50"
        elif age <= 70:
            return "51-70"
        return "71+"

    if "age" in df.columns:
        df["age_group"] = df["age"].apply(_age_group)
    else:
        df["age_group"] = "Unknown"

    # ─── Month and week from timestamp ───────────────────────────────────────
    if "timestamp" in df.columns:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        df["month"] = ts.dt.month_name()
        df["week"] = ts.dt.isocalendar().week.astype("Int64")
        df["date"] = ts.dt.date
    else:
        df["month"] = "Unknown"
        df["week"] = 0
        df["date"] = None

    # ─── Risk level normalisation ─────────────────────────────────────────────
    if "risk_level" in df.columns:
        df["risk_level"] = df["risk_level"].str.lower().fillna("unknown")

    logger.info("Analytics columns added. Final shape: %s", df.shape)
    print(f"[add_analytics_columns] Columns added: symptom_count, age_group, month, week, date")
    return df


def export_for_powerbi(df: pd.DataFrame, output_path: str = OUTPUT_PATH) -> None:
    """
    Save the enriched DataFrame as a CSV file ready for Power BI ingestion.

    The exported CSV contains all original fields plus:
        symptom_count, age_group, month, week, date

    Args:
        df: Enriched patient DataFrame.
        output_path: Destination CSV file path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8")
    logger.info("Power BI export saved to: %s (%d rows)", out, len(df))
    print(f"Exported {len(df)} rows to {out}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    print("=" * 55)
    print("POWER BI EXPORT PIPELINE")
    print("=" * 55)

    output = sys.argv[1] if len(sys.argv) > 1 else OUTPUT_PATH

    df_raw = load_curated_patients()
    df_enriched = add_analytics_columns(df_raw)
    export_for_powerbi(df_enriched, output_path=output)

    print(f"\nExported {len(df_enriched)} rows to {output}")
    print("Open exports/healthcare_powerbi.csv in Power BI Desktop.")
