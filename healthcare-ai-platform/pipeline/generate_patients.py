# stdlib
import csv
import logging
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List

# third-party
from faker import Faker

logger = logging.getLogger(__name__)

fake = Faker()

# ─── Real disease and symptom names from the Kaggle dataset ───────────────────
DISEASES = [
    "Fungal infection", "Allergy", "GERD", "Chronic cholestasis", "Drug Reaction",
    "Peptic ulcer disease", "AIDS", "Diabetes", "Gastroenteritis", "Bronchial Asthma",
    "Hypertension", "Migraine", "Cervical spondylosis", "Paralysis (brain hemorrhage)",
    "Jaundice", "Malaria", "Chicken pox", "Dengue", "Typhoid", "hepatitis A",
    "Hepatitis B", "Hepatitis C", "Hepatitis D", "Hepatitis E", "Alcoholic hepatitis",
    "Tuberculosis", "Common Cold", "Pneumonia", "Dimorphic hemmorhoids(piles)",
    "Heart attack", "Varicose veins", "Hypothyroidism", "Hyperthyroidism",
    "Hypoglycemia", "Osteoarthritis", "Arthritis",
    "(vertigo) Paroymsal  Positional Vertigo", "Acne", "Urinary tract infection",
    "Psoriasis", "Impetigo",
]

SYMPTOM_COLUMNS = [
    "itching", "skin_rash", "nodal_skin_eruptions", "continuous_sneezing", "shivering",
    "chills", "joint_pain", "stomach_pain", "acidity", "ulcers_on_tongue", "muscle_wasting",
    "vomiting", "burning_micturition", "spotting_urination", "fatigue", "weight_gain",
    "anxiety", "cold_hands_and_feets", "mood_swings", "weight_loss", "restlessness",
    "lethargy", "patches_in_throat", "irregular_sugar_level", "cough", "high_fever",
    "sunken_eyes", "breathlessness", "sweating", "dehydration", "indigestion",
    "headache", "yellowish_skin", "dark_urine", "nausea", "loss_of_appetite",
    "pain_behind_the_eyes", "back_pain", "constipation", "abdominal_pain", "diarrhoea",
    "mild_fever", "yellow_urine", "yellowing_of_eyes", "acute_liver_failure",
    "fluid_overload", "swelling_of_stomach", "swelled_lymph_nodes", "malaise",
    "blurred_and_distorted_vision", "phlegm", "throat_irritation", "redness_of_eyes",
    "sinus_pressure", "runny_nose", "congestion", "chest_pain", "weakness_in_limbs",
    "fast_heart_rate", "pain_during_bowel_movements", "pain_in_anal_region",
    "bloody_stool", "irritation_in_anus", "neck_stiffness", "word_finding_difficulty",
    "drifting", "loss_of_balance", "unsteadiness", "weakness_of_one_body_side",
    "loss_of_smell", "bladder_discomfort", "foul_smell_of_urine",
    "continuous_feel_of_urine", "passage_of_gases", "internal_itching",
    "toxic_look_(typhos)", "depression", "irritability", "muscle_pain",
    "altered_sensorium", "red_spots_over_body", "belly_pain", "abnormal_menstruation",
    "dischromic_patches", "watering_from_eyes", "increased_appetite", "polyuria",
    "family_history", "mucoid_sputum", "rusty_sputum", "lack_of_concentration",
    "visual_disturbances", "receiving_blood_transfusion",
    "receiving_unsterile_injections", "coma", "stomach_bleeding",
    "distention_of_abdomen", "history_of_alcohol_consumption",
    "fluid_overload.1", "blood_in_sputum", "prominent_veins_on_calf",
    "palpitations", "painful_walking", "pus_filled_pimples", "blackheads",
    "scurring", "skin_peeling", "silver_like_dusting", "small_dents_in_nails",
    "inflammatory_nails", "blister", "red_sore_around_nose", "yellow_crust_ooze",
]


def _derive_risk_level(age: int, symptom_count: int) -> str:
    """
    Derive risk level from simple clinical rules.

    Args:
        age: Patient age in years.
        symptom_count: Number of reported symptoms.

    Returns:
        Risk level string: "high", "medium", or "low".
    """
    if age > 60 or symptom_count > 4:
        return "high"
    elif age > 40 or symptom_count > 2:
        return "medium"
    return "low"


def generate_patient_records(n: int = 500) -> List[Dict[str, Any]]:
    """
    Generate n synthetic patient records using Faker and real dataset vocabulary.

    Each record contains:
        patient_id, name, age, gender, symptoms, diagnosis, risk_level,
        timestamp, doctor_notes

    Args:
        n: Number of patient records to generate.

    Returns:
        List of patient record dicts.
    """
    records: List[Dict[str, Any]] = []
    now = datetime.now(timezone.utc)

    for _ in range(n):
        age = random.randint(18, 80)
        num_symptoms = random.randint(3, 6)
        symptoms = random.sample(SYMPTOM_COLUMNS, num_symptoms)
        diagnosis = random.choice(DISEASES)
        risk_level = _derive_risk_level(age, num_symptoms)
        gender = random.choice(["Male", "Female", "Other"])
        days_ago = random.randint(0, 30)
        timestamp = now - timedelta(days=days_ago, hours=random.randint(0, 23))

        doctor_notes = (
            f"Patient presented with {symptoms[0].replace('_', ' ')} and "
            f"{symptoms[1].replace('_', ' ')}; assessed as {risk_level} risk."
        )

        records.append(
            {
                "patient_id": str(uuid.uuid4()),
                "name": fake.name(),
                "age": age,
                "gender": gender,
                "symptoms": symptoms,
                "diagnosis": diagnosis,
                "risk_level": risk_level,
                "timestamp": timestamp.isoformat(),
                "doctor_notes": doctor_notes,
            }
        )

    logger.info("Generated %d synthetic patient records.", n)
    return records


def save_to_csv(records: List[Dict[str, Any]], path: str) -> None:
    """
    Save patient records to a CSV file, serialising list fields as pipe-separated strings.

    Args:
        records: List of patient record dicts.
        path: Destination file path.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not records:
        logger.warning("No records to save.")
        return

    fieldnames = list(records[0].keys())
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            row = rec.copy()
            # Serialise list fields for CSV compatibility
            if isinstance(row.get("symptoms"), list):
                row["symptoms"] = "|".join(row["symptoms"])
            writer.writerow(row)

    logger.info("Saved %d patient records to %s", len(records), output_path)
    print(f"Saved {len(records)} patient records to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    print(f"Generating {n} synthetic patient records...")
    records = generate_patient_records(n)
    save_to_csv(records, "data/raw/patients.csv")
    print("Done. Run: python ml/data_pipeline.py to process Training.csv next.")
