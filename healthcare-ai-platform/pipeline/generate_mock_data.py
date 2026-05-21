import csv
import random
from pathlib import Path
import sys

# Import components from our existing patient generator
from pipeline.generate_patients import SYMPTOM_COLUMNS, DISEASES, generate_patient_records, save_to_csv

def generate_mock_training_csv(path: str = "data/raw/Training.csv"):
    """
    Generate a mock Training.csv that contains all 132 symptom columns
    and at least 15 stratified rows for each of the 41 diseases.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = []
    for disease in DISEASES:
        for _ in range(15):  # 15 samples per disease to support Stratified K-Fold
            row = {}
            for col in SYMPTOM_COLUMNS:
                # Randomly assign symptom (15% chance of being active)
                row[col] = 1 if random.random() < 0.15 else 0
            row["prognosis"] = disease
            rows.append(row)
            
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = SYMPTOM_COLUMNS + ["prognosis"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Generated mock training dataset with {len(rows)} samples at: {output_path}")

def main():
    print("=" * 60)
    print("HEALTHCARE AI PLATFORM — MOCK DATA BOOTSTRAPPER")
    print("=" * 60)
    
    # 1. Generate patients.csv
    print("\n[Step 1/2] Generating synthetic patient records...")
    patient_records = generate_patient_records(500)
    save_to_csv(patient_records, "data/raw/patients.csv")
    
    # 2. Generate Training.csv
    print("\n[Step 2/2] Generating mock Training.csv dataset...")
    generate_mock_training_csv("data/raw/Training.csv")
    
    print("\n" + "=" * 60)
    print("Bootstrap complete! All offline dataset mocks generated successfully.")
    print("Now run the ML pipeline: python ml/data_pipeline.py")
    print("=" * 60)

if __name__ == "__main__":
    main()
