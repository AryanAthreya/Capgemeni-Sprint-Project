import httpx
import random
import asyncio

async def seed():
    names = [
        "Alice Smith", "Bob Jones", "Charlie Brown", "Diana Prince", 
        "Ethan Hunt", "Fiona Gallagher", "George Costanza", "Hannah Abbott", 
        "Ian Malcolm", "Julia Roberts"
    ]
    symptoms_list = [
        ["fever", "cough"], 
        ["headache", "nausea"], 
        ["joint pain", "swelling"], 
        ["chest pain", "shortness of breath"], 
        ["fatigue", "dizziness"],
        ["sore throat", "runny nose"],
        ["abdominal pain"],
        ["rash", "itching"]
    ]
    
    print("Starting data ingestion to API...")
    async with httpx.AsyncClient() as client:
        for name in names:
            payload = {
                "name": name,
                "age": random.randint(18, 85),
                "gender": random.choice(["male", "female"]),
                "symptoms": random.choice(symptoms_list)
            }
            try:
                res = await client.post("http://localhost:8000/api/ingest", json=payload, timeout=10.0)
                if res.status_code == 200:
                    print(f"[OK] Successfully inserted: {name}")
                else:
                    print(f"[FAIL] Failed to insert {name}: {res.text}")
            except Exception as e:
                print(f"[ERROR] Error inserting {name}: {e}")

if __name__ == "__main__":
    asyncio.run(seed())
