from azure.cosmos import CosmosClient
import os
from dotenv import load_dotenv

load_dotenv()

# We switched to COSMOS_CONNECTION_STRING in the previous step,
# so we use from_connection_string here!
client = CosmosClient.from_connection_string(os.getenv("COSMOS_CONNECTION_STRING"))

db = client.get_database_client("healthcare_db")
container = db.get_container_client("patients")

# Insert a test document
container.upsert_item({
    "id": "test-001",
    "patient_id": "test-001",
    "name": "Test Patient",
    "age": 25
})
print("Write successful")

# Read it back
item = container.read_item("test-001", partition_key="test-001")
print(f"Read successful: {item['name']}")

# Clean up
container.delete_item("test-001", partition_key="test-001")
print("CosmosDB connection working perfectly")
