# stdlib
import json
import logging
import sqlite3
import uuid
from typing import Any, Dict, List, Optional

import os

logger = logging.getLogger(__name__)

class SQLiteClient:
    """
    Async-like SQLite client wrapper using standard sqlite3.
    Stores unstructured documents as JSON to mimic MongoDB interface.
    """

    def __init__(self, db_path: str = None) -> None:
        self.db_path = db_path or os.getenv("SQLITE_DB_PATH", "healthcare.db")
        self._init_db()

    def _init_db(self):
        """Initialise the SQLite database and create the patients table if it doesn't exist."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS patients (
                        patient_id TEXT PRIMARY KEY,
                        data TEXT NOT NULL
                    )
                    """
                )
            logger.info("Connected to SQLite: %s", self.db_path)
        except Exception as exc:
            logger.error("Failed to initialise SQLite database: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Public async API (wraps sync SQLite calls to keep interface identical)
    # ------------------------------------------------------------------

    async def insert_patient(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        Insert (or upsert) a patient document.
        """
        patient_id = doc.get("patient_id") or str(uuid.uuid4())
        doc["patient_id"] = patient_id
        doc["_id"] = patient_id

        doc_json = json.dumps(doc)

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO patients (patient_id, data) 
                VALUES (?, ?) 
                ON CONFLICT(patient_id) DO UPDATE SET data=excluded.data
                """,
                (patient_id, doc_json)
            )
            
        logger.info("Upserted patient into SQLite: %s", patient_id)
        return doc

    async def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a patient document by patient_id.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT data FROM patients WHERE patient_id = ?", (patient_id,))
            row = cursor.fetchone()
            
            if row:
                doc = json.loads(row[0])
                doc.pop("_id", None)  # mimic previous behavior
                return doc
                
        return None

    async def get_all_patients(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Return up to *limit* patient records.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT data FROM patients LIMIT ?", (limit,))
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                doc = json.loads(row[0])
                doc.pop("_id", None)
                results.append(doc)
                
            return results

    async def query_by_field(
        self, field: str, value: Any
    ) -> List[Dict[str, Any]]:
        """
        Return all patient documents where *field* equals *value*.
        (Uses python-side filtering since JSON structure is dynamic)
        """
        with sqlite3.connect(self.db_path) as conn:
            # For simplicity in SQLite with dynamic JSON schemas, we fetch all and filter in Python.
            # If performance becomes an issue, we can use SQLite JSON1 extension.
            cursor = conn.execute("SELECT data FROM patients")
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                doc = json.loads(row[0])
                if doc.get(field) == value:
                    doc.pop("_id", None)
                    results.append(doc)
                    
            return results

    async def close(self) -> None:
        """No persistent connection kept, just for interface compatibility."""
        logger.info("SQLite client closed.")

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_sqlite_client: Optional[SQLiteClient] = None

def get_db_client() -> SQLiteClient:
    """Return the module-level singleton SQLiteClient instance."""
    global _sqlite_client
    if _sqlite_client is None:
        _sqlite_client = SQLiteClient()
    return _sqlite_client
