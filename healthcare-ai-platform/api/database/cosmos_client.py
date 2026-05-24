# AZURE SETUP REQUIRED:
# 1. Create an Azure CosmosDB account (API: NoSQL / Core SQL)
# 2. Create a database named as per COSMOS_DATABASE env var (default: healthcare_db)
# 3. Create a container named as per COSMOS_CONTAINER env var (default: patients)
#    - Partition key: /patient_id
# 4. Set env vars: COSMOS_ENDPOINT, COSMOS_KEY, COSMOS_DATABASE, COSMOS_CONTAINER
# LOCAL FALLBACK: If COSMOS_ENDPOINT is not set, an in-memory dict store is used
#                 so all code works without any Azure resources.

# stdlib
import logging
import uuid
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class _MockCosmosStore:
    """
    In-memory dict-based mock that mimics CosmosDB CRUD operations.
    Used automatically when COSMOS_ENDPOINT is not configured.
    """

    def __init__(self) -> None:
        """Initialise the in-memory store."""
        self._store: Dict[str, Dict[str, Any]] = {}
        logger.warning(
            "CosmosDB not configured — using in-memory mock store. "
            "Data will be lost on restart."
        )

    def upsert_item(self, body: Dict[str, Any]) -> Dict[str, Any]:
        """Upsert a document into the mock store."""
        key = body.get("id") or body.get("patient_id", str(uuid.uuid4()))
        body["id"] = key
        self._store[key] = body
        return body

    def read_item(self, item: str, partition_key: str) -> Dict[str, Any]:
        """Read a single document by ID."""
        if item not in self._store:
            raise KeyError(f"Item '{item}' not found in mock store.")
        return self._store[item]

    def query_items(
        self, query: str, parameters: Optional[List[Dict]] = None, **kwargs: Any
    ) -> List[Dict[str, Any]]:
        """Simple mock query that returns all items (ignores SQL)."""
        return list(self._store.values())

    def __len__(self) -> int:
        """Return the number of stored documents."""
        return len(self._store)


class CosmosDBClient:
    """
    Azure CosmosDB client wrapper with CRUD operations for patient records.
    Falls back to an in-memory mock store when Azure credentials are absent.
    """

    def __init__(self) -> None:
        """
        Initialise the CosmosDB client lazily.
        The actual connection is established on first use via _get_container().
        """
        self._container: Any = None
        self._mock_store: Optional[_MockCosmosStore] = None
        self._use_mock: bool = False

    def _get_container(self) -> Any:
        """
        Return the CosmosDB container client, initialising it on first call.

        Returns:
            Azure ContainerProxy or _MockCosmosStore depending on config.
        """
        if self._container is not None:
            return self._container
        if self._mock_store is not None:
            return self._mock_store

        from config import settings

        if not settings.cosmos_configured:
            self._mock_store = _MockCosmosStore()
            self._use_mock = True
            return self._mock_store

        try:
            from azure.cosmos import CosmosClient, exceptions as cosmos_exceptions

            client = CosmosClient.from_connection_string(
                conn_str=settings.cosmos_connection_string
            )
            db = client.get_database_client(settings.cosmos_database)
            self._container = db.get_container_client(settings.cosmos_container)
            logger.info(
                "Connected to CosmosDB: %s / %s",
                settings.cosmos_database,
                settings.cosmos_container,
            )
            return self._container

        except Exception as exc:
            logger.error("Failed to connect to CosmosDB: %s", exc)
            logger.warning("Falling back to in-memory mock store.")
            self._mock_store = _MockCosmosStore()
            self._use_mock = True
            return self._mock_store

    def insert_patient(self, patient_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Upsert a patient document into CosmosDB.

        Args:
            patient_dict: Patient record dictionary. Must contain 'patient_id'.

        Returns:
            The upserted document dictionary.
        """
        container = self._get_container()
        patient_dict["id"] = patient_dict.get("patient_id", str(uuid.uuid4()))

        try:
            if self._use_mock:
                result = container.upsert_item(patient_dict)
            else:
                result = container.upsert_item(body=patient_dict)

            logger.info("Upserted patient: %s", patient_dict["id"])
            return result

        except Exception as exc:
            logger.error("Error upserting patient %s: %s", patient_dict.get("id"), exc)
            raise

    def get_patient(self, patient_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a patient document by ID.

        Args:
            patient_id: The patient UUID string.

        Returns:
            Patient dict if found, None otherwise.
        """
        container = self._get_container()
        try:
            if self._use_mock:
                return container.read_item(patient_id, partition_key=patient_id)
            else:
                return container.read_item(
                    item=patient_id, partition_key=patient_id
                )
        except KeyError:
            logger.warning("Patient not found: %s", patient_id)
            return None
        except Exception as exc:
            logger.error("Error reading patient %s: %s", patient_id, exc)
            return None

    def query_patients(
        self,
        query_str: str,
        params: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Run a parameterized SQL query against the patients container.

        Args:
            query_str: CosmosDB SQL query string.
            params: Optional list of parameter dicts for the query.

        Returns:
            List of matching patient documents.
        """
        container = self._get_container()
        try:
            if self._use_mock:
                return list(container.query_items(query=query_str, parameters=params))
            else:
                items = container.query_items(
                    query=query_str,
                    parameters=params or [],
                    enable_cross_partition_query=True,
                )
                return list(items)
        except Exception as exc:
            logger.error("Error querying patients: %s", exc)
            return []

    def get_all_patients(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Return up to `limit` patient records.

        Args:
            limit: Maximum number of records to return.

        Returns:
            List of patient document dicts.
        """
        query = f"SELECT TOP {limit} * FROM c"
        return self.query_patients(query)

    def close(self) -> None:
        """Clean up the CosmosDB client connection."""
        self._container = None
        self._mock_store = None
        logger.info("CosmosDB client closed.")


# Module-level singleton
_cosmos_client: Optional[CosmosDBClient] = None


def get_cosmos_client() -> CosmosDBClient:
    """Return the module-level singleton CosmosDBClient instance."""
    global _cosmos_client
    if _cosmos_client is None:
        _cosmos_client = CosmosDBClient()
    return _cosmos_client
