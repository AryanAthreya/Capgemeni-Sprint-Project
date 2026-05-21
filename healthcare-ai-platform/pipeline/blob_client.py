# AZURE SETUP REQUIRED:
# 1. Create an Azure Storage Account and Blob Container
# 2. Set env var: AZURE_BLOB_CONNECTION_STRING (from Access Keys in portal)
# 3. Set env var: AZURE_BLOB_CONTAINER_NAME (default: healthcare-data)
# LOCAL FALLBACK: If AZURE_BLOB_CONNECTION_STRING is not set, files are copied
#                 locally to a simulated blob directory and a warning is printed.

# stdlib
import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class BlobStorageClient:
    """
    Azure Blob Storage client for uploading and downloading data pipeline files.
    Transparently falls back to local file copy when Azure is not configured.
    """

    def __init__(self) -> None:
        """Initialise lazily — Azure client created on first method call."""
        self._client: object = None
        self._container_client: object = None
        self._use_local: bool = False
        self._local_base: Path = Path("data/blob_local_sim")

    def _get_container_client(self) -> object:
        """
        Return the Azure BlobServiceClient container client (or local mode sentinel).

        Returns:
            Azure ContainerClient or None (for local mode).
        """
        if self._container_client is not None:
            return self._container_client
        if self._use_local:
            return None

        from config import settings

        if not settings.blob_configured:
            logger.warning(
                "AZURE_BLOB_CONNECTION_STRING not set — running in local mode. "
                "Files will be copied to %s/",
                self._local_base,
            )
            self._use_local = True
            self._local_base.mkdir(parents=True, exist_ok=True)
            return None

        try:
            from azure.storage.blob import BlobServiceClient

            service_client = BlobServiceClient.from_connection_string(
                settings.azure_blob_connection_string
            )
            self._container_client = service_client.get_container_client(
                settings.azure_blob_container_name
            )
            logger.info(
                "Azure Blob connected to container: %s",
                settings.azure_blob_container_name,
            )
            return self._container_client
        except Exception as exc:
            logger.warning("Blob Storage init failed: %s — falling back to local mode.", exc)
            self._use_local = True
            self._local_base.mkdir(parents=True, exist_ok=True)
            return None

    def upload_file(self, local_path: str, blob_path: str) -> bool:
        """
        Upload a local file to Azure Blob Storage (or local sim folder).

        Args:
            local_path: Path to the local source file.
            blob_path: Destination blob path (e.g. "raw/Training.csv").

        Returns:
            True if upload succeeded, False otherwise.
        """
        src = Path(local_path)
        if not src.exists():
            logger.error("File not found for upload: %s", local_path)
            return False

        container = self._get_container_client()

        if self._use_local:
            dest = self._local_base / blob_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            print(f"[LOCAL MODE] Copied {local_path} → {dest}")
            logger.info("Local sim upload: %s → %s", local_path, dest)
            return True

        try:
            with open(src, "rb") as data:
                container.upload_blob(name=blob_path, data=data, overwrite=True)
            logger.info("Uploaded %s → blob://%s", local_path, blob_path)
            return True
        except Exception as exc:
            logger.error("Blob upload failed for %s: %s", local_path, exc)
            return False

    def download_file(self, blob_path: str, local_path: str) -> bool:
        """
        Download a blob to a local file.

        Args:
            blob_path: Source blob path in the container.
            local_path: Destination local file path.

        Returns:
            True if download succeeded, False otherwise.
        """
        container = self._get_container_client()
        dest = Path(local_path)
        dest.parent.mkdir(parents=True, exist_ok=True)

        if self._use_local:
            src = self._local_base / blob_path
            if src.exists():
                shutil.copy2(src, dest)
                print(f"[LOCAL MODE] Copied {src} → {dest}")
                return True
            logger.error("Local sim file not found: %s", src)
            return False

        try:
            blob_client = container.get_blob_client(blob_path)
            with open(dest, "wb") as f:
                stream = blob_client.download_blob()
                f.write(stream.readall())
            logger.info("Downloaded blob://%s → %s", blob_path, local_path)
            return True
        except Exception as exc:
            logger.error("Blob download failed for %s: %s", blob_path, exc)
            return False

    def list_files(self, prefix: str = "") -> List[str]:
        """
        List all blob names with a given prefix.

        Args:
            prefix: Filter blobs by this prefix (e.g. "raw/").

        Returns:
            List of blob name strings.
        """
        container = self._get_container_client()

        if self._use_local:
            base = self._local_base / prefix
            if not base.exists():
                return []
            return [str(p.relative_to(self._local_base)) for p in base.rglob("*") if p.is_file()]

        try:
            blobs = container.list_blobs(name_starts_with=prefix)
            return [b.name for b in blobs]
        except Exception as exc:
            logger.error("Blob list failed: %s", exc)
            return []


def upload_raw_data(client: Optional[BlobStorageClient] = None) -> None:
    """
    Upload all CSV files in data/raw/ to the blob container under /raw/ prefix.

    Args:
        client: Optional BlobStorageClient instance (creates one if not provided).
    """
    blob = client or BlobStorageClient()
    raw_dir = Path("data/raw")
    files = list(raw_dir.glob("*.csv"))

    if not files:
        logger.warning("No CSV files found in data/raw/ to upload.")
        return

    for f in files:
        blob.upload_file(str(f), f"raw/{f.name}")

    print(f"Uploaded {len(files)} raw file(s) to blob:/raw/")


def upload_curated_data(client: Optional[BlobStorageClient] = None) -> None:
    """
    Upload all Parquet files in data/curated/ to the blob container under /curated/ prefix.

    Args:
        client: Optional BlobStorageClient instance (creates one if not provided).
    """
    blob = client or BlobStorageClient()
    curated_dir = Path("data/curated")
    files = list(curated_dir.glob("*.parquet"))

    if not files:
        logger.warning("No Parquet files found in data/curated/ to upload.")
        return

    for f in files:
        blob.upload_file(str(f), f"curated/{f.name}")

    print(f"Uploaded {len(files)} curated file(s) to blob:/curated/")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    client = BlobStorageClient()
    upload_raw_data(client)
    upload_curated_data(client)
