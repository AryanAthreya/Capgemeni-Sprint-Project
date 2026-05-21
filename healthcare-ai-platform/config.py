# stdlib
import os
from functools import lru_cache
from typing import Optional

# third-party
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Centralized application configuration loaded from environment variables.
    All Azure credentials, model paths, and app settings are defined here.
    Provides safe defaults for local development without Azure.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ─── Azure OpenAI ─────────────────────────────────────────────────────────
    azure_openai_endpoint: str = Field(
        default="", description="Azure OpenAI resource endpoint URL"
    )
    azure_openai_api_key: str = Field(
        default="", description="Azure OpenAI API key"
    )
    azure_openai_deployment_name: str = Field(
        default="gpt-4o-mini", description="Chat model deployment name"
    )
    azure_openai_embedding_deployment: str = Field(
        default="text-embedding-3-small", description="Embedding model deployment name"
    )
    azure_openai_api_version: str = Field(
        default="2024-02-01", description="Azure OpenAI API version"
    )

    # ─── Azure AI Search ──────────────────────────────────────────────────────
    azure_search_endpoint: str = Field(
        default="", description="Azure AI Search service endpoint"
    )
    azure_search_key: str = Field(
        default="", description="Azure AI Search admin key"
    )
    azure_search_index_name: str = Field(
        default="medical-knowledge", description="Search index name"
    )

    # ─── Azure CosmosDB ───────────────────────────────────────────────────────
    cosmos_endpoint: str = Field(
        default="", description="CosmosDB account endpoint"
    )
    cosmos_key: str = Field(
        default="", description="CosmosDB primary key"
    )
    cosmos_database: str = Field(
        default="healthcare_db", description="CosmosDB database name"
    )
    cosmos_container: str = Field(
        default="patients", description="CosmosDB container name"
    )

    # ─── Azure Blob Storage ───────────────────────────────────────────────────
    azure_blob_connection_string: str = Field(
        default="", description="Azure Blob Storage connection string"
    )
    azure_blob_container_name: str = Field(
        default="healthcare-data", description="Blob container name"
    )

    # ─── Azure Key Vault ──────────────────────────────────────────────────────
    azure_keyvault_url: str = Field(
        default="", description="Azure Key Vault URL (optional)"
    )

    # ─── Azure Data Factory ───────────────────────────────────────────────────
    adf_subscription_id: str = Field(
        default="", description="Azure subscription ID"
    )
    adf_resource_group: str = Field(
        default="", description="Azure resource group name"
    )
    adf_factory_name: str = Field(
        default="", description="ADF factory name"
    )
    adf_pipeline_name: str = Field(
        default="healthcare-ingest-pipeline", description="ADF pipeline name"
    )

    # ─── Application Settings ─────────────────────────────────────────────────
    app_env: str = Field(
        default="development", description="Application environment"
    )
    log_level: str = Field(
        default="INFO", description="Logging level"
    )
    model_path: str = Field(
        default="models/model.pkl", description="Path to trained ML model"
    )
    label_encoder_path: str = Field(
        default="models/label_encoder.pkl", description="Path to label encoder"
    )

    @property
    def is_development(self) -> bool:
        """Returns True when running in local development mode."""
        return self.app_env.lower() == "development"

    @property
    def azure_openai_configured(self) -> bool:
        """Returns True when Azure OpenAI credentials are provided."""
        return bool(self.azure_openai_endpoint and self.azure_openai_api_key)

    @property
    def azure_search_configured(self) -> bool:
        """Returns True when Azure AI Search credentials are provided."""
        return bool(self.azure_search_endpoint and self.azure_search_key)

    @property
    def cosmos_configured(self) -> bool:
        """Returns True when CosmosDB credentials are provided."""
        return bool(self.cosmos_endpoint and self.cosmos_key)

    @property
    def blob_configured(self) -> bool:
        """Returns True when Azure Blob Storage connection string is provided."""
        return bool(self.azure_blob_connection_string)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Returns the singleton Settings instance (cached after first call)."""
    return Settings()


# Export a single instance for convenience
settings: Settings = get_settings()
