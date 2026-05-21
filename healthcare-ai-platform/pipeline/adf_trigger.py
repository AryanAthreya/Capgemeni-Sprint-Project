# AZURE SETUP REQUIRED:
# 1. Create an Azure Data Factory instance
# 2. Create a pipeline named as per ADF_PIPELINE_NAME env var
# 3. Set env vars: ADF_SUBSCRIPTION_ID, ADF_RESOURCE_GROUP,
#    ADF_FACTORY_NAME, ADF_PIPELINE_NAME
# 4. Configure Azure credentials (DefaultAzureCredential picks up env or managed identity)
# LOCAL FALLBACK: If ADF vars are not set, the trigger is skipped and a message is printed.

# stdlib
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def trigger_adf_pipeline(
    parameters: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """
    Trigger an Azure Data Factory pipeline run programmatically.

    Args:
        parameters: Optional dict of ADF pipeline parameters to pass at runtime.

    Returns:
        Run ID string if triggered successfully, None otherwise.
    """
    from config import settings

    if not all([
        settings.adf_subscription_id,
        settings.adf_resource_group,
        settings.adf_factory_name,
        settings.adf_pipeline_name,
    ]):
        logger.warning(
            "ADF environment variables not fully configured. Skipping pipeline trigger."
        )
        print(
            "Running in local mode — ADF pipeline trigger skipped.\n"
            "Set ADF_SUBSCRIPTION_ID, ADF_RESOURCE_GROUP, ADF_FACTORY_NAME, "
            "ADF_PIPELINE_NAME to enable."
        )
        return None

    try:
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.datafactory import DataFactoryManagementClient

        credential = DefaultAzureCredential()
        adf_client = DataFactoryManagementClient(
            credential=credential,
            subscription_id=settings.adf_subscription_id,
        )

        run_response = adf_client.pipelines.create_run(
            resource_group_name=settings.adf_resource_group,
            factory_name=settings.adf_factory_name,
            pipeline_name=settings.adf_pipeline_name,
            parameters=parameters or {},
        )

        run_id: str = run_response.run_id
        logger.info(
            "ADF pipeline '%s' triggered. Run ID: %s",
            settings.adf_pipeline_name,
            run_id,
        )
        print(f"ADF pipeline triggered. Run ID: {run_id}")
        return run_id

    except ImportError:
        logger.error(
            "azure-mgmt-datafactory not installed. "
            "Add it to requirements.txt to enable ADF triggering."
        )
        return None
    except Exception as exc:
        logger.error("Failed to trigger ADF pipeline: %s", exc)
        return None


def get_pipeline_run_status(run_id: str) -> Optional[str]:
    """
    Poll the status of an ADF pipeline run.

    Args:
        run_id: The pipeline run ID returned by trigger_adf_pipeline().

    Returns:
        Status string (e.g. "Succeeded", "InProgress", "Failed") or None.
    """
    from config import settings

    if not run_id:
        return None

    try:
        from azure.identity import DefaultAzureCredential
        from azure.mgmt.datafactory import DataFactoryManagementClient

        credential = DefaultAzureCredential()
        adf_client = DataFactoryManagementClient(
            credential=credential,
            subscription_id=settings.adf_subscription_id,
        )

        run = adf_client.pipeline_runs.get(
            resource_group_name=settings.adf_resource_group,
            factory_name=settings.adf_factory_name,
            run_id=run_id,
        )
        status: str = run.status
        logger.info("ADF run %s status: %s", run_id, status)
        return status

    except Exception as exc:
        logger.error("Failed to get ADF run status for %s: %s", run_id, exc)
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run = trigger_adf_pipeline()
    if run:
        status = get_pipeline_run_status(run)
        print(f"Pipeline status: {status}")
