# stdlib
import logging
import uuid
from datetime import datetime, timezone

# third-party
from fastapi import APIRouter, HTTPException, Request

# local
from api.database.sqlite_client import get_db_client
from api.models.schemas import PatientIngest, PatientIngestResponse

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/ingest",
    response_model=PatientIngestResponse,
    summary="Ingest a patient record",
    description="Store a new patient record in MongoDB. "
    "Generates a UUID if patient_id is not supplied.",
    tags=["Data Ingestion"],
)
async def ingest_patient(patient: PatientIngest, request: Request) -> PatientIngestResponse:
    """
    POST /api/ingest

    Accepts a PatientIngest payload, assigns a UUID if needed, and upserts
    the record into MongoDB. Returns the patient_id and operation status.

    Args:
        patient: Validated PatientIngest request body.
        request: Starlette request (used to read the X-Request-ID header).

    Returns:
        PatientIngestResponse with patient_id, status, and message.

    Raises:
        HTTPException 422: If validation fails (handled by FastAPI automatically).
        HTTPException 500: If the database write fails.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    # Assign a UUID if patient_id was not provided
    patient_id = patient.patient_id or str(uuid.uuid4())
    timestamp = patient.timestamp or datetime.now(timezone.utc)

    patient_doc = {
        "patient_id": patient_id,
        "id": patient_id,
        "name": patient.name,
        "age": patient.age,
        "gender": patient.gender,
        "symptoms": patient.symptoms,
        "timestamp": timestamp.isoformat(),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "request_id": request_id,
    }

    logger.info(
        "Ingesting patient | patient_id=%s | name=%s | request_id=%s",
        patient_id,
        patient.name,
        request_id,
    )

    try:
        db = get_db_client()
        await db.insert_patient(patient_doc)
        
        # Dual-write to CosmosDB
        from api.database.cosmos_client import get_cosmos_client
        cosmos_db = get_cosmos_client()
        cosmos_db.insert_patient(patient_doc.copy())

    except Exception as exc:
        logger.error(
            "DB error ingesting patient %s: %s | request_id=%s",
            patient_id,
            exc,
            request_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to store patient record: {str(exc)}",
        ) from exc

    logger.info(
        "Patient ingested successfully | patient_id=%s | request_id=%s",
        patient_id,
        request_id,
    )
    return PatientIngestResponse(
        patient_id=patient_id,
        status="created",
        message=f"Patient record '{patient.name}' ingested successfully.",
    )
