# stdlib
import logging
from datetime import datetime, timezone
from typing import List

# third-party
from fastapi import APIRouter, HTTPException, Request

# local
from api.database.cosmos_client import get_cosmos_client
from api.models.schemas import PredictRequest, PredictResponse, PredictionResult
from ml.predict import predict_disease

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Predict disease from symptoms",
    description=(
        "Accepts a list of symptom strings and returns the top 3 predicted diseases "
        "with confidence scores and risk levels. Optionally links the prediction to a "
        "patient record in CosmosDB."
    ),
    tags=["ML Inference"],
)
async def predict(body: PredictRequest, request: Request) -> PredictResponse:
    """
    POST /api/predict

    Calls the RandomForest model to predict diseases from a symptom list.
    Optionally updates the patient CosmosDB record with the prediction result.

    Args:
        body: PredictRequest with symptoms list and optional patient_id.
        request: Starlette request for tracing.

    Returns:
        PredictResponse with top-3 predictions and a medical disclaimer.

    Raises:
        HTTPException 400: If symptom list is empty.
        HTTPException 500: If the ML model fails.
    """
    request_id = getattr(request.state, "request_id", "unknown")

    if not body.symptoms:
        raise HTTPException(
            status_code=400,
            detail="Symptom list is empty. Provide at least one symptom.",
        )

    logger.info(
        "Prediction request | symptoms=%s | patient_id=%s | request_id=%s",
        body.symptoms,
        body.patient_id,
        request_id,
    )

    try:
        raw_predictions = predict_disease(body.symptoms, top_k=3)
    except FileNotFoundError as exc:
        logger.error("Model not found: %s | request_id=%s", exc, request_id)
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.error(
            "Prediction failed | symptoms=%s | error=%s | request_id=%s",
            body.symptoms,
            exc,
            request_id,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Prediction model error: {str(exc)}",
        ) from exc

    predictions = [
        PredictionResult(
            disease=p["disease"],
            confidence=p["confidence"],
            risk_level=p["risk_level"],
        )
        for p in raw_predictions
    ]

    logger.info(
        "Prediction result | top=%s (%.2f) | patient_id=%s | request_id=%s",
        predictions[0].disease if predictions else "none",
        predictions[0].confidence if predictions else 0.0,
        body.patient_id,
        request_id,
    )

    # Optionally enrich the patient record with this prediction
    if body.patient_id:
        try:
            cosmos = get_cosmos_client()
            existing = cosmos.get_patient(body.patient_id)
            if existing:
                existing["latest_prediction"] = {
                    "disease": predictions[0].disease,
                    "confidence": predictions[0].confidence,
                    "risk_level": predictions[0].risk_level,
                    "predicted_at": datetime.now(timezone.utc).isoformat(),
                }
                cosmos.insert_patient(existing)
                logger.info(
                    "Updated patient %s with prediction result.", body.patient_id
                )
        except Exception as exc:
            # Non-critical: log the error but don't fail the prediction response
            logger.warning(
                "Could not update patient record %s with prediction: %s",
                body.patient_id,
                exc,
            )

    return PredictResponse(
        predictions=predictions,
        patient_id=body.patient_id,
    )
