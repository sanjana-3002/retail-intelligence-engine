"""
FastAPI application — retail intelligence inference API.

Startup:
  - Load all models from GCS (via latest_manifest.json)
  - Build SHAP TreeExplainer from churn_xgb
  - Load full_customer_intelligence.csv from GCS into memory

Endpoints:
  POST /predict/churn
  POST /predict/clv
  GET  /customer/{customer_id}/profile
  GET  /anomalies/recent
  GET  /forecast/{category}
  GET  /health

Environment variables required:
    GCP_PROJECT_ID
    GOOGLE_APPLICATION_CREDENTIALS  (set automatically on Cloud Run)
"""

import io
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException, Query
from google.cloud import bigquery, storage
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ID   = os.environ.get("GCP_PROJECT_ID", "")
DATASET_ID   = "retail_intelligence"
BUCKET_NAME  = f"retail-intelligence-models-{PROJECT_ID}"
MANIFEST_KEY = "models/latest_manifest.json"
INTEL_CSV_KEY = "data/full_customer_intelligence.csv"

CHURN_FEATURE_NAMES = [
    "recency", "frequency", "monetary",
    "velocity_decay_ratio", "category_hhi", "spend_cv",
    "return_rate", "cancellation_rate",
    "customer_tenure_days", "avg_items_per_order", "country_count",
]

# ---------------------------------------------------------------------------
# Module-level globals (loaded once at startup)
# ---------------------------------------------------------------------------
_MODELS: dict = {}
_SHAP_EXPLAINER = None
_CUSTOMER_INTEL: Optional[pd.DataFrame] = None
_MANIFEST: dict = {}


# ---------------------------------------------------------------------------
# Startup / lifespan
# ---------------------------------------------------------------------------
def _load_globals() -> None:
    global _MODELS, _SHAP_EXPLAINER, _CUSTOMER_INTEL, _MANIFEST

    gcs    = storage.Client()
    bucket = gcs.bucket(BUCKET_NAME)

    # Load manifest
    manifest_blob = bucket.blob(MANIFEST_KEY)
    _MANIFEST = json.loads(manifest_blob.download_as_text())
    logger.info("manifest keys: %s", list(_MANIFEST.keys()))

    # Load all models
    for model_name, gcs_uri in _MANIFEST.items():
        gcs_path = gcs_uri.replace(f"gs://{BUCKET_NAME}/", "")
        blob     = bucket.blob(gcs_path)
        buf      = io.BytesIO(blob.download_as_bytes())
        _MODELS[model_name] = joblib.load(buf)
        logger.info("loaded model: %s", model_name)

    # Build SHAP explainer from churn model
    if "churn_xgb" in _MODELS:
        _SHAP_EXPLAINER = shap.TreeExplainer(_MODELS["churn_xgb"])
        logger.info("SHAP TreeExplainer ready")

    # Load full customer intelligence CSV
    intel_blob  = bucket.blob(INTEL_CSV_KEY)
    intel_bytes = intel_blob.download_as_bytes()
    _CUSTOMER_INTEL = pd.read_csv(io.BytesIO(intel_bytes))
    logger.info("full_customer_intelligence loaded: %d rows", len(_CUSTOMER_INTEL))


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_globals()
    yield


app = FastAPI(title="Retail Intelligence API", version="1.0.0", lifespan=lifespan)


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------
class ChurnRequest(BaseModel):
    customer_id: str
    recency: float = 0.0
    frequency: float = 1.0
    monetary: float = 0.0
    velocity_decay_ratio: float = 1.0
    category_hhi: float = 1.0
    spend_cv: float = 0.0
    return_rate: float = 0.0
    cancellation_rate: float = 0.0
    customer_tenure_days: float = 0.0
    avg_items_per_order: float = 1.0
    country_count: float = 1.0


class CLVRequest(BaseModel):
    customer_id: str
    frequency_repeat: float
    recency_bgnbd: float
    T_bgnbd: float
    monetary: float


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_uplift(customer_id: str) -> float:
    """Return uplift_score from full_customer_intelligence for a given customer."""
    if _CUSTOMER_INTEL is None:
        return 0.0
    row = _CUSTOMER_INTEL[_CUSTOMER_INTEL["customer_id"] == customer_id]
    if row.empty:
        return 0.0
    val = row.iloc[0].get("uplift_score", 0.0)
    return float(val) if pd.notna(val) else 0.0


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/predict/churn")
def predict_churn(req: ChurnRequest):
    if "churn_xgb" not in _MODELS:
        raise HTTPException(status_code=503, detail="churn_xgb model not loaded")

    features = np.array([[
        req.recency, req.frequency, req.monetary,
        req.velocity_decay_ratio, req.category_hhi, req.spend_cv,
        req.return_rate, req.cancellation_rate,
        req.customer_tenure_days, req.avg_items_per_order, req.country_count,
    ]])

    churn_prob = float(_MODELS["churn_xgb"].predict_proba(features)[0, 1])

    # Cox median survival days
    survival_days = None
    if "cox_model" in _MODELS:
        try:
            cox_df = pd.DataFrame(features, columns=CHURN_FEATURE_NAMES)
            sf = _MODELS["cox_model"].predict_survival_function(cox_df)
            below_half = sf[sf.columns[0]] < 0.5
            survival_days = float(below_half.idxmax()) if below_half.any() else None
        except Exception as exc:
            logger.warning("cox inference failed: %s", exc)

    # Top-3 SHAP drivers
    top_shap_drivers = []
    if _SHAP_EXPLAINER is not None:
        cust_df = pd.DataFrame(features, columns=CHURN_FEATURE_NAMES)
        sv      = _SHAP_EXPLAINER.shap_values(cust_df)
        sv_flat = np.array(sv).flatten()
        pairs   = sorted(zip(CHURN_FEATURE_NAMES, sv_flat), key=lambda x: abs(x[1]), reverse=True)
        top_shap_drivers = [
            {"feature": f, "shap_value": round(float(v), 4)} for f, v in pairs[:3]
        ]

    # Recommended action
    uplift = _get_uplift(req.customer_id)
    if churn_prob > 0.6 and uplift > 0.1:
        recommended_action = "Send retention offer"
    elif churn_prob > 0.6:
        recommended_action = "Monitor only"
    else:
        recommended_action = "No action"

    return {
        "customer_id":        req.customer_id,
        "churn_probability":  round(churn_prob, 4),
        "survival_days":      round(survival_days, 1) if survival_days is not None else None,
        "top_shap_drivers":   top_shap_drivers,
        "recommended_action": recommended_action,
    }


@app.post("/predict/clv")
def predict_clv(req: CLVRequest):
    if "bgf" not in _MODELS or "ggf" not in _MODELS:
        raise HTTPException(status_code=503, detail="bgf/ggf models not loaded")

    bgf = _MODELS["bgf"]
    ggf = _MODELS["ggf"]

    prob_alive = float(bgf.conditional_probability_alive(
        req.frequency_repeat, req.recency_bgnbd, req.T_bgnbd,
    ))

    expected_purchases_90d = float(bgf.conditional_expected_number_of_purchases_up_to_time(
        90, req.frequency_repeat, req.recency_bgnbd, req.T_bgnbd,
    ))

    clv_90d = float(ggf.customer_lifetime_value(
        bgf,
        [req.frequency_repeat], [req.recency_bgnbd], [req.T_bgnbd], [req.monetary],
        time=3, discount_rate=0.01,
    ).iloc[0])

    clv_365d = float(ggf.customer_lifetime_value(
        bgf,
        [req.frequency_repeat], [req.recency_bgnbd], [req.T_bgnbd], [req.monetary],
        time=12, discount_rate=0.01,
    ).iloc[0])

    return {
        "customer_id":            req.customer_id,
        "clv_90d":                round(clv_90d, 2),
        "clv_365d":               round(clv_365d, 2),
        "prob_alive":             round(prob_alive, 4),
        "expected_purchases_90d": round(expected_purchases_90d, 4),
    }


@app.get("/customer/{customer_id}/profile")
def customer_profile(customer_id: str):
    if _CUSTOMER_INTEL is None:
        raise HTTPException(status_code=503, detail="customer intelligence not loaded")

    row = _CUSTOMER_INTEL[_CUSTOMER_INTEL["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"customer '{customer_id}' not found")

    profile = row.iloc[0].where(pd.notna(row.iloc[0]), None).to_dict()

    # Fetch anomaly history from BigQuery
    bq = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT invoice_no, timestamp, anomaly_score, order_value
        FROM `{PROJECT_ID}.{DATASET_ID}.anomalies`
        WHERE customer_id = @customer_id
        ORDER BY timestamp DESC
        LIMIT 10
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id),
        ]
    )
    anomaly_history = [dict(r) for r in bq.query(query, job_config=job_config).result()]

    return {**profile, "anomaly_history": anomaly_history}


@app.get("/anomalies/recent")
def recent_anomalies(
    limit:     int   = Query(100, ge=1, le=1000),
    min_score: float = Query(0.5),
):
    bq = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.anomalies`
        WHERE anomaly_score >= @min_score
        ORDER BY timestamp DESC
        LIMIT @limit
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("min_score", "FLOAT64", min_score),
            bigquery.ScalarQueryParameter("limit",     "INT64",   limit),
        ]
    )
    rows = [dict(r) for r in bq.query(query, job_config=job_config).result()]
    return {"anomalies": rows, "count": len(rows)}


@app.get("/forecast/{category}")
def forecast_category(category: str):
    bq = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT forecast_date, sarima_forecast, prophet_forecast, lower_ci, upper_ci
        FROM `{PROJECT_ID}.{DATASET_ID}.forecasts`
        WHERE category = @category
        ORDER BY forecast_date
        LIMIT 12
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("category", "STRING", category),
        ]
    )
    rows = [dict(r) for r in bq.query(query, job_config=job_config).result()]
    if not rows:
        raise HTTPException(status_code=404, detail=f"no forecast found for category '{category}'")
    return {"category": category, "forecast": rows}


@app.get("/health")
def health():
    bq = bigquery.Client(project=PROJECT_ID)
    query = f"""
        SELECT MAX(timestamp) AS last_prediction_timestamp
        FROM `{PROJECT_ID}.{DATASET_ID}.predictions`
    """
    rows = list(bq.query(query).result())
    last_ts = None
    if rows and rows[0].last_prediction_timestamp:
        last_ts = rows[0].last_prediction_timestamp.isoformat()

    return {
        "status":                    "ok",
        "model_versions":            _MANIFEST,
        "last_prediction_timestamp": last_ts,
    }
