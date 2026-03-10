"""
Cloud Function — retail inference pipeline.

Triggered by: Pub/Sub topic 'retail-transactions'
Entry point : process_transaction(event, context)

On each event:
  1. Decode the Pub/Sub message (base64 → JSON)
  2. Load models from GCS on cold start, cache as module-level globals
  3. Extract per-customer features from BigQuery events history
  4. Run all 4 models: churn_xgb, cox_model, bgf+ggf, isolation_forest
  5. Write to BigQuery: events + predictions; anomalies if flagged
  6. Return HTTP 200

Environment variables required:
    GCP_PROJECT_ID
    GOOGLE_APPLICATION_CREDENTIALS  (set automatically inside Cloud Functions)
"""

import base64
import io
import json
import logging
import os
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from google.cloud import bigquery, storage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ID   = os.environ.get("GCP_PROJECT_ID", "")
DATASET_ID   = "retail_intelligence"
BUCKET_NAME  = f"retail-intelligence-models-{PROJECT_ID}"
MANIFEST_KEY = "models/latest_manifest.json"

# ---------------------------------------------------------------------------
# Module-level model cache (persists across warm invocations)
# ---------------------------------------------------------------------------
_MODELS: dict = {}


# ---------------------------------------------------------------------------
# Cold-start model loading
# ---------------------------------------------------------------------------
def _load_models() -> None:
    """Download and deserialise all models from GCS on cold start. No-op on warm start."""
    global _MODELS
    if _MODELS:
        return

    logger.info("cold start: loading models from GCS ...")
    gcs    = storage.Client()
    bucket = gcs.bucket(BUCKET_NAME)

    manifest_blob = bucket.blob(MANIFEST_KEY)
    manifest      = json.loads(manifest_blob.download_as_text())
    logger.info("manifest keys: %s", list(manifest.keys()))

    for model_name, gcs_uri in manifest.items():
        gcs_path = gcs_uri.replace(f"gs://{BUCKET_NAME}/", "")
        blob     = bucket.blob(gcs_path)
        buf      = io.BytesIO(blob.download_as_bytes())
        _MODELS[model_name] = joblib.load(buf)
        logger.info("loaded: %s", model_name)

    logger.info("all models ready: %s", list(_MODELS.keys()))


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def _get_customer_stats(bq: bigquery.Client, customer_id: str) -> dict:
    """Query BQ events table for per-customer AVG and STDDEV of revenue.
    Used to compute order_value_zscore for the anomaly model."""
    if not customer_id:
        return {"mean_revenue": 0.0, "std_revenue": 1.0}

    query = f"""
        SELECT
          AVG(revenue)    AS mean_revenue,
          STDDEV(revenue) AS std_revenue
        FROM `{PROJECT_ID}.{DATASET_ID}.events`
        WHERE customer_id = @customer_id
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("customer_id", "STRING", customer_id)
        ]
    )
    rows = list(bq.query(query, job_config=job_config).result())
    if rows and rows[0].mean_revenue is not None:
        return {
            "mean_revenue": float(rows[0].mean_revenue or 0.0),
            "std_revenue":  float(rows[0].std_revenue  or 1.0),
        }
    return {"mean_revenue": 0.0, "std_revenue": 1.0}


def _build_feature_vectors(msg: dict, customer_stats: dict):
    """Build churn and anomaly feature vectors for a single live transaction."""
    order_value        = float(msg.get("revenue", 0.0))
    mean_rev           = customer_stats["mean_revenue"]
    std_rev            = customer_stats["std_revenue"] or 1.0
    order_value_zscore = (order_value - mean_rev) / std_rev

    # Churn features — matches CHURN_FEATURES from churn_model.py.
    # For real-time scoring most per-customer aggregates are unavailable;
    # we supply the observable signals and zero-fill the rest.
    churn_features = np.array([[
        0.0,                         # recency
        1.0,                         # frequency
        order_value,                 # monetary proxy
        1.0,                         # velocity_decay_ratio
        1.0,                         # category_hhi
        0.0,                         # spend_cv
        0.0,                         # return_rate
        0.0,                         # cancellation_rate
        0.0,                         # customer_tenure_days
        float(msg.get("quantity", 1)),  # avg_items_per_order
        1.0,                         # country_count
    ]])

    # Anomaly features — matches build_anomaly_features in forecast_model.py
    ts = pd.Timestamp(msg["invoice_date"])
    anomaly_features = np.array([[
        order_value,
        order_value_zscore,
        float(msg.get("quantity", 0)),
        float(ts.hour),
        float(ts.dayofweek),
        0.0,   # is_new_country (unknown without a separate country lookup)
    ]])

    return churn_features, anomaly_features


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
CHURN_FEATURE_NAMES = [
    "recency", "frequency", "monetary", "velocity_decay_ratio",
    "category_hhi", "spend_cv", "return_rate", "cancellation_rate",
    "customer_tenure_days", "avg_items_per_order", "country_count",
]


def _run_inference(msg: dict, churn_features: np.ndarray, anomaly_features: np.ndarray) -> dict:
    """Run all available models and return a flat results dict."""
    results: dict = {}

    # 1. Churn probability (XGBoost)
    if "churn_xgb" in _MODELS:
        prob = float(_MODELS["churn_xgb"].predict_proba(churn_features)[0, 1])
        results["churn_probability"] = round(prob, 4)

    # 2. Survival days (Cox PH)
    if "cox_model" in _MODELS:
        try:
            cox_df = pd.DataFrame(churn_features, columns=CHURN_FEATURE_NAMES)
            sf     = _MODELS["cox_model"].predict_survival_function(cox_df)
            # median survival = first time survival drops below 0.5
            below_half = sf[sf.columns[0]] < 0.5
            median_days = float(below_half.idxmax()) if below_half.any() else 0.0
        except Exception as exc:
            logger.warning("cox inference failed: %s", exc)
            median_days = 0.0
        results["survival_days"] = round(median_days, 1)

    # 3. CLV — BG/NBD + Gamma-Gamma
    results["clv_90d"]  = 0.0
    results["clv_365d"] = 0.0
    if "bgf" in _MODELS and "ggf" in _MODELS:
        try:
            revenue = float(msg.get("revenue", 0.0))
            clv_90  = float(_MODELS["ggf"].customer_lifetime_value(
                _MODELS["bgf"], [1.0], [0.0], [1.0], [revenue],
                time=3, discount_rate=0.01,
            ).iloc[0])
            clv_365 = float(_MODELS["ggf"].customer_lifetime_value(
                _MODELS["bgf"], [1.0], [0.0], [1.0], [revenue],
                time=12, discount_rate=0.01,
            ).iloc[0])
            results["clv_90d"]  = round(clv_90,  2)
            results["clv_365d"] = round(clv_365, 2)
        except Exception as exc:
            logger.warning("CLV inference failed: %s", exc)

    # 4. Anomaly detection (Isolation Forest)
    if "isolation_forest" in _MODELS:
        pred  = _MODELS["isolation_forest"].predict(anomaly_features)[0]
        score = float(_MODELS["isolation_forest"].decision_function(anomaly_features)[0])
        results["anomaly_flag"]  = 1 if pred == -1 else 0
        results["anomaly_score"] = round(score, 4)

    return results
