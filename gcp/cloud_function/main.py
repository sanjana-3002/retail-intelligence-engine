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
