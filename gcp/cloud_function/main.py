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
