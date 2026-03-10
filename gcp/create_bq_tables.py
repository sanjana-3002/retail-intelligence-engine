"""
Create BigQuery dataset and tables for the retail intelligence platform.

Usage:
    python gcp/create_bq_tables.py

Tables created:
    retail_intelligence.predictions  — per-customer model scores
    retail_intelligence.events       — raw transaction events
    retail_intelligence.anomalies    — flagged anomalous transactions
    retail_intelligence.forecasts    — demand forecast outputs

Requires:
    GOOGLE_APPLICATION_CREDENTIALS env var pointing to a service account key.
    GCP_PROJECT_ID env var.
"""

import os

from google.cloud import bigquery
from google.api_core.exceptions import Conflict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
DATASET_ID = "retail_intelligence"
REGION = "us-central1"

# ---------------------------------------------------------------------------
# Table schemas
# ---------------------------------------------------------------------------
SCHEMA_PREDICTIONS = [
    bigquery.SchemaField("customer_id",        "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("timestamp",          "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("churn_probability",  "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("clv_90d",            "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("clv_365d",           "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("survival_days",      "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("uplift_score",       "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("customer_segment",   "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("top_shap_driver_1",  "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("top_shap_driver_2",  "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("top_shap_driver_3",  "STRING",    mode="NULLABLE"),
]

SCHEMA_EVENTS = [
    bigquery.SchemaField("invoice_no",   "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("customer_id",  "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("timestamp",    "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("revenue",      "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("quantity",     "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("stock_code",   "STRING",    mode="NULLABLE"),
    bigquery.SchemaField("country",      "STRING",    mode="NULLABLE"),
]

SCHEMA_ANOMALIES = [
    bigquery.SchemaField("invoice_no",           "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("timestamp",            "TIMESTAMP", mode="REQUIRED"),
    bigquery.SchemaField("anomaly_score",        "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("anomaly_flag",         "INTEGER",   mode="NULLABLE"),
    bigquery.SchemaField("order_value",          "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("order_value_zscore",   "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("customer_id",          "STRING",    mode="NULLABLE"),
]

SCHEMA_FORECASTS = [
    bigquery.SchemaField("category",         "STRING",    mode="REQUIRED"),
    bigquery.SchemaField("forecast_date",    "DATE",      mode="REQUIRED"),
    bigquery.SchemaField("sarima_forecast",  "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("prophet_forecast", "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("lower_ci",         "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("upper_ci",         "FLOAT",     mode="NULLABLE"),
    bigquery.SchemaField("created_at",       "TIMESTAMP", mode="NULLABLE"),
]

TABLES = {
    "predictions": SCHEMA_PREDICTIONS,
    "events":      SCHEMA_EVENTS,
    "anomalies":   SCHEMA_ANOMALIES,
    "forecasts":   SCHEMA_FORECASTS,
}


def get_or_create_dataset(client: bigquery.Client, dataset_id: str, region: str) -> bigquery.Dataset:
    """Return the dataset, creating it in the given region if it doesn't exist."""
    dataset_ref = bigquery.Dataset(f"{client.project}.{dataset_id}")
    dataset_ref.location = region

    try:
        dataset = client.create_dataset(dataset_ref)
        print(f"dataset created: {client.project}.{dataset_id}  (region={region})")
    except Conflict:
        dataset = client.get_dataset(dataset_ref)
        print(f"dataset already exists: {client.project}.{dataset_id}")

    return dataset


def create_table(
    client: bigquery.Client,
    dataset_id: str,
    table_id: str,
    schema: list,
) -> None:
    """Create a BigQuery table if it doesn't already exist."""
    table_ref = f"{client.project}.{dataset_id}.{table_id}"
    table = bigquery.Table(table_ref, schema=schema)

    try:
        client.create_table(table)
        print(f"  created  : {table_ref}")
    except Conflict:
        print(f"  exists   : {table_ref}")


def main():
    if not PROJECT_ID:
        raise EnvironmentError(
            "GCP_PROJECT_ID environment variable is not set. "
            "Export it before running this script."
        )

    client = bigquery.Client(project=PROJECT_ID)

    get_or_create_dataset(client, DATASET_ID, REGION)

    print(f"\ncreating tables in {PROJECT_ID}.{DATASET_ID} ...")
    for table_name, schema in TABLES.items():
        create_table(client, DATASET_ID, table_name, schema)

    print(f"\ndone — {len(TABLES)} table(s) ready.")


if __name__ == "__main__":
    main()
