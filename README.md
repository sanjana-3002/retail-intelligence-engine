# Retail Intelligence Engine

A real-time customer analytics platform built on Google Cloud Platform that streams retail transactions through a machine learning inference pipeline, persists predictions to BigQuery, and surfaces insights via a FastAPI service and Power BI dashboards.

---

## 1. The Business Problem

Retail businesses lose significant revenue each year to preventable churn — customers who quietly stop buying with no signal until it is too late to act. Identifying which customers are about to lapse, quantifying their remaining lifetime value, and distinguishing customers who will respond to a retention offer from those who will not requires combining multiple modelling disciplines that most retail analytics stacks treat separately. Without an integrated, real-time system, analysts are left running batch reports on last week's data while at-risk customers make their final purchase undetected.

---

## 2. System Architecture

![System Architecture](docs/architecture.png)

The pipeline follows a streaming-first design. Historical transactions are replayed through Pub/Sub to simulate a live event stream; in production this topic would be fed by a POS or e-commerce webhook.

```
all_transactions.csv
        │
        ▼
  pubsub_replay.py  ──► Pub/Sub  (retail-transactions)
                                │
                                ▼
                     Cloud Function  ─── GCS (model artefacts)
                       ├─ base64 decode + JSON parse
                       ├─ cold-start: load models via manifest
                       ├─ feature extraction from BQ events table
                       ├─ run 4 models in sequence
                       └─ write to BQ predictions / anomalies
                                │
                   ┌────────────┼────────────┐
                   ▼            ▼            ▼
              BQ events   BQ predictions  BQ anomalies
                   │            │            │
                   └────────────┼────────────┘
                                ▼
                    FastAPI service  (Cloud Run)
                                │
                                ▼
                           Power BI
                    (DirectQuery → BigQuery)
```

**GCS bucket** (`retail-intelligence-models-{PROJECT_ID}`) stores serialised model artefacts and a `latest_manifest.json` that maps each model name to its GCS URI. Both the Cloud Function and the FastAPI service resolve models exclusively from this manifest, keeping every compute layer stateless and independently deployable.
