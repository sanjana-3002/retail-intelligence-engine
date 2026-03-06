# Architecture Overview

## High-Level Data Flow

```
POS / E-Commerce Events
        │
        ▼
 Google Cloud Pub/Sub  ──────────────────────────────────────┐
        │                                                     │
        ▼                                                     ▼
 Cloud Function (ingest)                             Real-Time Scorer
        │                                                     │
        ▼                                                     │
  BigQuery (raw events)                                       │
        │                                                     │
        ▼                                                     │
  data/processed/  ◄──── Feature Engineering Pipeline        │
        │                                                     │
        ├── CLV Model (BG/NBD + Gamma-Gamma)                 │
        ├── Churn Model (XGBoost + SHAP)                     │
        ├── Demand Forecast (Prophet per category)            │
        ├── Uplift Model (CausalML meta-learner)              │
        └── Inventory Optimiser                               │
                │                                             │
                ▼                                             │
          models/*.pkl  ◄───────────────────────────────────-┘
                │
                ▼
         FastAPI Service  (/api)
                │
                ▼
         Dashboard  (/dashboard)
```

## Component Responsibilities

### Data Ingest Layer
- **Pub/Sub topic**: `retail-events` — receives JSON event payloads from POS terminals and e-commerce platform webhooks.
- **Cloud Function** (`gcp/cloud_function/`): validates, enriches, and writes each event to BigQuery.

### Storage Layer
- **BigQuery** (`retail_intelligence` dataset): stores raw events, aggregated customer features, and model prediction logs.
- **GCS bucket**: stores serialised model artefacts and batch prediction outputs.

### ML Layer
| Module | Algorithm | Output |
|---|---|---|
| CLV | BG/NBD + Gamma-Gamma | 90-day predicted revenue per customer |
| Churn | XGBoost classifier | Churn probability + SHAP feature importances |
| Demand | Prophet | 30-day sales forecast per category |
| Uplift | T-Learner / S-Learner | Incremental lift per promotion segment |
| Inventory | Newsvendor / Safety-stock | Reorder point + optimal order quantity |

### API Layer
- FastAPI application exposing `/predict/clv`, `/predict/churn`, `/predict/demand`, `/predict/uplift`, `/predict/inventory` endpoints.
- Models loaded once at startup; predictions served in < 50 ms p99.

### Dashboard Layer
- Plotly Dash or static HTML/JS consuming the FastAPI endpoints.
- Panels: CLV segments, churn risk heatmap, demand forecast chart, uplift waterfall, inventory status table.
