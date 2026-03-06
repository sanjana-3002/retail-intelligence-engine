# Real-Time Retail Intelligence Engine

A production-grade data science platform for retail analytics, combining real-time event streaming, advanced ML models, and a REST API to deliver actionable intelligence across customer behaviour, inventory, demand forecasting, and causal experimentation.

## Project Overview

This engine ingests live point-of-sale and e-commerce events via Google Cloud Pub/Sub, stores structured data in BigQuery, and surfaces predictions through a FastAPI service. Key capabilities:

- **Customer Lifetime Value (CLV)** — BG/NBD + Gamma-Gamma probabilistic models
- **Churn Prediction** — Gradient boosted classifier with SHAP explainability
- **Demand Forecasting** — Prophet time-series model per product category
- **Causal Uplift Modelling** — CausalML meta-learners for promotion ROI
- **Inventory Optimisation** — Safety-stock and reorder-point engine
- **Real-Time Streaming** — Cloud Pub/Sub ingest → BigQuery sink → live scoring

## Repository Structure

```
retail-intelligence-engine/
├── data/
│   ├── raw/               # Raw source data (git-ignored)
│   └── processed/         # Cleaned & feature-engineered datasets
├── notebooks/             # Exploratory analysis and model development
├── models/                # Serialised model artefacts (.pkl, .json)
├── gcp/
│   ├── cloud_function/    # GCP Cloud Function source code
│   └── bq_schemas/        # BigQuery table schema definitions
├── api/                   # FastAPI application
├── dashboard/             # Visualisation / front-end assets
├── docs/                  # Architecture diagrams and documentation
├── requirements.txt       # Python dependencies
└── README.md
```

## Setup

```bash
# Clone the repository
git clone https://github.com/sanjana-3002/retail-intelligence-engine.git
cd retail-intelligence-engine

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Copy and configure environment variables
cp .env.example .env
```

## Environment Variables

| Variable | Description |
|---|---|
| `GCP_PROJECT_ID` | Google Cloud project ID |
| `PUBSUB_TOPIC` | Pub/Sub topic for event ingest |
| `BQ_DATASET` | BigQuery dataset name |
| `MODEL_BUCKET` | GCS bucket for model artefacts |

## Tech Stack

- **Languages**: Python 3.11
- **ML / Stats**: scikit-learn, XGBoost, SHAP, Prophet, CausalML, lifetimes, lifelines, statsmodels
- **Cloud**: Google Cloud Platform (Pub/Sub, BigQuery, Cloud Storage, Cloud Functions)
- **API**: FastAPI + Uvicorn
- **Visualisation**: Plotly, Seaborn, Matplotlib

## License

MIT
