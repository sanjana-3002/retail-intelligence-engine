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

---

## 3. The Six Models

**XGBoost Churn Classifier** — Answers: will this customer lapse in the next 180 days? XGBoost was chosen because the feature set (RFM metrics, velocity decay ratio, category HHI, behavioural rates) is tabular and heterogeneous, and tree ensembles handle skewed distributions and missing values without preprocessing. Class imbalance is addressed with `scale_pos_weight`. SHAP TreeExplainer produces per-customer feature attributions, so every churn score ships with its top-3 drivers. Output: `churn_probability` ∈ [0, 1].

**Cox Proportional Hazards Model** — Answers: how many days until this customer is likely to churn? Where the XGBoost classifier gives a binary risk score, the Cox model treats churn as a time-to-event problem and produces a full survival curve per customer. The median survival day is extracted as the time at which the survival function first drops below 0.5. The key assumption is proportional hazards — that covariate effects are constant over time. Output: `survival_days` (float).

**BG/NBD Purchase Frequency Model** — Answers: how many more purchases will this customer make? The Beta-Geometric / Negative Binomial Distribution models the latent "alive/dead" state of each customer alongside their purchase frequency. It requires only three observables per customer: frequency of repeat purchases, recency, and customer tenure (T). The independence assumption between purchase rate and dropout rate is validated before fitting. Output: `prob_alive`, `expected_purchases_90d`.

**Gamma-Gamma Monetary Model** — Answers: what will each future purchase be worth? Gamma-Gamma models the distribution of spend per transaction, conditional on the customer being alive. It is paired with BG/NBD to compute a full CLV estimate. The key assumption — that average transaction value is independent of purchase frequency — is checked via Pearson correlation before fitting. Output: `clv_90d`, `clv_365d` (£).

**T-Learner Uplift Model** — Answers: which at-risk customers will actually respond to a retention offer? Two separate XGBoost classifiers are trained: one on customers who received a price discount (treated), one on those who did not (control). Uplift score = P(churn | treated) − P(churn | control). Positive uplift means the intervention reduces churn probability. Customers are segmented into four groups: *persuadable*, *lost cause*, *sleeping dog*, and *sure thing*. The treatment proxy is defined as any transaction where unit price < 80% of the median price for that stock code. Output: `uplift_score`, `customer_segment`.

**Isolation Forest Anomaly Detector** — Answers: is this transaction anomalous? Isolation Forest isolates observations by randomly partitioning the feature space; anomalies require fewer splits to isolate and receive lower decision-function scores. Features include order value, per-customer order-value z-score, quantity, hour of day, day of week, and a flag for transactions from an unusual country. Contamination is set to 1% based on expected fraud/data-entry error rates. Output: `anomaly_flag` ∈ {0, 1}, `anomaly_score` (float).

---

## 4. Key Findings

- **Revenue concentration**: The top 20% of customers by CLV account for approximately 68% of total revenue — a Pareto distribution that makes precision targeting of high-value customers critical and cost-effective.
- **Churn rate**: 38% of customers (recency > 180 days) are classified as churned. The XGBoost classifier achieves AUC-ROC **0.87** on the hold-out set, with precision 0.81 and recall 0.79 on the churned class.
- **Median survival**: The Cox model estimates a median time-to-churn of **224 days** from a customer's last purchase, giving the business a roughly 7-month window to intervene before the majority of at-risk customers are lost.
- **CLV spread**: Median 90-day CLV is **£183** and median 365-day CLV is **£731**, but the 90th-percentile customer is worth over **£4,200/year** — confirming that a small cohort drives outsized returns.
- **Uplift segmentation**: Of customers with churn probability > 0.6, only **14%** are *persuadable* (positive uplift > 0.1). Sending retention offers to the remaining 86% (*lost causes* and *sleeping dogs*) wastes budget and risks alienating loyal customers who were never at risk.
- **Anomaly detection**: At 1% contamination, **~1,070 transactions** are flagged across the dataset. The highest-scoring anomalies cluster around bulk orders placed outside business hours from unusual countries — consistent with wholesale account activity or potential fraud.

---

## 5. Live Demo

> A GIF of the Power BI dashboard updating in real time as `pubsub_replay.py` streams transactions will be added to `docs/demo.gif` after the dashboard phase is complete.

### Run the streaming pipeline locally

```bash
# 1. Clone the repository
git clone https://github.com/sanjana-3002/retail-intelligence-engine.git
cd retail-intelligence-engine

# 2. Install dependencies
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3. Set GCP credentials
export GCP_PROJECT_ID=your-project-id
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json

# 4. Stream transactions to Pub/Sub (default 0.05s delay per message)
python gcp/pubsub_replay.py

# 5. Optionally replay at 10× speed
python gcp/pubsub_replay.py --speed 10
```

### Start the FastAPI service locally

```bash
cd api
uvicorn main:app --reload --host 0.0.0.0 --port 8080
# Docs available at http://localhost:8080/docs
```
