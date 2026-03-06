# Data Dictionary

## Raw Event Schema (Pub/Sub payload)

| Field | Type | Description |
|---|---|---|
| `event_id` | STRING | UUID for deduplication |
| `event_type` | STRING | `purchase`, `return`, `page_view`, `cart_add` |
| `customer_id` | STRING | Anonymised customer identifier |
| `timestamp` | TIMESTAMP | Event timestamp (UTC) |
| `store_id` | STRING | Store / channel identifier |
| `product_id` | STRING | SKU code |
| `category` | STRING | Product category |
| `quantity` | INTEGER | Units in transaction |
| `unit_price` | FLOAT | Price per unit (GBP) |
| `total_value` | FLOAT | `quantity × unit_price` |
| `session_id` | STRING | Web/app session identifier (nullable) |

## Processed Customer Features

| Feature | Type | Description |
|---|---|---|
| `customer_id` | STRING | Primary key |
| `frequency` | INTEGER | Number of repeat transactions |
| `recency` | FLOAT | Days since last purchase |
| `T` | FLOAT | Customer age in days (first purchase → today) |
| `monetary_value` | FLOAT | Average order value |
| `clv_90d` | FLOAT | Predicted 90-day revenue (BG/NBD model output) |
| `churn_prob` | FLOAT | Probability of churning within 90 days |
| `segment` | STRING | `high_value`, `at_risk`, `new`, `lapsed` |

## Processed Product/Category Features

| Feature | Type | Description |
|---|---|---|
| `product_id` | STRING | Primary key |
| `category` | STRING | Product category |
| `ds` | DATE | Date (Prophet date column) |
| `y` | FLOAT | Daily units sold (Prophet target column) |
| `lead_time_days` | INTEGER | Supplier lead time |
| `holding_cost` | FLOAT | Daily holding cost per unit |
| `stockout_cost` | FLOAT | Estimated lost-sale cost per unit |
| `reorder_point` | FLOAT | Safety-stock reorder trigger |
| `order_quantity` | FLOAT | Optimal order quantity |

## BigQuery Tables

| Table | Description |
|---|---|
| `retail_intelligence.raw_events` | Immutable event log from Pub/Sub |
| `retail_intelligence.customer_features` | Daily snapshot of customer-level features |
| `retail_intelligence.product_features` | Daily product / inventory feature snapshot |
| `retail_intelligence.predictions` | Model prediction log with timestamp |
| `retail_intelligence.experiments` | Uplift experiment assignment and outcome log |
