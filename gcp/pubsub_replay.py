"""
Replay all_transactions.csv to the retail-transactions Pub/Sub topic.

Simulates a live event stream for end-to-end pipeline testing.

Usage:
    python gcp/pubsub_replay.py [--speed MULTIPLIER]

    --speed  float  Speed multiplier for publish delay.
                    Default 1.0 = 0.05s per message.
                    Use 10.0 for faster replay, 0.1 for slower.

Requires:
    GOOGLE_APPLICATION_CREDENTIALS env var.
    GCP_PROJECT_ID env var.
"""

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd
from google.cloud import pubsub_v1

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ID    = os.environ.get("GCP_PROJECT_ID", "")
TOPIC_ID      = "retail-transactions"
DATA_PATH     = Path("data/processed/all_transactions.csv")
DEFAULT_DELAY = 0.05   # seconds between messages (speed multiplier = 1.0)
MAX_RETRIES   = 3


def load_transactions(path: Path) -> pd.DataFrame:
    """Load and sort all_transactions.csv by invoice_date ascending."""
    df = pd.read_csv(path, parse_dates=["invoice_date"])
    df = df.sort_values("invoice_date").reset_index(drop=True)
    df["category"] = df["stock_code"].astype(str).str[:2]
    print(
        f"loaded {len(df):,} transactions  "
        f"({df['invoice_date'].min().date()} → {df['invoice_date'].max().date()})"
    )
    return df


def build_message(row: dict) -> bytes:
    """Serialise a transaction row to a JSON-encoded bytes payload."""
    payload = {
        "invoice_no":   str(row.get("invoice_no", "")),
        "customer_id":  str(row["customer_id"]) if pd.notna(row.get("customer_id")) else None,
        "stock_code":   str(row.get("stock_code", "")),
        "quantity":     int(row.get("quantity", 0)),
        "unit_price":   float(row.get("unit_price", 0.0)),
        "revenue":      float(row.get("revenue", 0.0)),
        "country":      str(row.get("country", "")),
        "invoice_date": pd.Timestamp(row["invoice_date"]).isoformat(),
        "category":     str(row.get("category", "")),
    }
    return json.dumps(payload).encode("utf-8")


def publish_with_retry(
    publisher: pubsub_v1.PublisherClient,
    topic_path: str,
    data: bytes,
    max_retries: int = MAX_RETRIES,
) -> bool:
    """Publish a single message with up to max_retries attempts. Returns True on success."""
    for attempt in range(1, max_retries + 1):
        try:
            future = publisher.publish(topic_path, data=data)
            future.result(timeout=10)
            return True
        except Exception as exc:
            if attempt == max_retries:
                print(f"  skipped after {max_retries} attempts: {exc}")
                return False
            time.sleep(0.5 * attempt)
    return False


def replay(speed: float = 1.0) -> None:
    """Stream all transactions to Pub/Sub at the configured speed."""
    if not PROJECT_ID:
        raise EnvironmentError("GCP_PROJECT_ID environment variable is not set.")

    df = load_transactions(DATA_PATH)
    publisher = pubsub_v1.PublisherClient()
    topic_path = publisher.topic_path(PROJECT_ID, TOPIC_ID)
    delay = DEFAULT_DELAY / speed

    published = 0
    skipped   = 0

    print(f"\nstarting replay → {topic_path}  (delay={delay:.3f}s per message)")

    for i, row in df.iterrows():
        data = build_message(row.to_dict())
        ok = publish_with_retry(publisher, topic_path, data)
        if ok:
            published += 1
        else:
            skipped += 1

        if (i + 1) % 1000 == 0:
            print(f"  {i + 1:,} processed  |  published={published:,}  skipped={skipped}")

        time.sleep(delay)

    print(f"\nreplay complete — published={published:,}  skipped={skipped}  total={len(df):,}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay transactions to Pub/Sub.")
    parser.add_argument(
        "--speed", type=float, default=1.0,
        help="Speed multiplier (default 1.0 = 0.05s delay). Higher = faster.",
    )
    args = parser.parse_args()
    replay(speed=args.speed)


if __name__ == "__main__":
    main()
