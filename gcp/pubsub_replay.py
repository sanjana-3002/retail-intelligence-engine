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
