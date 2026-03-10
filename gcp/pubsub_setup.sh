#!/usr/bin/env bash
# Step 5.4 — Create Pub/Sub topic and subscription for retail transaction events.
#
# Usage:
#   export GCP_PROJECT_ID=your-project-id
#   bash gcp/pubsub_setup.sh
#
# Requires: gcloud CLI authenticated and GCP_PROJECT_ID set.

set -euo pipefail

if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
  echo "ERROR: GCP_PROJECT_ID is not set."
  echo "Run: export GCP_PROJECT_ID=your-project-id"
  exit 1
fi

gcloud config set project "$GCP_PROJECT_ID"

echo "creating Pub/Sub topic..."
gcloud pubsub topics create retail-transactions \
  --project="$GCP_PROJECT_ID" \
  2>/dev/null && echo "  topic created: retail-transactions" \
  || echo "  topic already exists: retail-transactions"

echo "creating Pub/Sub subscription..."
gcloud pubsub subscriptions create retail-transactions-sub \
  --topic=retail-transactions \
  --ack-deadline=60 \
  --project="$GCP_PROJECT_ID" \
  2>/dev/null && echo "  subscription created: retail-transactions-sub" \
  || echo "  subscription already exists: retail-transactions-sub"

echo ""
echo "Pub/Sub setup complete."
echo "  Topic        : projects/$GCP_PROJECT_ID/topics/retail-transactions"
echo "  Subscription : projects/$GCP_PROJECT_ID/subscriptions/retail-transactions-sub"
