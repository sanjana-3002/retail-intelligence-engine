#!/usr/bin/env bash
# Step 6.3 — Deploy the retail-inference Cloud Function.
#
# Usage:
#   export GCP_PROJECT_ID=your-project-id
#   bash gcp/deploy_function.sh
#
# Prerequisites:
#   - gcloud CLI authenticated
#   - GCP_PROJECT_ID set
#   - Service account retail-intelligence-sa exists with required IAM roles:
#       roles/bigquery.dataEditor
#       roles/storage.objectViewer
#       roles/pubsub.subscriber

set -euo pipefail

if [[ -z "${GCP_PROJECT_ID:-}" ]]; then
  echo "ERROR: GCP_PROJECT_ID is not set."
  echo "Run: export GCP_PROJECT_ID=your-project-id"
  exit 1
fi

SERVICE_ACCOUNT="retail-intelligence-sa@${GCP_PROJECT_ID}.iam.gserviceaccount.com"

echo "deploying retail-inference Cloud Function ..."
echo "  project : $GCP_PROJECT_ID"
echo "  SA      : $SERVICE_ACCOUNT"
echo ""

cd "$(dirname "$0")/cloud_function"

gcloud functions deploy retail-inference \
  --runtime python311 \
  --trigger-topic retail-transactions \
  --entry-point process_transaction \
  --memory 512MB \
  --timeout 60s \
  --service-account "$SERVICE_ACCOUNT" \
  --set-env-vars "GCP_PROJECT_ID=${GCP_PROJECT_ID}" \
  --region us-central1 \
  --project "$GCP_PROJECT_ID"

echo ""
echo "deployment complete."
echo "  function URL: https://us-central1-${GCP_PROJECT_ID}.cloudfunctions.net/retail-inference"
