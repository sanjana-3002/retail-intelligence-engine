#!/usr/bin/env bash
# Deploy retail-api to Cloud Run
set -euo pipefail

PROJECT_ID="${GCP_PROJECT_ID:?GCP_PROJECT_ID not set}"
IMAGE="gcr.io/${PROJECT_ID}/retail-api"
SERVICE_ACCOUNT="retail-intelligence-sa@${PROJECT_ID}.iam.gserviceaccount.com"

cd "$(dirname "$0")"

echo "==> Building and pushing image: ${IMAGE}"
gcloud builds submit --tag "${IMAGE}"

echo "==> Deploying to Cloud Run"
gcloud run deploy retail-api \
  --image "${IMAGE}" \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 1Gi \
  --service-account "${SERVICE_ACCOUNT}" \
  --set-env-vars "GCP_PROJECT_ID=${PROJECT_ID}"

echo ""
echo "==> Health check"
SERVICE_URL=$(gcloud run services describe retail-api \
  --platform managed --region us-central1 \
  --format "value(status.url)")
curl "${SERVICE_URL}/health"
