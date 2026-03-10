"""
Upload trained model artefacts from models/ to Google Cloud Storage.

Usage:
    python gcp/upload_models.py

Requires:
    GOOGLE_APPLICATION_CREDENTIALS env var pointing to a service account key.
    GCP_PROJECT_ID env var (or set PROJECT_ID below directly).
"""

import os
import json
from datetime import date
from pathlib import Path

from google.cloud import storage
from google.api_core.exceptions import Conflict

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROJECT_ID = os.environ.get("GCP_PROJECT_ID", "")
BUCKET_NAME = f"retail-intelligence-models-{PROJECT_ID}"
REGION = "us-central1"
MODELS_DIR = Path("models")
GCS_PREFIX = f"models/{date.today().isoformat()}"


def get_or_create_bucket(client: storage.Client, bucket_name: str, region: str) -> storage.Bucket:
    """Return the bucket, creating it in the given region if it doesn't exist."""
    try:
        bucket = client.create_bucket(bucket_name, location=region)
        print(f"bucket created: gs://{bucket_name}  (region={region})")
    except Conflict:
        bucket = client.bucket(bucket_name)
        print(f"bucket already exists: gs://{bucket_name}")
    return bucket


def upload_models(bucket: storage.Bucket, models_dir: Path, prefix: str) -> dict:
    """Upload all .pkl files from models_dir to GCS under prefix. Returns manifest dict."""
    pkl_files = sorted(models_dir.glob("*.pkl"))
    if not pkl_files:
        print(f"no .pkl files found in {models_dir}/")
        return {}

    manifest = {}
    for local_path in pkl_files:
        gcs_path = f"{prefix}/{local_path.name}"
        blob = bucket.blob(gcs_path)
        blob.upload_from_filename(str(local_path))
        gcs_uri = f"gs://{bucket.name}/{gcs_path}"
        manifest[local_path.stem] = gcs_uri
        print(f"  uploaded {local_path.name}  →  {gcs_uri}")

    return manifest


def upload_manifest(bucket: storage.Bucket, manifest: dict) -> None:
    """Write manifest as JSON and upload to models/latest_manifest.json."""
    blob = bucket.blob("models/latest_manifest.json")
    blob.upload_from_string(
        json.dumps(manifest, indent=2),
        content_type="application/json",
    )
    print(f"\nmanifest uploaded → gs://{bucket.name}/models/latest_manifest.json")
    print(json.dumps(manifest, indent=2))


def main():
    if not PROJECT_ID:
        raise EnvironmentError(
            "GCP_PROJECT_ID environment variable is not set. "
            "Export it before running this script."
        )

    client = storage.Client(project=PROJECT_ID)
    bucket = get_or_create_bucket(client, BUCKET_NAME, REGION)

    print(f"\nuploading models from {MODELS_DIR}/ with prefix '{GCS_PREFIX}' ...")
    manifest = upload_models(bucket, MODELS_DIR, GCS_PREFIX)

    if manifest:
        upload_manifest(bucket, manifest)
        print(f"\n{len(manifest)} model(s) uploaded successfully.")
    else:
        print("nothing to upload.")


if __name__ == "__main__":
    main()
