#!/bin/bash

# Setup script for Imagen 3 with GCS permissions
# This grants the Cloud Run service account the necessary permissions

set -e

# Get project ID from gcloud config
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    echo "Error: No project ID found. Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)" 2>/dev/null)

if [ -z "$PROJECT_NUMBER" ]; then
    echo "Error: Could not get project number for $PROJECT_ID"
    exit 1
fi

SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

echo "Setting up Imagen 3 permissions for project: $PROJECT_ID"
echo "Service Account: $SERVICE_ACCOUNT"
echo ""

# Grant Storage Object Admin role (for GCS upload/download/delete)
echo "Granting Storage Object Admin role..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/storage.objectAdmin" \
    --condition=None

# Grant Vertex AI User role (for Imagen 3 API)
echo "Granting Vertex AI User role..."
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:${SERVICE_ACCOUNT}" \
    --role="roles/aiplatform.user" \
    --condition=None

echo ""
echo "✓ Permissions granted successfully!"
echo ""
echo "The Cloud Run service account can now:"
echo "  - Upload/download/delete files in GCS"
echo "  - Use Vertex AI Imagen models"
echo ""
echo "Next steps:"
echo "  1. The GCS bucket will be created automatically on first use"
echo "  2. Redeploy your application: ./deploy.sh us-central1"
echo "  3. Imagen 3 will be used for background replacement"

