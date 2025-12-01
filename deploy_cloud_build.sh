#!/bin/bash

# Alternative deployment using Cloud Build (no Docker auth needed)
# This method builds in the cloud, so you don't need to authenticate Docker locally

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Passport Pro - Cloud Build Deployment${NC}"
echo "=========================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed.${NC}"
    exit 1
fi

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}No project ID found.${NC}"
    read -p "Enter your Google Cloud Project ID: " PROJECT_ID
    gcloud config set project $PROJECT_ID
fi

echo -e "${GREEN}Using project: ${PROJECT_ID}${NC}"

# Set region
REGION=${1:-us-central1}
echo -e "${GREEN}Using region: ${REGION}${NC}"

# Enable required APIs
echo -e "${YELLOW}Enabling required Google Cloud APIs...${NC}"
gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    containerregistry.googleapis.com \
    aiplatform.googleapis.com \
    secretmanager.googleapis.com \
    --project=$PROJECT_ID

# Build using Cloud Build (no local Docker needed)
echo -e "${YELLOW}Building Docker image using Cloud Build...${NC}"
gcloud builds submit --tag gcr.io/$PROJECT_ID/passport-pro:latest --project=$PROJECT_ID

# Check if secrets exist and prepare secret mounts
SECRET_ARGS=""
if gcloud secrets describe google-api-key --project=$PROJECT_ID &>/dev/null 2>&1; then
    SECRET_ARGS="--update-secrets=GOOGLE_API_KEY=google-api-key:latest"
    echo -e "${GREEN}✓ Found google-api-key secret${NC}"
fi

if gcloud secrets describe secret-key --project=$PROJECT_ID &>/dev/null 2>&1; then
    if [ -n "$SECRET_ARGS" ]; then
        SECRET_ARGS="$SECRET_ARGS,SECRET_KEY=secret-key:latest"
    else
        SECRET_ARGS="--update-secrets=SECRET_KEY=secret-key:latest"
    fi
    echo -e "${GREEN}✓ Found secret-key secret${NC}"
fi

# Email notification secrets
if gcloud secrets describe admin-email --project=$PROJECT_ID &>/dev/null 2>&1; then
    if [ -n "$SECRET_ARGS" ]; then
        SECRET_ARGS="$SECRET_ARGS,ADMIN_EMAIL=admin-email:latest"
    else
        SECRET_ARGS="--update-secrets=ADMIN_EMAIL=admin-email:latest"
    fi
    echo -e "${GREEN}✓ Found admin-email secret${NC}"
fi

if gcloud secrets describe smtp-username --project=$PROJECT_ID &>/dev/null 2>&1; then
    if [ -n "$SECRET_ARGS" ]; then
        SECRET_ARGS="$SECRET_ARGS,SMTP_USERNAME=smtp-username:latest"
    else
        SECRET_ARGS="--update-secrets=SMTP_USERNAME=smtp-username:latest"
    fi
    echo -e "${GREEN}✓ Found smtp-username secret${NC}"
fi

if gcloud secrets describe smtp-password --project=$PROJECT_ID &>/dev/null 2>&1; then
    if [ -n "$SECRET_ARGS" ]; then
        SECRET_ARGS="$SECRET_ARGS,SMTP_PASSWORD=smtp-password:latest"
    else
        SECRET_ARGS="--update-secrets=SMTP_PASSWORD=smtp-password:latest"
    fi
    echo -e "${GREEN}✓ Found smtp-password secret${NC}"
fi

# Optional: SMTP server and port (can also be in env vars)
if gcloud secrets describe smtp-server --project=$PROJECT_ID &>/dev/null 2>&1; then
    if [ -n "$SECRET_ARGS" ]; then
        SECRET_ARGS="$SECRET_ARGS,SMTP_SERVER=smtp-server:latest"
    else
        SECRET_ARGS="--update-secrets=SMTP_SERVER=smtp-server:latest"
    fi
    echo -e "${GREEN}✓ Found smtp-server secret${NC}"
fi

if gcloud secrets describe smtp-port --project=$PROJECT_ID &>/dev/null 2>&1; then
    if [ -n "$SECRET_ARGS" ]; then
        SECRET_ARGS="$SECRET_ARGS,SMTP_PORT=smtp-port:latest"
    else
        SECRET_ARGS="--update-secrets=SMTP_PORT=smtp-port:latest"
    fi
    echo -e "${GREEN}✓ Found smtp-port secret${NC}"
fi

# Deploy to Cloud Run
echo -e "${YELLOW}Deploying to Cloud Run...${NC}"

if [ -n "$SECRET_ARGS" ]; then
    echo -e "${BLUE}Using secrets from Secret Manager${NC}"
    gcloud run deploy passport-pro \
        --image gcr.io/$PROJECT_ID/passport-pro:latest \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory 4Gi \
        --cpu 2 \
        --timeout 300 \
        --max-instances 10 \
        --set-env-vars "GOOGLE_CLOUD_PROJECT_ID=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION" \
        $SECRET_ARGS \
        --project=$PROJECT_ID
else
    echo -e "${YELLOW}⚠ No secrets found in Secret Manager. Using environment variables only.${NC}"
    gcloud run deploy passport-pro \
        --image gcr.io/$PROJECT_ID/passport-pro:latest \
        --platform managed \
        --region $REGION \
        --allow-unauthenticated \
        --memory 4Gi \
        --cpu 2 \
        --timeout 300 \
        --max-instances 10 \
        --set-env-vars "GOOGLE_CLOUD_PROJECT_ID=$PROJECT_ID,GOOGLE_CLOUD_LOCATION=$REGION" \
        --project=$PROJECT_ID
fi

# Get the service URL
SERVICE_URL=$(gcloud run services describe passport-pro --region=$REGION --format='value(status.url)' --project=$PROJECT_ID)

echo -e "${GREEN}=========================================="
echo -e "Deployment successful!${NC}"
echo -e "${GREEN}Service URL: ${SERVICE_URL}${NC}"
echo -e "${GREEN}=========================================="

