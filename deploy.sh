#!/bin/bash

# Google Cloud Deployment Script for Passport Pro
# This script helps deploy the application to Google Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Passport Pro - Google Cloud Deployment${NC}"
echo "=========================================="

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo -e "${RED}Error: gcloud CLI is not installed.${NC}"
    echo "Please install it from: https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if docker is installed
if ! command -v docker &> /dev/null; then
    echo -e "${RED}Error: Docker is not installed.${NC}"
    echo "Please install it from: https://docs.docker.com/get-docker/"
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

# Authenticate Docker with Google Container Registry
echo -e "${YELLOW}Configuring Docker authentication...${NC}"
if ! gcloud auth configure-docker gcr.io --quiet 2>/dev/null; then
    echo -e "${YELLOW}Configuring Docker authentication (interactive)...${NC}"
    gcloud auth configure-docker gcr.io
else
    echo -e "${GREEN}✓ Docker authentication already configured${NC}"
fi

# Build and deploy
echo -e "${YELLOW}Building and deploying to Cloud Run...${NC}"

# Build the Docker image (force fresh build to avoid cache issues)
echo -e "${YELLOW}Building Docker image (fresh build, no cache)...${NC}"
docker build --no-cache -t gcr.io/$PROJECT_ID/passport-pro:latest .

# Push to Container Registry
echo -e "${YELLOW}Pushing image to Container Registry...${NC}"
docker push gcr.io/$PROJECT_ID/passport-pro:latest

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
    echo -e "${YELLOW}   Run ./setup_secrets.sh to set up secrets securely.${NC}"
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
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Update your frontend to use: ${SERVICE_URL}"
if [ -z "$SECRET_ARGS" ]; then
    echo "2. Set up secrets securely:"
    echo "   ./setup_secrets.sh"
    echo "   Then redeploy to use secrets"
else
    echo "2. Secrets are configured from Secret Manager ✓"
fi
echo ""
echo -e "${YELLOW}To view logs:${NC}"
echo "gcloud run logs read passport-pro --region=$REGION"
echo ""
echo -e "${YELLOW}Security Notes:${NC}"
echo "- API keys are stored in Secret Manager (not in environment variables)"
echo "- Secrets are automatically mounted at runtime"
echo "- Never commit .env files or API keys to version control"

