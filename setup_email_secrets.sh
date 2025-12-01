#!/bin/bash

# Script to set up email secrets in Google Cloud Secret Manager

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${GREEN}PassportPro - Email Secrets Setup${NC}"
echo "=========================================="

# Get project ID
PROJECT_ID=$(gcloud config get-value project 2>/dev/null)

if [ -z "$PROJECT_ID" ]; then
    echo -e "${YELLOW}No project ID found.${NC}"
    read -p "Enter your Google Cloud Project ID: " PROJECT_ID
    gcloud config set project $PROJECT_ID
fi

echo -e "${GREEN}Using project: ${PROJECT_ID}${NC}"

# Email configuration
# Prompt user for email credentials (don't hardcode)
echo ""
echo -e "${YELLOW}Enter email configuration:${NC}"
read -p "Admin Email (to receive notifications): " ADMIN_EMAIL
read -p "SMTP Username (your email): " SMTP_USERNAME
read -sp "SMTP Password (Gmail App Password): " SMTP_PASSWORD
echo ""
read -p "SMTP Server (default: smtp.gmail.com): " SMTP_SERVER
SMTP_SERVER=${SMTP_SERVER:-smtp.gmail.com}
read -p "SMTP Port (default: 587): " SMTP_PORT
SMTP_PORT=${SMTP_PORT:-587}

echo ""
echo -e "${YELLOW}Setting up email secrets in Secret Manager...${NC}"

# Create or update admin-email secret
if gcloud secrets describe admin-email --project=$PROJECT_ID &>/dev/null 2>&1; then
    echo -n "$ADMIN_EMAIL" | gcloud secrets versions add admin-email --data-file=- --project=$PROJECT_ID
    echo -e "${GREEN}✓ Updated admin-email secret${NC}"
else
    echo -n "$ADMIN_EMAIL" | gcloud secrets create admin-email --data-file=- --project=$PROJECT_ID
    echo -e "${GREEN}✓ Created admin-email secret${NC}"
fi

# Create or update smtp-username secret
if gcloud secrets describe smtp-username --project=$PROJECT_ID &>/dev/null 2>&1; then
    echo -n "$SMTP_USERNAME" | gcloud secrets versions add smtp-username --data-file=- --project=$PROJECT_ID
    echo -e "${GREEN}✓ Updated smtp-username secret${NC}"
else
    echo -n "$SMTP_USERNAME" | gcloud secrets create smtp-username --data-file=- --project=$PROJECT_ID
    echo -e "${GREEN}✓ Created smtp-username secret${NC}"
fi

# Create or update smtp-password secret
if gcloud secrets describe smtp-password --project=$PROJECT_ID &>/dev/null 2>&1; then
    echo -n "$SMTP_PASSWORD" | gcloud secrets versions add smtp-password --data-file=- --project=$PROJECT_ID
    echo -e "${GREEN}✓ Updated smtp-password secret${NC}"
else
    echo -n "$SMTP_PASSWORD" | gcloud secrets create smtp-password --data-file=- --project=$PROJECT_ID
    echo -e "${GREEN}✓ Created smtp-password secret${NC}"
fi

# Create or update smtp-server secret
if gcloud secrets describe smtp-server --project=$PROJECT_ID &>/dev/null 2>&1; then
    echo -n "$SMTP_SERVER" | gcloud secrets versions add smtp-server --data-file=- --project=$PROJECT_ID
    echo -e "${GREEN}✓ Updated smtp-server secret${NC}"
else
    echo -n "$SMTP_SERVER" | gcloud secrets create smtp-server --data-file=- --project=$PROJECT_ID
    echo -e "${GREEN}✓ Created smtp-server secret${NC}"
fi

# Create or update smtp-port secret
if gcloud secrets describe smtp-port --project=$PROJECT_ID &>/dev/null 2>&1; then
    echo -n "$SMTP_PORT" | gcloud secrets versions add smtp-port --data-file=- --project=$PROJECT_ID
    echo -e "${GREEN}✓ Updated smtp-port secret${NC}"
else
    echo -n "$SMTP_PORT" | gcloud secrets create smtp-port --data-file=- --project=$PROJECT_ID
    echo -e "${GREEN}✓ Created smtp-port secret${NC}"
fi

# Grant Cloud Run service account access
# Get the project number for the compute service account
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)" 2>/dev/null)
if [ -z "$PROJECT_NUMBER" ]; then
    echo -e "${YELLOW}⚠ Could not get project number. Trying alternative service account...${NC}"
    SERVICE_ACCOUNT="${PROJECT_ID}@appspot.gserviceaccount.com"
else
    # Use the compute service account (used by Cloud Run)
    SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
fi

echo ""
echo -e "${YELLOW}Granting Cloud Run service account access to secrets...${NC}"
echo -e "${YELLOW}Service Account: ${SERVICE_ACCOUNT}${NC}"

for secret in admin-email smtp-username smtp-password smtp-server smtp-port; do
    echo -e "${YELLOW}  Granting access to ${secret}...${NC}"
    gcloud secrets add-iam-policy-binding $secret \
        --member="serviceAccount:${SERVICE_ACCOUNT}" \
        --role="roles/secretmanager.secretAccessor" \
        --project=$PROJECT_ID 2>&1 | grep -v "Updated IAM policy" || echo -e "${GREEN}    ✓ Access granted${NC}"
done

echo -e "${GREEN}✓ Granted access to email secrets${NC}"

# Remove password from .env file
echo ""
echo -e "${YELLOW}Removing SMTP_PASSWORD from backend/.env file...${NC}"
if [ -f "backend/.env" ]; then
    # Remove SMTP_PASSWORD line
    sed -i.bak '/^SMTP_PASSWORD=/d' backend/.env
    rm -f backend/.env.bak
    echo -e "${GREEN}✓ Removed SMTP_PASSWORD from .env file${NC}"
    echo -e "${YELLOW}  Note: Password is now stored securely in Secret Manager${NC}"
else
    echo -e "${YELLOW}⚠ backend/.env file not found${NC}"
fi

echo ""
echo -e "${GREEN}=========================================="
echo -e "Email secrets setup complete!${NC}"
echo -e "${GREEN}=========================================="
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Redeploy your Cloud Run service: ./deploy.sh"
echo "2. The email secrets will be automatically mounted from Secret Manager"
echo ""
echo -e "${YELLOW}Secrets created:${NC}"
echo "  - admin-email"
echo "  - smtp-username"
echo "  - smtp-password (secure)"
echo "  - smtp-server"
echo "  - smtp-port"

