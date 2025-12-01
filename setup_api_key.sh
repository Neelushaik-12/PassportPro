#!/bin/bash

# Script to help set up API key for Passport Pro

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Passport Pro - API Key Setup${NC}"
echo "=========================================="

# Check if .env exists
ENV_FILE="backend/.env"
if [ ! -f "$ENV_FILE" ]; then
    echo -e "${YELLOW}Creating .env file from template...${NC}"
    cp backend/config_template.txt "$ENV_FILE"
fi

# Get API key from user
echo ""
echo -e "${BLUE}Enter your Google API Key:${NC}"
echo -e "${YELLOW}(Get it from: Google Cloud Console → APIs & Services → Credentials)${NC}"
read -p "API Key: " API_KEY

if [ -z "$API_KEY" ]; then
    echo -e "${RED}Error: API key cannot be empty${NC}"
    exit 1
fi

# Update .env file
if grep -q "GOOGLE_API_KEY=" "$ENV_FILE"; then
    # Update existing
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        sed -i '' "s|GOOGLE_API_KEY=.*|GOOGLE_API_KEY=$API_KEY|" "$ENV_FILE"
    else
        # Linux
        sed -i "s|GOOGLE_API_KEY=.*|GOOGLE_API_KEY=$API_KEY|" "$ENV_FILE"
    fi
    echo -e "${GREEN}✓ Updated GOOGLE_API_KEY in .env${NC}"
else
    # Add new
    echo "" >> "$ENV_FILE"
    echo "GOOGLE_API_KEY=$API_KEY" >> "$ENV_FILE"
    echo -e "${GREEN}✓ Added GOOGLE_API_KEY to .env${NC}"
fi

# Ask about Secret Manager
echo ""
read -p "Do you want to store this in Secret Manager for production? (y/n): " USE_SECRET

if [ "$USE_SECRET" = "y" ] || [ "$USE_SECRET" = "Y" ]; then
    PROJECT_ID=$(gcloud config get-value project 2>/dev/null)
    
    if [ -z "$PROJECT_ID" ]; then
        echo -e "${YELLOW}No project ID found.${NC}"
        read -p "Enter your Google Cloud Project ID: " PROJECT_ID
        gcloud config set project $PROJECT_ID
    fi
    
    echo -e "${YELLOW}Storing API key in Secret Manager...${NC}"
    echo -n "$API_KEY" | gcloud secrets create google-api-key \
        --data-file=- \
        --replication-policy="automatic" \
        --project=$PROJECT_ID 2>/dev/null || \
    echo -n "$API_KEY" | gcloud secrets versions add google-api-key \
        --data-file=- \
        --project=$PROJECT_ID
    
    echo -e "${GREEN}✓ API key stored in Secret Manager${NC}"
    
    # Grant access
    PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)" 2>/dev/null)
    if [ -n "$PROJECT_NUMBER" ]; then
        gcloud secrets add-iam-policy-binding google-api-key \
            --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
            --role="roles/secretmanager.secretAccessor" \
            --project=$PROJECT_ID 2>/dev/null || echo "Note: Access already granted or will be set during deployment"
    fi
fi

echo ""
echo -e "${GREEN}=========================================="
echo -e "✓ API Key configured!${NC}"
echo -e "${GREEN}=========================================="
echo ""
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Restart your server to use the new API key"
echo "2. You should see: '✓ Google GenAI client initialized'"
echo ""
echo -e "${YELLOW}To verify:${NC}"
echo "  cat $ENV_FILE | grep GOOGLE_API_KEY"

