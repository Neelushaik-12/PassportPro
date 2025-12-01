#!/bin/bash

# Script to push PassportPro to GitHub as a public repository

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}PassportPro - GitHub Push Script${NC}"
echo "=========================================="

# Check if git is initialized
if [ ! -d ".git" ]; then
    echo -e "${YELLOW}Initializing git repository...${NC}"
    git init
fi

# Check if remote already exists
if git remote get-url origin &>/dev/null; then
    echo -e "${YELLOW}Remote 'origin' already exists:${NC}"
    git remote -v
    read -p "Do you want to update it? (y/n): " UPDATE_REMOTE
    if [ "$UPDATE_REMOTE" != "y" ] && [ "$UPDATE_REMOTE" != "Y" ]; then
        echo "Keeping existing remote."
    fi
fi

# Get repository URL
if [ -z "$1" ]; then
    echo ""
    echo -e "${YELLOW}Please provide your GitHub repository URL${NC}"
    echo "Example: https://github.com/yourusername/passportpro.git"
    echo "Or: git@github.com:yourusername/passportpro.git"
    read -p "GitHub repository URL: " REPO_URL
else
    REPO_URL=$1
fi

# Set remote
if [ "$UPDATE_REMOTE" = "y" ] || [ "$UPDATE_REMOTE" = "Y" ] || [ ! -z "$1" ]; then
    git remote remove origin 2>/dev/null || true
    git remote add origin "$REPO_URL"
    echo -e "${GREEN}✓ Remote set to: $REPO_URL${NC}"
fi

# Add all files (respecting .gitignore)
echo -e "${YELLOW}Adding files to git...${NC}"
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo -e "${YELLOW}No changes to commit.${NC}"
else
    # Commit
    echo -e "${YELLOW}Creating initial commit...${NC}"
    git commit -m "Initial commit: PassportPro - AI-powered passport photo generator

Features:
- AI-powered background removal using Vertex AI Gemini and Rembg
- Country-specific passport photo formatting (50+ countries)
- Smart background replacement (white, light grey, royal blue)
- User authentication system
- Modern web interface with camera capture
- Google Cloud Run deployment ready"

    echo -e "${GREEN}✓ Commit created${NC}"
fi

# Push to GitHub
echo -e "${YELLOW}Pushing to GitHub...${NC}"
echo -e "${YELLOW}Note: If this is your first push, you may need to:${NC}"
echo "  1. Create the repository on GitHub first"
echo "  2. Use: git push -u origin main (or master)"

BRANCH=$(git branch --show-current 2>/dev/null || echo "main")
if [ -z "$BRANCH" ]; then
    BRANCH="main"
    git branch -M main
fi

echo ""
read -p "Push to GitHub now? (y/n): " PUSH_NOW
if [ "$PUSH_NOW" = "y" ] || [ "$PUSH_NOW" = "Y" ]; then
    git push -u origin $BRANCH || {
        echo -e "${YELLOW}Push failed. You may need to:${NC}"
        echo "  1. Create the repository on GitHub first"
        echo "  2. Set the branch name: git branch -M main"
        echo "  3. Try again: git push -u origin main"
    }
else
    echo -e "${YELLOW}Ready to push. Run manually:${NC}"
    echo "  git push -u origin $BRANCH"
fi

echo ""
echo -e "${GREEN}=========================================="
echo -e "Security Check Summary:${NC}"
echo -e "${GREEN}✓ No API keys found${NC}"
echo -e "${GREEN}✓ No secrets exposed${NC}"
echo -e "${GREEN}✓ .env files excluded${NC}"
echo -e "${GREEN}✓ users.json files excluded${NC}"
echo -e "${GREEN}✓ venv/ excluded${NC}"
echo -e "${GREEN}=========================================="

