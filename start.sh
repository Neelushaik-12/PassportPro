#!/bin/bash

# Passport Pro Startup Script

echo "🚀 Starting Passport Pro..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Check for .env file
if [ ! -f "backend/.env" ]; then
    echo "⚠️  Warning: backend/.env file not found!"
    echo "   Please create it from backend/.env.example"
    echo "   The app will run in fallback mode without Vertex AI."
fi

# Start the backend server
echo "🎯 Starting backend server on http://localhost:8001"
echo "📸 Open frontend/index.html in your browser to use the app"
echo ""
cd backend
python3 main.py

