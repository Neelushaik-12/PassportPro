#!/bin/bash
# Startup script for Cloud Run
# Reads PORT from environment variable and starts uvicorn

set -e

# Get port from environment variable (Cloud Run sets this)
PORT=${PORT:-8080}

echo "Starting Passport Pro API on port $PORT"

# Start uvicorn
exec uvicorn backend.main:app --host 0.0.0.0 --port "$PORT"

