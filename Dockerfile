# Use Python 3.12 slim image
FROM python:3.11-slim-bullseye

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (includes start_server.sh)
COPY . .

# Set PYTHONPATH to include /app so backend imports work
ENV PYTHONPATH=/app

# Make startup script executable
RUN chmod +x /app/start_server.sh

# Expose port (Cloud Run will set PORT environment variable)
EXPOSE 8080

# Run the application using startup script
CMD ["/app/start_server.sh"]

