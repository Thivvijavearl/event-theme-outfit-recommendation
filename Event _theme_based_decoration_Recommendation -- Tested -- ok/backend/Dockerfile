FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY . /app

# Expose the port the app runs on
EXPOSE 5002

# Run with gunicorn for production
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5002", "outfit_recommendation_api:app"]
