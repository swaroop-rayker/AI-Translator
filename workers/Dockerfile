FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY backend/requirements.txt /app/backend-requirements.txt
RUN pip install --no-cache-dir -r /app/backend-requirements.txt

# Copy source code
COPY backend /app/backend
COPY dataset_service /app/dataset_service
COPY workers /app/workers

ENV PYTHONPATH=/app

# Start celery worker
ENTRYPOINT ["celery", "-A", "workers.tasks.celery_app", "worker", "--loglevel=info"]
