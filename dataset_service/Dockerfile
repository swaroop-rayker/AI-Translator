FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY dataset_service/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY dataset_service/app /app/dataset_service/app
COPY backend /app/backend
COPY workers /app/workers

ENV PYTHONPATH=/app

EXPOSE 8001

CMD ["uvicorn", "dataset_service.app.main:app", "--host", "0.0.0.0", "--port", "8001"]
