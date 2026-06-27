FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python requirements
COPY inference/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY inference/app /app/inference/app

ENV PYTHONPATH=/app

EXPOSE 8080

CMD ["uvicorn", "inference.app.main:app", "--host", "0.0.0.0", "--port", "8080"]
