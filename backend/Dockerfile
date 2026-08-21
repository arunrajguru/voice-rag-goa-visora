FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app/backend

RUN mkdir -p /app/backend/data/index

# Build the English MSMARCO-XI index
RUN python backend/scripts/build_index.py --split train --sample 1000

# Verify that the build actually created the files
RUN echo "===== INDEX FILES =====" && \
    ls -lh /app/backend/data/index && \
    echo "======================="

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
