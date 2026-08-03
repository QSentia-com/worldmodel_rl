FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    QSENTIA_ARTIFACT_DIR=/app/artifacts/current \
    QSENTIA_OUTPUT_DIR=/app/outputs

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src ./src
COPY config ./config
COPY vendor ./vendor

RUN pip install --no-cache-dir "./vendor/qsentia-utils[db]" .

RUN mkdir -p /app/artifacts/current /app/outputs /app/logs

CMD ["python", "-m", "qsentia_worldmodel_rl_containerized.run_job"]
