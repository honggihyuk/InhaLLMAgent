# Multi-purpose image: retrieval/ops API, Kafka ingestion consumer, source pollers, research CronJob.
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ALPHA_DATA_DIR=/data \
    HF_HOME=/models

RUN apt-get update && apt-get install -y --no-install-recommends build-essential librdkafka-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY alphaagent ./alphaagent
COPY scripts ./scripts
RUN pip install ".[embeddings,llm,backtest]" confluent-kafka \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu

# non-root, read-only friendly: all writes go to the /data volume (and /tmp for the sandbox)
RUN useradd --uid 10001 --create-home alpha && mkdir -p /data /models && chown -R alpha /data /models
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1
ENTRYPOINT ["alphaagent"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--with-ops"]
