FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    EMORA_CORS_ORIGINS=http://localhost:5173

WORKDIR /app

COPY requirements-mlops.txt requirements-ci.txt ./
RUN pip install --no-cache-dir -r requirements-mlops.txt -r requirements-ci.txt \
    uvicorn[standard] python-multipart transformers

COPY api ./api
COPY src ./src
COPY scripts ./scripts

# Checkpoints and reports are deliberately external runtime mounts.
RUN mkdir -p /models /reports

EXPOSE 8001
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/live')"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8001"]
