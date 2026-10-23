# Serving image. Build context is the Space/staging directory produced by
# scripts/publish_hf.py (or scripts/publish_hf.py --stage-only), which has this layout:
#   Dockerfile  service/  src/weedcls/{__init__,preprocess}.py  src/weedcls/inference/predictor.py  scripts/fetch_model.py
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/artifacts/model.onnx \
    CARD_PATH=/app/artifacts/model_card.json

RUN useradd -m -u 1000 appuser
WORKDIR /app

COPY service/requirements.txt /app/service/requirements.txt
RUN pip install --no-cache-dir -r /app/service/requirements.txt

COPY src /app/src
COPY service /app/service
COPY scripts/fetch_model.py /app/scripts/fetch_model.py

# Weights are not in git: pull the pinned revision from the HF model repo and verify its hash.
RUN python scripts/fetch_model.py --manifest service/model_manifest.json --out artifacts \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:7860/health').status==200 else 1)"

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "7860"]
