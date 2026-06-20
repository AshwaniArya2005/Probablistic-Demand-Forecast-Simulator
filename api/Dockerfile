# build context: the repository root. The models are NOT in the image: mount the hash-verified serving files at /models (see models.lock.json).
FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi==0.141.1 uvicorn==0.53.0 xgboost==3.4.1 numpy==2.2.6
COPY api/serving.py api/app.py ./
ENV MODEL_DIR=/models LOCK_FILE=/models/models.lock.json MODEL_NAME=v4_P10
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
