FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir fastapi uvicorn pydantic \
    && pip install --no-cache-dir llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
COPY src/schema.py src/schema.py
COPY serve serve
ENV MODEL_PATH=/models/model.gguf
EXPOSE 8000
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000"]
# build: docker build -t sales-extractor .
# run:   docker run -p 8000:8000 -v "$PWD/outputs/gguf:/models" -e MODEL_PATH=/models/<your-file>.gguf sales-extractor
