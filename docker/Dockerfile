FROM python:3.11-slim

WORKDIR /app

# System dependencies for faster-whisper and lxml
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer-cached unless requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project source
COPY . .

# Create data dirs that must exist at runtime
RUN mkdir -p data/chromadb data/pdfs evaluation/audio_samples

# Download BGE-m3 model at build time so containers start faster
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

EXPOSE 8000

CMD ["python", "-m", "api.main"]
