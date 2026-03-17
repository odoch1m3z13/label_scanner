FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libzbar0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

# CPU-only torch — much smaller than default (~500MB vs ~2GB)
RUN pip install torch==2.3.0+cpu torchvision==0.18.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu \
    --no-cache-dir

RUN pip install paddlepaddle==2.6.2 --no-cache-dir

RUN pip install paddleocr==2.7.3 --no-cache-dir

COPY requirements-render.txt .
RUN pip install -r requirements-render.txt --no-cache-dir

COPY . .

RUN mkdir -p data/labels data/scans static

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]