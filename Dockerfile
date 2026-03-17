FROM python:3.10-slim

# System dependencies for OpenCV, zbar, GL
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libzbar0 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install pip tools first
RUN pip install --upgrade pip setuptools wheel

# Install PyTorch CPU (smaller, faster to install)
RUN pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu

# Install PaddlePaddle from PyPI (Linux CPU version)
RUN pip install paddlepaddle==2.6.2

# Install PaddleOCR
RUN pip install paddleocr==2.7.3

# Install remaining dependencies
COPY requirements-render.txt .
RUN pip install -r requirements-render.txt

# Copy application code
COPY . .

# Create data directories
RUN mkdir -p data/labels data/scans static

EXPOSE 8000

CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}