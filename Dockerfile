FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libzbar0 \
    libgomp1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip setuptools wheel

RUN pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cpu

RUN pip install paddlepaddle==2.6.2

RUN pip install paddleocr==2.7.3

# Pre-download CLIP model into image during build
RUN python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')" || echo "CLIP download skipped"

COPY requirements-render.txt .
RUN pip install -r requirements-render.txt

COPY . .

RUN mkdir -p data/labels data/scans static

EXPOSE 8080

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]