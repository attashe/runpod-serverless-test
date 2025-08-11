FROM python:3.10-slim

# Install wget
RUN apt-get update && apt-get install -y wget && rm -rf /var/lib/apt/lists/*

WORKDIR /

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download model during build
RUN mkdir -p /models
RUN wget -O /models/realvis6.safetensors \
    "https://huggingface.co/alexgenovese/checkpoint/resolve/5d96d799d7e943878a2c7614674eb48435891d00/realisticVisionV60B1_v60B1VAE.safetensors"

# Copy application code
COPY . .

EXPOSE 8000
CMD ["python3", "-u", "rp_handler.py"]

