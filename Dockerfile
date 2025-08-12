FROM python:3.10-slim

WORKDIR /

RUN pip install huggingface_hub

# Или через CLI
COPY download_models.py .
RUN python model_download.py

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8000
CMD ["python3", "-u", "sd15_handler.py"]

