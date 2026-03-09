# Backend for Place Names Voice Recorder (Flask + ffmpeg for WAV conversion)
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt requirements-training.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-training.txt
COPY app.py .
COPY templates/ templates/
COPY training/ training/
RUN mkdir -p dataset

ENV PORT=5000
EXPOSE 5000
CMD ["python", "app.py"]
