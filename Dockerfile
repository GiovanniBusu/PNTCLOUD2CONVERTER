FROM python:3.11-slim

# System dependencies for pye57 (libE57Format + its own deps: xerces-c, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git \
    libxerces-c-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
ENV HOST=0.0.0.0
ENV PORT=8000

CMD ["python", "run.py"]
