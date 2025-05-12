FROM python:3.10-slim

# Keep Python output unbuffered (helpful for logging)
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install any system-level dependencies (if you need e.g. FAISS or build tools)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgl1 && rm -rf /var/lib/apt/lists/*

COPY requirement.txt .
RUN pip install -r requirement.txt

COPY . .

# Ensure data directories exist (and will persist as volumes as we mount them)
RUN mkdir -p app/data app/faiss_index .cache_data

# (Optional but recommended) Run as non-root user
RUN groupadd -r app && useradd -r -g app app \
    && chown -R app:app /app
USER app

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
# CMD ["python", "app.main"]
