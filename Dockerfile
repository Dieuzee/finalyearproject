# syntax=docker/dockerfile:1

# ---- Crop Disease Detection web app ----
# Diagnosis runs on an Ollama vision model over HTTP (Ollama Cloud by default),
# so there is no bundled ML model and the image stays small.
FROM python:3.11-slim

# Keep Python output unbuffered and skip .pyc files.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=5000

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code.
COPY . .

# Create the uploads directory and run as a non-root user.
RUN mkdir -p static/shots \
    && adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

# Simple health check hitting the home page.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os; urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PORT','5000')+'/')" || exit 1

# Production WSGI server. Threads help while waiting on the Ollama Cloud API.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --timeout 120 --workers 2 --threads 4 app:app"]
