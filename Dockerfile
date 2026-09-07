# Use Python 3.11 slim (Debian bookworm)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first (Docker layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create upload directory
RUN mkdir -p static/shots

# Expose port
EXPOSE 5000

# Diagnosis is done via the Gemini API (no local ML model), so the app is light.
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --timeout 120 --workers 2 --threads 4 app:app
