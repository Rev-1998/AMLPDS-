# --- Dockerfile ---
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data/models logs

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"

# Expose port
EXPOSE 5000

# Set environment variables
ENV FLASK_APP=src.web.app
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "src.web.app:app"]
