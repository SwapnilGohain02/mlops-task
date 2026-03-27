# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy dependency file first (layer caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY run.py .
COPY config.yaml .
COPY data.csv .

# Default command
CMD ["python", "run.py", \
     "--input",    "data.csv", \
     "--config",   "config.yaml", \
     "--output",   "metrics.json", \
     "--log-file", "run.log"]