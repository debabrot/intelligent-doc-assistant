FROM python:3.11-slim

WORKDIR /app

# Set environment for unbuffered output
ENV PYTHONUNBUFFERED=1

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create non-root user
RUN adduser --disabled-password --gecos '' appuser && \
    chown -R appuser:appuser /app

USER appuser

# Run entrypoint
ENTRYPOINT ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8001"]