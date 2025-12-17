# Use Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml uv.lock README.md ./

# Sync dependencies
RUN uv sync --frozen --no-install-project

# Copy source code
COPY src/ ./src/
COPY examples/ ./examples/
COPY test_data/ ./test_data/

# Expose ports
EXPOSE 8000 8501

# Default command
CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]