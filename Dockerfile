# Use NVIDIA PyTorch container as base
FROM nvcr.io/nvidia/pytorch:23.09-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install Triton client
RUN pip install --no-cache-dir tritonclient[all]

# Set working directory
WORKDIR /workspace

# Copy source code
COPY . .

# Set environment variables
ENV PYTHONPATH=/workspace

# Default command
CMD ["python", "src/triton_client.py"]