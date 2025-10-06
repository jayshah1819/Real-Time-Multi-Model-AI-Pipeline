#!/bin/bash

# Check CUDA version
REQUIRED_CUDA="11.7"
if ! command -v nvcc &> /dev/null; then
    echo "CUDA not found. Please install CUDA $REQUIRED_CUDA"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
if [[ "$CUDA_VERSION" < "$REQUIRED_CUDA" ]]; then
    echo "CUDA version $CUDA_VERSION found, but $REQUIRED_CUDA is required"
    exit 1
fi

# Check GPU availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "No NVIDIA GPU found"
    exit 1
fi

# Check Python version
REQUIRED_PYTHON="3.8"
PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
if [[ "$PYTHON_VERSION" < "$REQUIRED_PYTHON" ]]; then
    echo "Python version $PYTHON_VERSION found, but $REQUIRED_PYTHON or higher is required"
    exit 1
fi

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Download models
mkdir -p models
python3 scripts/download_models.py

# Convert models to TensorRT
python3 scripts/convert_to_tensorrt.py

# Verify installation
python3 scripts/verify_setup.py

echo "Setup complete!"