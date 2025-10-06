#!/usr/bin/env python3
"""
Verify the setup of the pipeline components
"""
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
import numpy as np
from pathlib import Path
import yaml
import logging
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def check_gpu():
    """Verify GPU and CUDA setup"""
    if not torch.cuda.is_available():
        logger.error("CUDA is not available")
        return False

    logger.info(f"Found {torch.cuda.device_count()} CUDA device(s)")
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        logger.info(f"  Device {i}: {props.name}")
        logger.info(f"    Compute Capability: {props.major}.{props.minor}")
        logger.info(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
    return True


def check_models():
    """Verify model files"""
    models_dir = Path("models")
    required_files = [
        "yolov8n.pt",
        "arcface_mobilenet.pth",
        "yolov8n.engine",
        "arcface_mobilenet.engine",
    ]

    missing = []
    for file in required_files:
        if not (models_dir / file).exists():
            missing.append(file)

    if missing:
        logger.error(f"Missing model files: {missing}")
        return False

    logger.info("All model files found")
    return True


def check_tensorrt():
    """Verify TensorRT setup"""
    try:
        logger = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(logger)
        logger.info(f"TensorRT version: {trt.__version__}")
        return True
    except Exception as e:
        logger.error(f"TensorRT setup error: {e}")
        return False


def check_triton_models():
    """Verify Triton model repository"""
    triton_dir = Path("triton_models")
    required_models = ["yolo", "face_recognition"]

    missing = []
    for model in required_models:
        model_dir = triton_dir / model
        if not model_dir.exists():
            missing.append(model)
            continue

        # Check config and model files
        if not (model_dir / "config.pbtxt").exists():
            missing.append(f"{model}/config.pbtxt")
        if not (model_dir / "1/model.plan").exists():
            missing.append(f"{model}/1/model.plan")

    if missing:
        logger.error(f"Missing Triton files: {missing}")
        return False

    logger.info("All Triton model files found")
    return True


def check_opencv():
    """Verify OpenCV setup"""
    try:
        logger.info(f"OpenCV version: {cv2.__version__}")
        if cv2.cuda.getCudaEnabledDeviceCount() == 0:
            logger.warning("OpenCV CUDA support not available")
            return False
        return True
    except Exception as e:
        logger.error(f"OpenCV setup error: {e}")
        return False


def main():
    """Run all verification checks"""
    checks = [
        ("GPU/CUDA", check_gpu),
        ("Models", check_models),
        ("TensorRT", check_tensorrt),
        ("Triton Models", check_triton_models),
        ("OpenCV", check_opencv),
    ]

    success = True
    logger.info("Running setup verification...")

    for name, check in checks:
        logger.info(f"\nChecking {name}...")
        if not check():
            success = False

    if not success:
        logger.error("\nSetup verification failed!")
        sys.exit(1)

    logger.info("\nSetup verification completed successfully!")


if __name__ == "__main__":
    main()
