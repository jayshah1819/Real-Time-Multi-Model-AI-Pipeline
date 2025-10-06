#!/usr/bin/env python3
"""
Download and prepare models for the pipeline
"""
import torch
from torchvision.models import mobilenet_v3_small
import os
import gdown
import yaml
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
CONFIGS_DIR = Path("configs/model_configs")

# Model URLs
MODEL_URLS = {
    "yolov8n": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    "arcface_mobilenet": "https://drive.google.com/uc?id=YOUR_GDRIVE_ID",  # Replace with actual model URL
}


def download_yolo():
    """Download YOLOv8 model"""
    output_path = MODELS_DIR / "yolov8n.pt"
    if output_path.exists():
        logger.info(f"YOLO model already exists at {output_path}")
        return output_path

    logger.info("Downloading YOLOv8 model...")
    torch.hub.download_url_to_file(MODEL_URLS["yolov8n"], output_path)
    logger.info(f"Downloaded YOLO model to {output_path}")
    return output_path


def prepare_arcface():
    """Prepare ArcFace model"""
    output_path = MODELS_DIR / "arcface_mobilenet.pth"
    if output_path.exists():
        logger.info(f"ArcFace model already exists at {output_path}")
        return output_path

    logger.info("Preparing ArcFace model...")

    # Load config
    with open(CONFIGS_DIR / "models.yaml") as f:
        config = yaml.safe_load(f)

    # Create model
    backbone = mobilenet_v3_small(pretrained=True)
    backbone.classifier = torch.nn.Identity()

    model = torch.nn.Sequential(
        backbone,
        torch.nn.Linear(576, config["face_recognition"]["embedding_size"]),
        torch.nn.BatchNorm1d(config["face_recognition"]["embedding_size"]),
        torch.nn.PReLU(),
    )

    # Save model
    torch.save(model.state_dict(), output_path)
    logger.info(f"Saved ArcFace model to {output_path}")
    return output_path


def main():
    """Download and prepare all models"""
    # Create directories
    MODELS_DIR.mkdir(exist_ok=True)

    # Download models
    yolo_path = download_yolo()
    arcface_path = prepare_arcface()

    logger.info("All models downloaded and prepared successfully!")


if __name__ == "__main__":
    main()
