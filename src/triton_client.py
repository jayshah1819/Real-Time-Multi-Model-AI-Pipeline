"""
Triton Inference Server client for multi-model pipeline
"""

import numpy as np
import tritonclient.http as tritonhttpclient
import tritonclient.grpc as tritonclient
from tritonclient.utils import InferenceServerException
import time
from typing import List, Tuple, Optional, Dict
import cv2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TritonPipeline:
    def __init__(
        self,
        url: str = "localhost:8001",
        use_grpc: bool = True,
        model_version: str = "1",
    ):
        """
        Initialize Triton client for multi-model pipeline

        Args:
            url: Triton server URL
            use_grpc: Use gRPC instead of HTTP
            model_version: Model version to use
        """
        self.model_version = model_version
        if use_grpc:
            self.client = tritonclient.InferenceServerClient(url=url)
        else:
            self.client = tritonhttpclient.InferenceServerClient(url=url)

        # Verify server is ready
        if not self.client.is_server_ready():
            raise RuntimeError("Triton server is not ready!")

        # Verify models are ready
        models = ["yolo", "face_recognition"]
        for model in models:
            if not self.client.is_model_ready(model, model_version):
                raise RuntimeError(f"Model {model} is not ready!")

        logger.info("Triton pipeline initialized successfully")

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Preprocess frame for YOLO inference"""
        # Resize
        input_size = (640, 640)
        resized = cv2.resize(frame, input_size)

        # Normalize and transpose to CHW
        img = resized.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))

        # Add batch dimension
        img = np.expand_dims(img, 0)
        return img

    def detect_objects(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run YOLO object detection

        Returns:
            Tuple of (boxes, scores)
        """
        input_name = "images"
        output_name = "output0"

        # Preprocess
        input_data = self._preprocess_frame(frame)

        # Prepare inputs
        inputs = [tritonclient.InferInput(input_name, input_data.shape, "FP32")]
        inputs[0].set_data_from_numpy(input_data)

        # Run inference
        start_time = time.time()
        results = self.client.infer("yolo", inputs)
        inference_time = (time.time() - start_time) * 1000

        # Process outputs
        output = results.as_numpy(output_name)
        boxes = output[..., :4]
        scores = output[..., 4]

        logger.debug(f"YOLO inference time: {inference_time:.2f}ms")
        return boxes, scores

    def extract_face_embeddings(self, face_crops: np.ndarray) -> np.ndarray:
        """
        Extract face embeddings using ArcFace model

        Args:
            face_crops: Batch of face crops (N, 3, 112, 112)

        Returns:
            Face embeddings (N, 512)
        """
        input_name = "input"
        output_name = "output"

        # Prepare inputs
        inputs = [tritonclient.InferInput(input_name, face_crops.shape, "FP32")]
        inputs[0].set_data_from_numpy(face_crops)

        # Run inference
        start_time = time.time()
        results = self.client.infer("face_recognition", inputs)
        inference_time = (time.time() - start_time) * 1000

        # Get embeddings
        embeddings = results.as_numpy(output_name)

        logger.debug(f"Face recognition inference time: {inference_time:.2f}ms")
        return embeddings

    def process_frame(
        self, frame: np.ndarray, conf_thresh: float = 0.5, nms_thresh: float = 0.45
    ) -> Dict[str, np.ndarray]:
        """
        Process a single frame through the complete pipeline

        Args:
            frame: Input frame (H, W, 3)
            conf_thresh: Confidence threshold for detections
            nms_thresh: NMS IoU threshold

        Returns:
            Dict containing boxes, scores, and face embeddings
        """
        # Run object detection
        boxes, scores = self.detect_objects(frame)

        # Filter by confidence and class
        mask = scores > conf_thresh
        boxes = boxes[mask]
        scores = scores[mask]

        if len(boxes) == 0:
            return {
                "boxes": np.array([]),
                "scores": np.array([]),
                "embeddings": np.array([]),
            }

        # Perform NMS
        from src.kernels.preprocessing import PreprocessingKernel

        preprocessor = PreprocessingKernel()
        keep_idx = preprocessor.nms(boxes, scores, nms_thresh)

        boxes = boxes[keep_idx]
        scores = scores[keep_idx]

        # Crop and preprocess faces
        face_crops = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)
            face = frame[y1:y2, x1:x2]
            face = cv2.resize(face, (112, 112))
            face = face.astype(np.float32) / 255.0
            face = np.transpose(face, (2, 0, 1))
            face_crops.append(face)

        face_crops = np.stack(face_crops)

        # Extract face embeddings
        embeddings = self.extract_face_embeddings(face_crops)

        return {"boxes": boxes, "scores": scores, "embeddings": embeddings}


def main():
    """Test the Triton pipeline"""
    import cv2

    # Initialize pipeline
    pipeline = TritonPipeline()

    # Open video
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        results = pipeline.process_frame(frame)

        # Draw results
        for box, score in zip(results["boxes"], results["scores"]):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{score:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Triton Pipeline", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
