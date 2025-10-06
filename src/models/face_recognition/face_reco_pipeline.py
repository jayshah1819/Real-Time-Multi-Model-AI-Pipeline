import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np
import cv2
import time
import os
from typing import List, Tuple, Optional, Dict
import logging
from dataclasses import dataclass
from collections import deque


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ArcFaceMobileNet(nn.Module):

    def __init__(self, num_features=512, num_classes=10000):
        super(ArcFaceMobileNet, self).__init__()
        self.backbone = mobilenet_v3_small(pretrained=True)
        self.backbone.classifier = nn.Identity()

        # Feature extraction layer
        self.feature_layer = nn.Sequential(
            nn.Linear(576, num_features),  # MobileNetV3-small outputs 576 features
            nn.BatchNorm1d(num_features),
            nn.PReLU(),
        )

        # ArcFace layer would go here in full implementation
        # For demo purposes, we'll use the features directly

    def forward(self, x):
        # Extract backbone features
        features = self.backbone(x)
        # Normalize features
        features = self.feature_layer(features)
        features = nn.functional.normalize(features, p=2, dim=1)
        return features


class TensorRTFaceRecognizer:
    def __del__(self):
        """Cleanup CUDA resources"""
        try:
            if self.input_mem_gpu:
                self.input_mem_gpu.free()
            if self.output_mem_gpu:
                self.output_mem_gpu.free()
            if self.stream:
                self.stream.synchronize()
            if self.context:
                self.context.destroy()
            if self.engine:
                self.engine.destroy()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    """
    TensorRT optimized face recognition engine
    """

    def __init__(
        self, onnx_path: str, engine_path: str, input_shape: Tuple[int, int, int, int]
    ):
        self.input_shape = input_shape  # (batch, channels, height, width)
        self.engine_path = engine_path
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.input_mem_gpu = None
        self.output_mem_gpu = None

        # Build or load TensorRT engine
        if not self._load_engine():
            self._build_engine_from_onnx(onnx_path)

        self._allocate_buffers()

    def _build_engine_from_onnx(self, onnx_path: str):
        """Build TensorRT engine from ONNX model with FP16 optimization"""
        logger.info("Building TensorRT engine from ONNX...")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        builder = trt.Builder(TRT_LOGGER)
        config = builder.create_builder_config()

        # Enable FP16 precision for speed
        config.set_flag(trt.BuilderFlag.FP16)

        # Set memory pool size (2GB)
        config.max_workspace_size = 2 << 30

        # Parse ONNX model
        network = builder.create_network(
            1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        )
        parser = trt.OnnxParser(network, TRT_LOGGER)

        with open(onnx_path, "rb") as model:
            if not parser.parse(model.read()):
                logger.error("Failed to parse ONNX model")
                for error in range(parser.num_errors):
                    logger.error(parser.get_error(error))
                return False

        # Build engine
        self.engine = builder.build_engine(network, config)

        # Save engine to file
        with open(self.engine_path, "wb") as f:
            f.write(self.engine.serialize())

        logger.info(f"TensorRT engine saved to {self.engine_path}")
        return True

    def _load_engine(self):
        """Load existing TensorRT engine"""
        try:
            TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
            runtime = trt.Runtime(TRT_LOGGER)

            with open(self.engine_path, "rb") as f:
                engine_data = f.read()

            self.engine = runtime.deserialize_cuda_engine(engine_data)
            logger.info(f"Loaded existing TensorRT engine from {self.engine_path}")
            return True
        except:
            logger.info("No existing engine found, will build new one")
            return False

    def _allocate_buffers(self):
        """Allocate GPU memory buffers"""
        if not self.engine:
            return

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()

        # Input buffer
        input_size = np.prod(self.input_shape) * np.dtype(np.float32).itemsize
        self.input_mem_gpu = cuda.mem_alloc(input_size)

        # Output buffer (assuming 512-dim feature vector)
        output_size = self.input_shape[0] * 512 * np.dtype(np.float32).itemsize
        self.output_mem_gpu = cuda.mem_alloc(output_size)

        self.bindings = [int(self.input_mem_gpu), int(self.output_mem_gpu)]

    def infer(self, face_batch: np.ndarray) -> np.ndarray:
        """
        Run inference on batch of face crops
        Args:
            face_batch: numpy array of shape (batch, 3, 112, 112)
        Returns:
            Feature vectors of shape (batch, 512)
        """
        # Copy input to GPU
        cuda.memcpy_htod_async(self.input_mem_gpu, face_batch, self.stream)

        # Run inference
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle
        )

        # Copy output back to CPU
        output = np.empty((face_batch.shape[0], 512), dtype=np.float32)
        cuda.memcpy_dtoh_async(output, self.output_mem_gpu, self.stream)
        self.stream.synchronize()

        return output


class CUDABatchCropper:
    """
    Custom CUDA kernel for zero-copy batch face cropping
    """

    CUDA_KERNEL_CODE = """
    __global__ void batch_crop_faces(
        const float* input_frame,      // Input: (H, W, 3) - Original 4K frame
        float* output_faces,           // Output: (N, 3, 112, 112) - Cropped faces
        const float* bboxes,           // Input: (N, 4) - Bounding boxes [x1, y1, x2, y2]
        int frame_height,
        int frame_width,
        int num_faces,
        int face_size                  // Target face size (112x112)
    ) {
        int face_idx = blockIdx.x;
        int pixel_idx = threadIdx.x + blockIdx.y * blockDim.x;
        
        if (face_idx >= num_faces) return;
        
        int pixels_per_face = face_size * face_size;
        if (pixel_idx >= pixels_per_face) return;
        
        // Get bounding box for this face
        float x1 = bboxes[face_idx * 4 + 0];
        float y1 = bboxes[face_idx * 4 + 1];
        float x2 = bboxes[face_idx * 4 + 2];
        float y2 = bboxes[face_idx * 4 + 3];
        
        // Calculate crop dimensions
        float crop_width = x2 - x1;
        float crop_height = y2 - y1;
        
        // Calculate position in face crop
        int face_y = pixel_idx / face_size;
        int face_x = pixel_idx % face_size;
        
        // Map to original frame coordinates (with bilinear interpolation)
        float src_x = x1 + (face_x / (float)face_size) * crop_width;
        float src_y = y1 + (face_y / (float)face_size) * crop_height;
        
        // Ensure coordinates are within frame bounds
        src_x = fmaxf(0.0f, fminf(src_x, frame_width - 1.0f));
        src_y = fmaxf(0.0f, fminf(src_y, frame_height - 1.0f));
        
        // Bilinear interpolation
        int x_floor = (int)floorf(src_x);
        int y_floor = (int)floorf(src_y);
        int x_ceil = min(x_floor + 1, frame_width - 1);
        int y_ceil = min(y_floor + 1, frame_height - 1);
        
        float x_frac = src_x - x_floor;
        float y_frac = src_y - y_floor;
        
        // Sample and interpolate for each channel (RGB)
        for (int c = 0; c < 3; c++) {
            float tl = input_frame[(y_floor * frame_width + x_floor) * 3 + c];
            float tr = input_frame[(y_floor * frame_width + x_ceil) * 3 + c];
            float bl = input_frame[(y_ceil * frame_width + x_floor) * 3 + c];
            float br = input_frame[(y_ceil * frame_width + x_ceil) * 3 + c];
            
            float top = tl + (tr - tl) * x_frac;
            float bottom = bl + (br - bl) * x_frac;
            float interpolated = top + (bottom - top) * y_frac;
            
            // Store in CHW format: output_faces[face_idx, channel, y, x]
            int output_idx = face_idx * (3 * face_size * face_size) + 
                           c * (face_size * face_size) + 
                           face_y * face_size + face_x;
            output_faces[output_idx] = interpolated / 255.0f; // Normalize to [0,1]
        }
    }
    """

    def __init__(self):
        # Compile CUDA kernel
        self.mod = SourceModule(self.CUDA_KERNEL_CODE)
        self.batch_crop_func = self.mod.get_function("batch_crop_faces")

    def crop_faces(
        self,
        frame_gpu: cuda.DeviceAllocation,
        bboxes: np.ndarray,
        frame_shape: Tuple[int, int],
        face_size: int = 112,
    ) -> cuda.DeviceAllocation:
        """
        Crop faces from frame using GPU kernel

        Args:
            frame_gpu: GPU memory containing the frame (H, W, 3)
            bboxes: Bounding boxes array (N, 4) [x1, y1, x2, y2]
            frame_shape: (height, width) of original frame
            face_size: Target size for face crops (default: 112x112)

        Returns:
            GPU memory containing cropped faces (N, 3, face_size, face_size)
        """
        num_faces = bboxes.shape[0]
        if num_faces == 0:
            return None

        frame_height, frame_width = frame_shape

        # Allocate output buffer on GPU
        output_size = (
            num_faces * 3 * face_size * face_size * np.dtype(np.float32).itemsize
        )
        output_gpu = cuda.mem_alloc(output_size)

        # Allocate and copy bboxes to GPU
        bboxes_gpu = cuda.mem_alloc(bboxes.nbytes)
        cuda.memcpy_htod(bboxes_gpu, bboxes.astype(np.float32))

        # Configure kernel launch parameters
        threads_per_block = 256
        blocks_per_face = (
            face_size * face_size + threads_per_block - 1
        ) // threads_per_block

        grid_dim = (num_faces, blocks_per_face, 1)
        block_dim = (threads_per_block, 1, 1)

        # Launch kernel
        self.batch_crop_func(
            frame_gpu,
            output_gpu,
            bboxes_gpu,
            np.int32(frame_height),
            np.int32(frame_width),
            np.int32(num_faces),
            np.int32(face_size),
            grid=grid_dim,
            block=block_dim,
        )

        # Cleanup
        bboxes_gpu.free()

        return output_gpu


@dataclass
class FaceIdentity:
    name: str
    embedding: np.ndarray
    confidence: float


class FaceRecognitionPipeline:
    """
    Complete zero-copy face recognition pipeline
    """

    def __init__(
        self, model_path: str, engine_path: str, similarity_threshold: float = 0.6
    ):
        self.similarity_threshold = similarity_threshold
        self.face_database: Dict[str, FaceIdentity] = {}

        # Preprocessing normalization constants
        self.mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
        self.std = np.array([0.229, 0.224, 0.225]).reshape(1, 3, 1, 1)

        # Initialize components
        self.cropper = CUDABatchCropper()

        # Create and export model to ONNX first (if needed)
        self._create_onnx_model(model_path)

        # Initialize TensorRT engine
        input_shape = (16, 3, 112, 112)  # Max batch size of 16 faces
        self.face_recognizer = TensorRTFaceRecognizer(
            onnx_path=model_path, engine_path=engine_path, input_shape=input_shape
        )

    def _create_onnx_model(self, onnx_path: str):
        """Create and export ArcFace model to ONNX"""
        if os.path.exists(onnx_path):
            return

        logger.info("Creating ONNX model...")
        model = ArcFaceMobileNet()
        model.eval()

        # Dummy input for ONNX export
        dummy_input = torch.randn(1, 3, 112, 112)

        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )
        logger.info(f"ONNX model saved to {onnx_path}")

    def add_face_to_database(
        self, name: str, embedding: np.ndarray, confidence: float = 1.0
    ) -> None:
        """Add a face embedding to the database"""
        # Normalize embedding
        embedding = embedding / np.linalg.norm(embedding)
        self.face_database[name] = FaceIdentity(
            name=name, embedding=embedding, confidence=confidence
        )

    def match_face(self, embedding: np.ndarray) -> Optional[FaceIdentity]:
        """Match a face embedding against the database"""
        if not self.face_database:
            return None

        # Normalize query embedding
        embedding = embedding / np.linalg.norm(embedding)

        # Find best match
        best_match = None
        best_similarity = -1

        for identity in self.face_database.values():
            similarity = np.dot(embedding, identity.embedding)
            if similarity > best_similarity and similarity > self.similarity_threshold:
                best_similarity = similarity
                best_match = identity

        return best_match

    def process_frame_with_detections(
        self,
        frame_gpu: cuda.DeviceAllocation,
        person_bboxes: np.ndarray,
        frame_shape: Tuple[int, int],
    ) -> np.ndarray:
        """
        Complete zero-copy pipeline: frame + detections -> face features

        Args:
            frame_gpu: GPU memory containing the original frame
            person_bboxes: Detected person bounding boxes (N, 4)
            frame_shape: (height, width) of the frame

        Returns:
            Face feature vectors (N, 512)
        """
        if len(person_bboxes) == 0:
            return np.array([]).reshape(0, 512)

        start_time = time.time()

        faces_gpu = self.cropper.crop_faces(frame_gpu, person_bboxes, frame_shape)

        crop_time = time.time()

        num_faces = person_bboxes.shape[0]
        face_batch = np.empty((num_faces, 3, 112, 112), dtype=np.float32)
        cuda.memcpy_dtoh(face_batch, faces_gpu)

        # Normalize input batch
        face_batch = (face_batch - self.mean) / self.std

        try:
            # Run inference
            features = self.face_recognizer.infer(face_batch)
        except cuda.Error as e:
            logger.error(f"CUDA error during inference: {e}")
            return np.array([]).reshape(0, 512)

        inference_time = time.time()

        faces_gpu.free()

        logger.info(
            f"Face pipeline - Crop: {(crop_time - start_time)*1000:.2f}ms, "
            f"Inference: {(inference_time - crop_time)*1000:.2f}ms, "
            f"Total: {(inference_time - start_time)*1000:.2f}ms for {num_faces} faces"
        )

        return features


def benchmark_pipeline():
    import os

    # Create sample data
    frame_height, frame_width = 2160, 3840  # 4K
    sample_frame = np.random.randint(
        0, 256, (frame_height, frame_width, 3), dtype=np.uint8
    )

    # Sample person detections
    sample_bboxes = np.array(
        [
            [100, 200, 300, 500],  # Person 1
            [800, 150, 1000, 450],  # Person 2
            [1500, 300, 1700, 650],  # Person 3
            [2000, 100, 2200, 400],  # Person 4
        ],
        dtype=np.float32,
    )

    # Initialize pipeline
    model_path = "arcface_mobilenet.onnx"
    engine_path = "face_recognition_fp16.trt"

    pipeline = FaceRecognitionPipeline(model_path, engine_path)

    # Upload frame to GPU
    frame_gpu = cuda.mem_alloc(sample_frame.nbytes)
    cuda.memcpy_htod(frame_gpu, sample_frame.astype(np.float32))

    # Benchmark
    num_runs = 100
    total_time = 0

    logger.info("Starting benchmark...")
    for i in range(num_runs):
        start = time.time()
        features = pipeline.process_frame_with_detections(
            frame_gpu, sample_bboxes, (frame_height, frame_width)
        )
        total_time += time.time() - start

        if i == 0:
            logger.info(f"First run - Output shape: {features.shape}")

    avg_time = (total_time / num_runs) * 1000
    fps = 1000 / avg_time

    logger.info(f"Benchmark Results:")
    logger.info(f"Average latency: {avg_time:.2f}ms")
    logger.info(f"Effective FPS: {fps:.1f}")
    logger.info(f"Faces per second: {len(sample_bboxes) * fps:.0f}")

    # Cleanup
    frame_gpu.free()


if __name__ == "__main__":
    benchmark_pipeline()
