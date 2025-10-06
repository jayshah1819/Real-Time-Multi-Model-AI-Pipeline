"""
Python wrapper for CUDA preprocessing kernels
"""

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import os


def get_kernel_path(filename):
    """Get absolute path to kernel file"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(current_dir, filename)


class PreprocessingKernel:
    """Wrapper for video preprocessing CUDA kernels"""

    def __init__(self):
        with open(get_kernel_path("preprocessing_kernel.cu"), "r") as f:
            kernel_source = f.read()

        self.module = SourceModule(kernel_source)
        self.preprocess_kernel = self.module.get_function("launch_preprocess_kernel")
        self.nms_kernel = self.module.get_function("launch_nms_kernel")

        # Default normalization parameters (ImageNet)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        # Allocate GPU memory for normalization parameters
        self.d_mean = cuda.mem_alloc(self.mean.nbytes)
        self.d_std = cuda.mem_alloc(self.std.nbytes)
        cuda.memcpy_htod(self.d_mean, self.mean)
        cuda.memcpy_htod(self.d_std, self.std)

    def preprocess_frame(self, frame: np.ndarray, output_size=(640, 640)) -> np.ndarray:
        """
        Preprocess a video frame using fused CUDA kernel

        Args:
            frame: numpy array of shape (H, W, 3) in BGR format
            output_size: tuple of (height, width) for output tensor

        Returns:
            Preprocessed tensor of shape (3, output_height, output_width)
        """
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError("Input frame must be a uint8 numpy array")

        input_height, input_width = frame.shape[:2]
        output_height, output_width = output_size

        # Allocate GPU memory
        input_size = frame.nbytes
        output_size = 3 * output_height * output_width * 4  # float32 output

        d_input = cuda.mem_alloc(input_size)
        d_output = cuda.mem_alloc(output_size)

        # Copy input to GPU
        cuda.memcpy_htod(d_input, frame)

        # Create stream for async operations
        stream = cuda.Stream()

        # Launch kernel
        self.preprocess_kernel(
            d_input,
            d_output,
            np.int32(input_height),
            np.int32(input_width),
            np.int32(output_height),
            np.int32(output_width),
            np.int32(1),  # batch_size
            self.d_mean,
            self.d_std,
            stream,
            block=(16, 16, 1),
            grid=((output_width + 15) // 16, (output_height + 15) // 16, 1),
        )

        # Allocate output array and copy result
        output = np.empty((3, output_height, output_width), dtype=np.float32)
        cuda.memcpy_dtoh_async(output, d_output, stream)
        stream.synchronize()

        # Cleanup
        d_input.free()
        d_output.free()

        return output

    def nms(
        self, boxes: np.ndarray, scores: np.ndarray, nms_thresh: float = 0.45
    ) -> np.ndarray:
        """
        Perform Non-Maximum Suppression using CUDA kernel

        Args:
            boxes: numpy array of shape (N, 4) with boxes in (x1, y1, x2, y2) format
            scores: numpy array of shape (N,) with confidence scores
            nms_thresh: IoU threshold for NMS

        Returns:
            Array of indices for kept boxes
        """
        if not isinstance(boxes, np.ndarray) or not isinstance(scores, np.ndarray):
            raise ValueError("Inputs must be numpy arrays")

        n_boxes = len(boxes)
        if n_boxes == 0:
            return np.array([], dtype=np.int32)

        # Allocate GPU memory
        boxes_size = boxes.nbytes
        scores_size = scores.nbytes
        keep_size = n_boxes * 4  # int32

        d_boxes = cuda.mem_alloc(boxes_size)
        d_scores = cuda.mem_alloc(scores_size)
        d_keep = cuda.mem_alloc(keep_size)
        d_num_keep = cuda.mem_alloc(4)  # single int32

        # Initialize keep array to 1s
        keep = np.ones(n_boxes, dtype=np.int32)
        num_keep = np.array([n_boxes], dtype=np.int32)

        cuda.memcpy_htod(d_boxes, boxes.astype(np.float32))
        cuda.memcpy_htod(d_scores, scores.astype(np.float32))
        cuda.memcpy_htod(d_keep, keep)
        cuda.memcpy_htod(d_num_keep, num_keep)

        # Create stream
        stream = cuda.Stream()

        # Launch kernel
        self.nms_kernel(
            d_boxes,
            d_scores,
            d_keep,
            d_num_keep,
            np.int32(n_boxes),
            np.float32(nms_thresh),
            stream,
            block=(16, 16, 1),
            grid=((n_boxes + 15) // 16, (n_boxes + 15) // 16, 1),
        )

        # Copy result back
        cuda.memcpy_dtoh_async(keep, d_keep, stream)
        stream.synchronize()

        # Cleanup
        d_boxes.free()
        d_scores.free()
        d_keep.free()
        d_num_keep.free()

        return np.where(keep > 0)[0]
