#!/usr/bin/env python3
"""
Comprehensive benchmarking suite for the multi-model inference pipeline
"""

import time
import numpy as np
import torch
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PipelineBenchmark:
    """Benchmarks different components of the pipeline"""

    def __init__(self, config_path: str):
        with open(config_path) as f:
            self.config = json.load(f)

        self.results = {}
        self.warmup_iterations = 50
        self.benchmark_iterations = 1000

    def benchmark_preprocessing(self, frame_size: Tuple[int, int] = (3840, 2160)):
        """Benchmark preprocessing kernel vs OpenCV-CUDA"""
        from src.kernels import preprocessing_kernel

        # Create test data
        test_frames = np.random.randint(
            0, 255, (self.benchmark_iterations, *frame_size, 3), dtype=np.uint8
        )

        # Warmup
        for i in range(self.warmup_iterations):
            _ = preprocessing_kernel.preprocess_frame(test_frames[0])

        # Benchmark CUDA kernel
        cuda_times = []
        cuda.Context.synchronize()

        for i in range(self.benchmark_iterations):
            start = time.perf_counter()
            _ = preprocessing_kernel.preprocess_frame(test_frames[i])
            cuda.Context.synchronize()
            cuda_times.append(time.perf_counter() - start)

        # Benchmark OpenCV-CUDA
        cv_times = []
        gpu_mat = cv2.cuda_GpuMat()

        for i in range(self.benchmark_iterations):
            start = time.perf_counter()
            gpu_mat.upload(test_frames[i])
            resized = cv2.cuda.resize(gpu_mat, (640, 640))
            normalized = cv2.cuda.divide(resized, 255.0)
            cv_times.append(time.perf_counter() - start)

        self.results["preprocessing"] = {
            "cuda_kernel_mean_ms": np.mean(cuda_times) * 1000,
            "cuda_kernel_std_ms": np.std(cuda_times) * 1000,
            "opencv_cuda_mean_ms": np.mean(cv_times) * 1000,
            "opencv_cuda_std_ms": np.std(cv_times) * 1000,
            "speedup": np.mean(cv_times) / np.mean(cuda_times),
        }

    def benchmark_yolo(self):
        """Benchmark INT8 YOLO inference"""
        from src.models.object_detection import yolo_int8

        engine = yolo_int8.load_engine(self.config["yolo_engine_path"])
        context = engine.create_execution_context()

        # Create random input batch
        batch_size = 16
        input_batch = np.random.rand(batch_size, 3, 640, 640).astype(np.float32)

        # Allocate GPU memory
        d_input = cuda.mem_alloc(input_batch.nbytes)
        d_output = cuda.mem_alloc(batch_size * 8400 * 85 * 4)  # YOLO output size

        # Warmup
        for i in range(self.warmup_iterations):
            cuda.memcpy_htod(d_input, input_batch)
            context.execute_v2([int(d_input), int(d_output)])

        # Benchmark
        times = []
        for i in range(self.benchmark_iterations):
            start = time.perf_counter()
            cuda.memcpy_htod(d_input, input_batch)
            context.execute_v2([int(d_input), int(d_output)])
            cuda.Context.synchronize()
            times.append(time.perf_counter() - start)

        self.results["yolo_int8"] = {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "fps": batch_size / np.mean(times),
            "batch_size": batch_size,
        }

    def benchmark_face_recognition(self):
        """Benchmark face recognition pipeline"""
        from src.models.face_recognition import face_reco_pipeline

        pipeline = face_reco_pipeline.FaceRecognitionPipeline(
            model_path=self.config["arcface_model_path"],
            engine_path=self.config["arcface_engine_path"],
        )

        # Create test data
        frame = np.random.randint(0, 255, (2160, 3840, 3), dtype=np.uint8)
        bboxes = np.array(
            [
                [100, 200, 300, 500],
                [800, 150, 1000, 450],
                [1500, 300, 1700, 650],
                [2000, 100, 2200, 400],
            ],
            dtype=np.float32,
        )

        # Upload frame to GPU
        frame_size = frame.nbytes
        d_frame = cuda.mem_alloc(frame_size)
        cuda.memcpy_htod(d_frame, frame.astype(np.float32))

        # Warmup
        for i in range(self.warmup_iterations):
            _ = pipeline.process_frame_with_detections(d_frame, bboxes, (2160, 3840))

        # Benchmark
        times = []
        for i in range(self.benchmark_iterations):
            start = time.perf_counter()
            _ = pipeline.process_frame_with_detections(d_frame, bboxes, (2160, 3840))
            cuda.Context.synchronize()
            times.append(time.perf_counter() - start)

        self.results["face_recognition"] = {
            "mean_latency_ms": np.mean(times) * 1000,
            "std_latency_ms": np.std(times) * 1000,
            "faces_per_second": len(bboxes) / np.mean(times),
        }

    def benchmark_full_pipeline(self):
        """Benchmark complete multi-model pipeline"""
        pass  # TODO: Implement full pipeline benchmark

    def save_results(self, output_path: str):
        """Save benchmark results to JSON file"""
        with open(output_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Benchmark results saved to {output_path}")


def main():
    benchmark = PipelineBenchmark("configs/benchmark_config.json")

    # Run individual component benchmarks
    benchmark.benchmark_preprocessing()
    benchmark.benchmark_yolo()
    benchmark.benchmark_face_recognition()

    # Save results
    benchmark.save_results("results/benchmark_results.json")


if __name__ == "__main__":
    main()
