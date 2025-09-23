#ifndef FUSED_PREPROCESS_CUH
#define FUSED_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <cstdint>

/*
 * Fused preprocessing for image batches on GPU.
 *
 * This kernel does everything in one pass:
 * 1. Resize images with bilinear interpolation
 * 2. Convert BGR <-> RGB if needed
 * 3. Normalize pixel values (scale, mean, std)
 * 4. Store result in NCHW layout
 *
 * Works on uint8 input images and produces float output.
 */

// Launch the fused preprocessing kernel
// Handles batch or single images
cudaError_t launch_fused_preprocess(
    const uint8_t* src_frames,   // Input images in GPU memory (HWC)
    float* dst_tensor,           // Output tensor in GPU memory (CHW)
    int batch_size,              // Number of images
    int src_width, int src_height,
    int dst_width, int dst_height,
    int channels,                // Number of channels (3 for RGB/BGR)
    bool bgr_to_rgb = true,      // Convert BGR to RGB if true
    float norm_factor = 1.0f / 255.0f, // Scale factor to convert 0-255 -> 0-1
    const float* mean_vals = nullptr,  // Per-channel mean for normalization
    const float* std_vals = nullptr,   // Per-channel std for normalization
    cudaStream_t stream = 0           // CUDA stream
);

// Raw kernel declarations (if you want to call them directly)
__global__ void fused_preprocess_kernel(
    const uint8_t* __restrict__ src_frames,
    float* __restrict__ dst_tensor,
    int batch_size,
    int src_width, int src_height,
    int dst_width, int dst_height,
    int channels,
    bool bgr_to_rgb,
    float norm_factor,
    const float* __restrict__ mean_vals,
    const float* __restrict__ std_vals
);

__global__ void fused_preprocess_batch_kernel(
    const uint8_t* __restrict__ src_frames,
    float* __restrict__ dst_tensor,
    int batch_size,
    int src_width, int src_height,
    int dst_width, int dst_height,
    int channels,
    bool bgr_to_rgb,
    float norm_factor,
    const float* __restrict__ mean_vals,
    const float* __restrict__ std_vals
);

#endif // FUSED_PREPROCESS_CUH
