#ifndef FUSED_PREPROCESS_CUH
#define FUSED_PREPROCESS_CUH

#include <cuda_runtime.h>
#include <cstdint>

/**
 * @brief Fused preprocessing kernel for video frames
 * 
 * Performs resize, color space conversion, data type conversion,
 * normalization, and tensor layout conversion in a single kernel pass.
 */

/**
 * @brief Launch fused preprocessing kernel
 * 
 * @param src_frames Input frames in GPU memory (uint8_t, HWC format)
 * @param dst_tensor Output tensor in GPU memory (float, CHW format)
 * @param batch_size Number of frames to process
 * @param src_width Source frame width
 * @param src_height Source frame height
 * @param dst_width Target width after resize
 * @param dst_height Target height after resize
 * @param channels Number of channels (typically 3 for RGB/BGR)
 * @param bgr_to_rgb Whether to convert BGR to RGB
 * @param norm_factor Normalization factor (typically 1/255.0)
 * @param mean_vals Channel mean values for normalization (can be nullptr)
 * @param std_vals Channel standard deviation values (can be nullptr)
 * @param stream CUDA stream for kernel execution
 * @return cudaError_t CUDA error code
 */
cudaError_t launch_fused_preprocess(
    const uint8_t* src_frames,
    float* dst_tensor,
    int batch_size,
    int src_width, int src_height,
    int dst_width, int dst_height,
    int channels,
    bool bgr_to_rgb = true,
    float norm_factor = 1.0f / 255.0f,
    const float* mean_vals = nullptr,
    const float* std_vals = nullptr,
    cudaStream_t stream = 0
);

// Kernel declarations for direct access if needed
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

#endif 