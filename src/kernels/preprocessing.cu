#include "fused_preprocess.cuh"
#include "../common/cuda_utils.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Optimized bilinear interpolation device function
__device__ __forceinline__ float bilinear_interpolate(
    const uint8_t* src, int src_width, int src_height, int channels,
    float x, float y, int c) {
    
    int x1 = __float2int_rd(x);
    int y1 = __float2int_rd(y);
    int x2 = min(x1 + 1, src_width - 1);
    int y2 = min(y1 + 1, src_height - 1);
    
    float dx = x - x1;
    float dy = y - y1;
    
    int idx_tl = (y1 * src_width + x1) * channels + c;
    int idx_tr = (y1 * src_width + x2) * channels + c;
    int idx_bl = (y2 * src_width + x1) * channels + c;
    int idx_br = (y2 * src_width + x2) * channels + c;
    
    float tl = src[idx_tl];
    float tr = src[idx_tr];
    float bl = src[idx_bl];
    float br = src[idx_br];
    
    float top = tl + dx * (tr - tl);
    float bottom = bl + dx * (br - bl);
    return top + dy * (bottom - top);
}

// Main fused preprocessing kernel
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
    const float* __restrict__ std_vals) {
    
    // Calculate global thread indices
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Bounds checking
    if (dst_x >= dst_width || dst_y >= dst_height || batch_idx >= batch_size) {
        return;
    }
    
    // Calculate source coordinates for bilinear interpolation
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;
    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;
    
    // Clamp source coordinates
    src_x = fmaxf(0.0f, fminf(src_x, src_width - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_height - 1.0f));
    
    // Calculate base pointers
    const uint8_t* src_frame = src_frames + batch_idx * src_width * src_height * channels;
    
    // Process each channel
    for (int c = 0; c < channels; ++c) {
        // Bilinear interpolation (resize)
        float pixel_val = bilinear_interpolate(src_frame, src_width, src_height, 
                                             channels, src_x, src_y, c);
        
        // Color space conversion (BGR -> RGB if needed)
        int channel_idx = c;
        if (bgr_to_rgb && channels == 3) {
            channel_idx = 2 - c; // Swap B and R channels
        }
        
        // Data type conversion (uint8 -> float) and normalization
        pixel_val = pixel_val * norm_factor;
        
        // Apply mean and std normalization if provided
        if (mean_vals && std_vals) {
            pixel_val = (pixel_val - mean_vals[channel_idx]) / std_vals[channel_idx];
        }
        
        // Tensor layout conversion: HWC -> CHW
        // CHW layout: [batch][channel][height][width]
        int dst_idx = batch_idx * channels * dst_height * dst_width +
                      channel_idx * dst_height * dst_width +
                      dst_y * dst_width + dst_x;
        
        dst_tensor[dst_idx] = pixel_val;
    }
}

// Optimized kernel for batch processing with shared memory
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
    const float* __restrict__ std_vals) {
    
    // Shared memory for caching normalization values
    __shared__ float shared_mean[4];
    __shared__ float shared_std[4];
    
    // Load normalization values to shared memory
    if (threadIdx.x < channels && threadIdx.y == 0 && threadIdx.z == 0) {
        if (mean_vals && std_vals) {
            shared_mean[threadIdx.x] = mean_vals[threadIdx.x];
            shared_std[threadIdx.x] = std_vals[threadIdx.x];
        } else {
            shared_mean[threadIdx.x] = 0.0f;
            shared_std[threadIdx.x] = 1.0f;
        }
    }
    __syncthreads();
    
    // Calculate global thread indices
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    
    // Bounds checking
    if (dst_x >= dst_width || dst_y >= dst_height || batch_idx >= batch_size) {
        return;
    }
    
    // Calculate source coordinates
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;
    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;
    
    src_x = fmaxf(0.0f, fminf(src_x, src_width - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_height - 1.0f));
    
    const uint8_t* src_frame = src_frames + batch_idx * src_width * src_height * channels;
    
    // Process each channel
    #pragma unroll
    for (int c = 0; c < 3; ++c) { // Assume RGB/BGR (3 channels)
        float pixel_val = bilinear_interpolate(src_frame, src_width, src_height, 
                                             channels, src_x, src_y, c);
        
        int channel_idx = (bgr_to_rgb && channels == 3) ? (2 - c) : c;
        
        // Normalize using shared memory values
        pixel_val = pixel_val * norm_factor;
        pixel_val = (pixel_val - shared_mean[channel_idx]) / shared_std[channel_idx];
        
        int dst_idx = batch_idx * channels * dst_height * dst_width +
                      channel_idx * dst_height * dst_width +
                      dst_y * dst_width + dst_x;
        
        dst_tensor[dst_idx] = pixel_val;
    }
}

// Host function to launch the kernel
cudaError_t launch_fused_preprocess(
    const uint8_t* src_frames,
    float* dst_tensor,
    int batch_size,
    int src_width, int src_height,
    int dst_width, int dst_height,
    int channels,
    bool bgr_to_rgb,
    float norm_factor,
    const float* mean_vals,
    const float* std_vals,
    cudaStream_t stream) {
    
    // Optimal block dimensions for different scenarios
    dim3 block_size;
    dim3 grid_size;
    
    if (batch_size == 1) {
        // Single frame optimization
        block_size = dim3(16, 16, 1);
        grid_size = dim3(
            (dst_width + block_size.x - 1) / block_size.x,
            (dst_height + block_size.y - 1) / block_size.y,
            1
        );
        
        fused_preprocess_kernel<<<grid_size, block_size, 0, stream>>>(
            src_frames, dst_tensor, batch_size,
            src_width, src_height, dst_width, dst_height, channels,
            bgr_to_rgb, norm_factor, mean_vals, std_vals
        );
    } else {
        // Batch processing optimization
        block_size = dim3(8, 8, 4);
        grid_size = dim3(
            (dst_width + block_size.x - 1) / block_size.x,
            (dst_height + block_size.y - 1) / block_size.y,
            (batch_size + block_size.z - 1) / block_size.z
        );
        
        fused_preprocess_batch_kernel<<<grid_size, block_size, 0, stream>>>(
            src_frames, dst_tensor, batch_size,
            src_width, src_height, dst_width, dst_height, channels,
            bgr_to_rgb, norm_factor, mean_vals, std_vals
        );
    }
    
    return cudaGetLastError();
}