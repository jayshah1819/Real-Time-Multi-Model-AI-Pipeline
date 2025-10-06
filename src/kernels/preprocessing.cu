
// This file implements CUDA kernels for preprocessing image batches for deep learning inference.
// The main steps are:
// 1. Takes raw input images (uint8 format).
// 2. Resizes them to target width/height with bilinear interpolation.
// 3. Converts from BGR to RGB if needed.
// 4. Normalizes pixel values (scaling, subtract mean, divide by std).
// 5. Packs the results into a GPU tensor suitable for inference.


// Bilinear interpolation explanation:
// Bilinear interpolation is a resampling technique used to estimate the value of a pixel
// at a non-integer (floating point) position within a 2D grid (such as an image).
// It works by performing linear interpolation first in one direction (e.g., x-axis),
// then in the other direction (e.g., y-axis), using the four nearest neighboring pixels.
// The result is a smooth, weighted average that provides better visual quality than
// nearest-neighbor interpolation, especially when resizing images.


#include "fused_preprocess.cuh"
#include "../common/cuda_utils.hpp"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


// Bilinear interpolation function:
__device__ __forceinline__ float bilinear_interpolate(
    const uint8_t* src, int src_width, int src_height, int channels,
    float x, float y, int c) {
    

    // Calculate coordinates of the 4 surrounding pixels
    //top-left, top-right, bottom-left, bottom-right
    int x1= __float2int_rd(x);
    int y1 = __float2int_rd(y);
    int x2= min(x1 + 1, src_width - 1);
    int y2= min(y1 + 1, src_height - 1);
    
    //fractions for 2x2 pixel interpolation
    float dx = x - x1;
    float dy = y - y1;
    
    //Compute 1D indices for the 2x2 pixels
    int idx_tl =(y1 * src_width + x1) * channels + c;
    int idx_tr =(y1 * src_width + x2) * channels + c;
    int idx_bl= (y2 * src_width + x1) * channels + c;
    int idx_br= (y2 * src_width + x2) * channels + c;
    
    float tl= src[idx_tl];
    float tr= src[idx_tr];
    float bl= src[idx_bl];
    float br= src[idx_br];

    //Interpolate horizontally
    float top= tl + dx * (tr - tl);
    float bottom= bl + dx * (br - bl);

    return top + dy * (bottom - top);
}

/**
 * 
 *
 * Each GPU thread processes a single output pixel for a given image. The kernel computes the corresponding
 * source pixel location using geometric mapping, and applies bilinear interpolation to obtain a smooth pixel value.
 * Optionally, the kernel converts the color format from BGR to RGB. It then applies scaling and normalization
 * to the pixel value as required by the model's input specification. The final result is stored in a contiguous
 * GPU tensor, ready for consumption by deep learning models.
 *
 * 
 */
//Each GPU thread handles one output pixel of one image.

// Step 1: Calculate output pixel coordinates (dst_x, dst_y) and batch index
// Step 2: Perform bounds check to ignore threads outside the image or batch
// Step 3: Compute scale factors for width and height (scale_x, scale_y)
// Step 4: Map destination pixel coordinate to source pixel coordinate (src_x, src_y)
// Step 5: Clamp source coordinates to valid bounds of the source image
// Step 6: Select the current image pointer in the batch (src_frame)
// Step 7: Loop over channels (c = 0 to channels-1)
// Step 8: Perform bilinear interpolation to get pixel value at (src_x, src_y) for channel c
// Step 9: Swap channels if converting BGR to RGB
// Step 10: Scale pixel value by normalization factor (norm_factor, e.g., 1/255)
// Step 11: Normalize pixel value using mean and std (if provided)
// Step 12: Compute 1D index in output tensor (NCHW layout)
// Step 13: Store the processed pixel value into dst_tensor at the computed index


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
    


    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y= blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx =blockIdx.z * blockDim.z + threadIdx.z;
//Bounds check
    if (dst_x >= dst_width || dst_y>= dst_height || batch_idx >= batch_size) {
        return;
    }
    

    float scale_x= (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;



    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y= (dst_y + 0.5f) * scale_y - 0.5f;


//Clamp to valid bounds of the source image
     float x_min=fminf(src_x, src_width - 1.0f) 
     float y_min= fminf(src_y, src_height - 1.0f);

    src_x= fmaxf(0.0f, x_min);
    src_y = fmaxf(0.0f, y_min);
    

    const uint8_t* src_frame = src_frames + batch_idx * src_width * src_height * channels;
    

    for (int c = 0; c < channels; ++c) {

        float pixel_val = bilinear_interpolate(src_frame, src_width, src_height, 
                                             channels, src_x, src_y, c);
        

        int channel_idx = c;
        //coverting BGR to RGB by swapping channels and if we want to swap channels it would 2-c
        if (bgr_to_rgb && channels == 3) {
            channel_idx = 2 - c; 
        }
        //norm factor is typically 1/255.0 to scale 0-255 to 0-1
       //as our pixel values are in range 0-255
        pixel_val = pixel_val * norm_factor;
        
//Normalize using mean and std 
        if (mean_vals && std_vals) {
            pixel_val = (pixel_val - mean_vals[channel_idx]) / std_vals[channel_idx];
        }
//Compute 1D index in output tensor  using NCHW layout
        int dst_idx = batch_idx * channels * dst_height * dst_width +
                      channel_idx * dst_height * dst_width +
                      dst_y * dst_width + dst_x;
        //Store the processed pixel value
        dst_tensor[dst_idx] = pixel_val;
    }
}


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
    

    __shared__ float shared_mean[4];
    __shared__ float shared_std[4];
    

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
    
//Calculate output pixel coordinates and batch index
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;
    int batch_idx = blockIdx.z * blockDim.z + threadIdx.z;
    

    if (dst_x >= dst_width || dst_y >= dst_height || batch_idx >= batch_size) {
        return;
    }
    
 //Compute scale factors.
    float scale_x = (float)src_width / dst_width;
    float scale_y = (float)src_height / dst_height;

//Maps destination pixel coordinate → source pixel coordinate.
//The +0.5 and -0.5 shifts are to align pixel centers (not edges), 
//which makes bilinear interpolation smoother and less blurry.

    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;


//Clamp to valid bounds of the source image
    src_x = fmaxf(0.0f, fminf(src_x, src_width - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_height - 1.0f));

//current frame pointer
    const uint8_t* src_frame = src_frames + batch_idx * src_width * src_height * channels;
    
//we will loop around chanels R,G,B (or B,G,R)
// pragma unroll tells compiler to unroll the loop for better performance
    #pragma unroll
    for (int c = 0; c < 3; ++c) {
        float pixel_val = bilinear_interpolate(src_frame, src_width, src_height, 
                                             channels, src_x, src_y, c);
        int channel_idx;
        if (bgr_to_rgb && channels == 3) {
            channel_idx = 2 - c;
        } else {
            channel_idx = c;
        }
        
      
        pixel_val = pixel_val * norm_factor;
        pixel_val = (pixel_val - shared_mean[channel_idx]) / shared_std[channel_idx];
        
        int dst_idx = batch_idx * channels * dst_height * dst_width +
                      channel_idx * dst_height * dst_width +
                      dst_y * dst_width + dst_x;
        
        dst_tensor[dst_idx] = pixel_val;
    }
}

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
    

    dim3 block_size;
    dim3 grid_size;
    
    if (batch_size == 1) { 

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