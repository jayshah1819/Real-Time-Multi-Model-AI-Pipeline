// preprocessing_kernel.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_SIZE 16
#define MAX_BATCH_SIZE 32

// Fused preprocessing kernel for video frames
// Operations: Resize → RGB→BGR → Float conversion → Normalization → CHW layout
__global__ void preprocess_video_frame(
    const unsigned char* input,  // Input: HWC uint8 BGR
    float* output,              // Output: NCHW float32 normalized
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    float* mean,               // 3-element array for channel means
    float* std                 // 3-element array for channel stds
) {
    const int out_x = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_y = blockIdx.y * blockDim.y + threadIdx.y;
    const int batch_idx = blockIdx.z;
    
    if (out_x >= output_width || out_y >= output_height || batch_idx >= MAX_BATCH_SIZE)
        return;
        
    // Calculate input coordinates with bilinear interpolation
    const float scale_x = static_cast<float>(input_width) / output_width;
    const float scale_y = static_cast<float>(input_height) / output_height;
    
    const float in_x = (out_x + 0.5f) * scale_x - 0.5f;
    const float in_y = (out_y + 0.5f) * scale_y - 0.5f;
    
    const int in_x0 = max(0, static_cast<int>(floorf(in_x)));
    const int in_y0 = max(0, static_cast<int>(floorf(in_y)));
    const int in_x1 = min(input_width - 1, in_x0 + 1);
    const int in_y1 = min(input_height - 1, in_y0 + 1);
    
    const float alpha = in_x - in_x0;
    const float beta = in_y - in_y0;
    
    // Process each channel
    for (int c = 0; c < 3; c++) {
        // Sample four nearest pixels
        const unsigned char p00 = input[(in_y0 * input_width + in_x0) * 3 + (2-c)]; // BGR to RGB
        const unsigned char p01 = input[(in_y0 * input_width + in_x1) * 3 + (2-c)];
        const unsigned char p10 = input[(in_y1 * input_width + in_x0) * 3 + (2-c)];
        const unsigned char p11 = input[(in_y1 * input_width + in_x1) * 3 + (2-c)];
        
        // Bilinear interpolation
        const float top = p00 * (1.0f - alpha) + p01 * alpha;
        const float bottom = p10 * (1.0f - alpha) + p11 * alpha;
        const float pixel = top * (1.0f - beta) + bottom * beta;
        
        // Normalize and store in CHW format
        const float normalized = (pixel / 255.0f - mean[c]) / std[c];
        output[
            batch_idx * (3 * output_height * output_width) +
            c * (output_height * output_width) +
            out_y * output_width +
            out_x
        ] = normalized;
    }
}

// NMS kernel for object detection post-processing
__global__ void nms_kernel(
    const float* boxes,       // [N, 4] boxes in (x1, y1, x2, y2) format
    const float* scores,      // [N] confidence scores
    int* keep,               // [N] output indices
    int* num_keep,          // Number of boxes after NMS
    const int n_boxes,
    const float nms_thresh
) {
    const int row_start = blockIdx.y;
    const int col_start = blockIdx.x;
    
    const int row_size = min(n_boxes - row_start * blockDim.y, blockDim.y);
    const int col_size = min(n_boxes - col_start * blockDim.x, blockDim.x);
    
    __shared__ float block_boxes[BLOCK_SIZE * 4];
    
    if (threadIdx.x < col_size) {
        block_boxes[threadIdx.x * 4 + 0] = boxes[(col_start * blockDim.x + threadIdx.x) * 4 + 0];
        block_boxes[threadIdx.x * 4 + 1] = boxes[(col_start * blockDim.x + threadIdx.x) * 4 + 1];
        block_boxes[threadIdx.x * 4 + 2] = boxes[(col_start * blockDim.x + threadIdx.x) * 4 + 2];
        block_boxes[threadIdx.x * 4 + 3] = boxes[(col_start * blockDim.x + threadIdx.x) * 4 + 3];
    }
    
    __syncthreads();
    
    if (threadIdx.x < row_size) {
        const int cur_box_idx = row_start * blockDim.y + threadIdx.x;
        const float cur_box_x1 = boxes[cur_box_idx * 4 + 0];
        const float cur_box_y1 = boxes[cur_box_idx * 4 + 1];
        const float cur_box_x2 = boxes[cur_box_idx * 4 + 2];
        const float cur_box_y2 = boxes[cur_box_idx * 4 + 3];
        const float cur_box_area = (cur_box_x2 - cur_box_x1) * (cur_box_y2 - cur_box_y1);
        
        for (int i = 0; i < col_size; i++) {
            float box_x1 = block_boxes[i * 4 + 0];
            float box_y1 = block_boxes[i * 4 + 1];
            float box_x2 = block_boxes[i * 4 + 2];
            float box_y2 = block_boxes[i * 4 + 3];
            
            float x1 = max(cur_box_x1, box_x1);
            float y1 = max(cur_box_y1, box_y1);
            float x2 = min(cur_box_x2, box_x2);
            float y2 = min(cur_box_y2, box_y2);
            
            float w = max(0.0f, x2 - x1);
            float h = max(0.0f, y2 - y1);
            float inter = w * h;
            
            float box_area = (box_x2 - box_x1) * (box_y2 - box_y1);
            float union_area = cur_box_area + box_area - inter;
            float iou = inter / union_area;
            
            if (iou > nms_thresh) {
                if (scores[cur_box_idx] <= scores[col_start * blockDim.x + i]) {
                    keep[cur_box_idx] = 0;
                }
            }
        }
    }
}

extern "C" {

// Host wrapper for preprocessing kernel
void launch_preprocess_kernel(
    const unsigned char* input,
    float* output,
    int input_height,
    int input_width,
    int output_height,
    int output_width,
    int batch_size,
    float* mean,
    float* std,
    cudaStream_t stream
) {
    dim3 block(16, 16);
    dim3 grid(
        (output_width + block.x - 1) / block.x,
        (output_height + block.y - 1) / block.y,
        batch_size
    );
    
    preprocess_video_frame<<<grid, block, 0, stream>>>(
        input, output, input_height, input_width,
        output_height, output_width, mean, std
    );
}

// Host wrapper for NMS kernel
void launch_nms_kernel(
    const float* boxes,
    const float* scores,
    int* keep,
    int* num_keep,
    int n_boxes,
    float nms_thresh,
    cudaStream_t stream
) {
    dim3 block(BLOCK_SIZE);
    dim3 grid((n_boxes + block.x - 1) / block.x, (n_boxes + block.y - 1) / block.y);
    
    nms_kernel<<<grid, block, 0, stream>>>(
        boxes, scores, keep, num_keep, n_boxes, nms_thresh
    );
}

}