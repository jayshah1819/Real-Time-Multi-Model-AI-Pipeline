// cuda_face_crop.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>

#define FACE_SIZE 112 
#define MAX_FACES 32

struct BoundingBox {
    float x1, y1, x2, y2;
    float confidence;
    int class_id;
};


__device__ float bilinear_interpolate(const float* img, 
                                      int width, int height, int channels,
                                      float x, float y, int c) {
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= width - 1) x = width - 1.001f;
    if (y >= height - 1) y = height - 1.001f;
    
    int x0 = (int)x;
    int y0 = (int)y;
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    
    float dx = x - x0;
    float dy = y - y0;

    float v00 = img[(y0 * width + x0) * channels + c];
    float v10 = img[(y0 * width + x1) * channels + c];
    float v01 = img[(y1 * width + x0) * channels + c];
    float v11 = img[(y1 * width + x1) * channels + c];
    
    // Bilinear interpolation
    float v0 = v00 * (1 - dx) + v10 * dx;
    float v1 = v01 * (1 - dx) + v11 * dx;
    
    return v0 * (1 - dy) + v1 * dy;
}

// Zero-copy batch face cropping kernel
__global__ void batch_face_crop_kernel(
    const float* input_frame,      // Original frame in GPU memory (HWC format)
    const BoundingBox* boxes,      // Detection boxes from YOLO
    float* face_batch,             // Output batch of face tensors
    int* valid_faces,              // Mask indicating valid faces
    int frame_width,
    int frame_height,
    int num_detections,
    int max_batch_size
) {
    int face_idx = blockIdx.x;
    if (face_idx >= num_detections || face_idx >= max_batch_size) return;
    
    const BoundingBox& box = boxes[face_idx];
    
    // Filter for person class (class_id 0 in COCO)
    if (box.class_id != 0 || box.confidence < 0.5f) {
        valid_faces[face_idx] = 0;
        return;
    }
    
    valid_faces[face_idx] = 1;
    
    // Calculate crop region with padding
    float box_width = box.x2 - box.x1;
    float box_height = box.y2 - box.y1;
    float padding = 0.2f; // 20% padding for better face coverage
    
    float x1 = fmaxf(0.0f, box.x1 - box_width * padding);
    float y1 = fmaxf(0.0f, box.y1 - box_height * padding);
    float x2 = fminf(frame_width - 1.0f, box.x2 + box_width * padding);
    float y2 = fminf(frame_height - 1.0f, box.y2 + box_height * padding);
    
    float crop_width = x2 - x1;
    float crop_height = y2 - y1;
    
    int tid_x = threadIdx.x;
    int tid_y = threadIdx.y;
    int tid_c = threadIdx.z;
    
    // Each thread processes one pixel of the output face
    if (tid_x < FACE_SIZE && tid_y < FACE_SIZE && tid_c < 3) {
        // Map output coordinates to input coordinates
        float src_x = x1 + (tid_x * crop_width) / FACE_SIZE;
        float src_y = y1 + (tid_y * crop_height) / FACE_SIZE;
        
        // Bilinear interpolation for smooth resizing
        float pixel_value = bilinear_interpolate(
            input_frame, frame_width, frame_height, 3,
            src_x, src_y, tid_c
        );
        
        // Write to output batch (CHW format for TensorRT)
        int output_idx = face_idx * (3 * FACE_SIZE * FACE_SIZE) +
                        tid_c * (FACE_SIZE * FACE_SIZE) +
                        tid_y * FACE_SIZE + tid_x;
        
        // Normalize to [-1, 1] for face recognition models
        face_batch[output_idx] = (pixel_value / 255.0f - 0.5f) / 0.5f;
    }
}

// Optimized kernel for face alignment (5-point landmarks)
__global__ void face_alignment_kernel(
    float* face_batch,
    const float* landmarks,  // 5 facial landmarks per face
    int num_faces
) {
    int face_idx = blockIdx.x;
    if (face_idx >= num_faces) return;
    
    int pixel_idx = threadIdx.x + threadIdx.y * FACE_SIZE;
    if (pixel_idx >= FACE_SIZE * FACE_SIZE) return;
    
    // Simplified alignment using landmarks
    // This is a placeholder - real implementation would include
    // affine transformation based on eye positions
    
    // For now, just ensure data coherency
    __syncthreads();
}

// Host wrapper class for face cropping pipeline
class FaceCropProcessor {
private:
    float* d_face_batch;
    BoundingBox* d_boxes;
    int* d_valid_faces;
    cudaStream_t stream;
    
public:
    FaceCropProcessor(cudaStream_t processing_stream) : stream(processing_stream) {
        // Allocate GPU memory for face batch
        cudaMalloc(&d_face_batch, MAX_FACES * 3 * FACE_SIZE * FACE_SIZE * sizeof(float));
        cudaMalloc(&d_boxes, MAX_FACES * sizeof(BoundingBox));
        cudaMalloc(&d_valid_faces, MAX_FACES * sizeof(int));
    }
    
    ~FaceCropProcessor() {
        cudaFree(d_face_batch);
        cudaFree(d_boxes);
        cudaFree(d_valid_faces);
    }
    
    float* process_faces(
        const float* d_frame,
        const BoundingBox* h_boxes,
        int num_detections,
        int frame_width,
        int frame_height,
        int& num_valid_faces
    ) {
        // Copy boxes to GPU (this is minimal data transfer)
        cudaMemcpyAsync(d_boxes, h_boxes, 
                       num_detections * sizeof(BoundingBox),
                       cudaMemcpyHostToDevice, stream);
        
        // Launch kernel with optimal block configuration
        dim3 block_dim(FACE_SIZE, FACE_SIZE, 3);
        dim3 grid_dim(min(num_detections, MAX_FACES));
        
        batch_face_crop_kernel<<<grid_dim, block_dim, 0, stream>>>(
            d_frame,
            d_boxes,
            d_face_batch,
            d_valid_faces,
            frame_width,
            frame_height,
            num_detections,
            MAX_FACES
        );
        
        // Count valid faces (async operation)
        int* h_valid_faces = new int[MAX_FACES];
        cudaMemcpyAsync(h_valid_faces, d_valid_faces,
                       MAX_FACES * sizeof(int),
                       cudaMemcpyDeviceToHost, stream);
        cudaStreamSynchronize(stream);
        
        num_valid_faces = 0;
        for (int i = 0; i < min(num_detections, MAX_FACES); i++) {
            num_valid_faces += h_valid_faces[i];
        }
        
        delete[] h_valid_faces;
        
        return d_face_batch;  // Return GPU pointer - zero copy!
    }
    
    // Performance monitoring
    void profile_kernel(int num_detections) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start, stream);
        
        // Dummy run for profiling
        dim3 block_dim(FACE_SIZE, FACE_SIZE, 3);
        dim3 grid_dim(min(num_detections, MAX_FACES));
        
        batch_face_crop_kernel<<<grid_dim, block_dim, 0, stream>>>(
            nullptr, nullptr, d_face_batch, d_valid_faces,
            1920, 1080, num_detections, MAX_FACES
        );
        
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        printf("Face Crop Kernel Time: %.3f ms for %d faces\n", 
               milliseconds, num_detections);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }
};

// Memory pool for zero-copy operations
class GPUMemoryPool {
private:
    struct MemoryBlock {
        void* ptr;
        size_t size;
        bool in_use;
    };
    
    std::vector<MemoryBlock> blocks;
    cudaStream_t stream;
    
public:
    GPUMemoryPool(cudaStream_t s) : stream(s) {}
    
    void* allocate(size_t size) {
        // Find available block or allocate new
        for (auto& block : blocks) {
            if (!block.in_use && block.size >= size) {
                block.in_use = true;
                return block.ptr;
            }
        }
        
        // Allocate new block
        void* ptr;
        cudaMalloc(&ptr, size);
        blocks.push_back({ptr, size, true});
        return ptr;
    }
    
    void deallocate(void* ptr) {
        for (auto& block : blocks) {
            if (block.ptr == ptr) {
                block.in_use = false;
                return;
            }
        }
    }
    
    ~GPUMemoryPool() {
        for (auto& block : blocks) {
            cudaFree(block.ptr);
        }
    }
};

// Optimized batch normalization for face tensors
__global__ void batch_normalize_faces(
    float* face_batch,
    int batch_size,
    const float* mean,
    const float* std
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * 3 * FACE_SIZE * FACE_SIZE;
    
    if (idx < total_elements) {
        int channel = (idx / (FACE_SIZE * FACE_SIZE)) % 3;
        face_batch[idx] = (face_batch[idx] - mean[channel]) / std[channel];
    }
}