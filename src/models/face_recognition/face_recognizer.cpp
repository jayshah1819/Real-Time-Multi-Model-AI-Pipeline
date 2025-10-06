// face_recognition_pipeline.cpp
#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <chrono>
#include <cuda_runtime.h>
#include <cuda.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include <vector>
#include <fstream>
using namespace nvinfer1;

// Forward declarations for missing types
struct BoundingBox {
  float x1, y1, x2, y2;
};

class FaceCropProcessor {
public:
  FaceCropProcessor(cudaStream_t) {}
  float* process_faces(float*, BoundingBox*, int, int, int, int&) { return nullptr; }
};

class FaceRecognitionPipeline {
#include <cmath>

using namespace nvinfer1;

class FaceRecognitionPipeline {
private:
  // TensorRT components
  std::unique_ptr<ICudaEngine> engine;
  std::unique_ptr<IExecutionContext> context;
  std::unique_ptr<IRuntime> runtime;

  // GPU buffers
  void *d_input;
  void *d_embeddings;

  // Stream for async operations
  cudaStream_t stream;

  // Face database for matching
  struct FaceEmbedding {
    std::string name;
    std::vector<float> embedding;
  };
  std::vector<FaceEmbedding> face_database;

  const int EMBEDDING_SIZE = 512;
  const int BATCH_SIZE = 32;

public:
  FaceRecognitionPipeline() {
    cudaStreamCreate(&stream);
    initialize_tensorrt();
  }

  ~FaceRecognitionPipeline() {
    if (d_input)
      cudaFree(d_input);
    if (d_embeddings)
      cudaFree(d_embeddings);
    cudaStreamDestroy(stream);
  }

  void initialize_tensorrt() {
    // Create TensorRT runtime
    Logger logger;
    runtime.reset(createInferRuntime(logger));

    // Build engine from ONNX with FP16 optimization
    build_engine_from_onnx("arcface_mobilenet.onnx");

    // Create execution context
    context.reset(engine->createExecutionContext());

    // Allocate GPU memory for input and output
    cudaMalloc(&d_input, BATCH_SIZE * 3 * 112 * 112 * sizeof(float));
    cudaMalloc(&d_embeddings, BATCH_SIZE * EMBEDDING_SIZE * sizeof(float));
  }

  void build_engine_from_onnx(const std::string &onnx_path) {
    // Create builder with FP16 support
    auto builder = std::unique_ptr<IBuilder>(createInferBuilder(Logger()));
    auto config =
        std::unique_ptr<IBuilderConfig>(builder->createBuilderConfig());

    // Enable FP16
    config->setFlag(BuilderFlag::kFP16);

    // Set memory pool limits
    config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, 1 << 30); // 1GB

    // Parse ONNX model
    auto network = std::unique_ptr<INetworkDefinition>(builder->createNetworkV2(
        1U << static_cast<uint32_t>(
            NetworkDefinitionCreationFlag::kEXPLICIT_BATCH)));

    auto parser = std::unique_ptr<nvonnxparser::IParser>(
        nvonnxparser::createParser(*network, Logger()));

    parser->parseFromFile(onnx_path.c_str(),
                          static_cast<int>(ILogger::Severity::kWARNING));

    // Optimize for batch processing
    auto profile = builder->createOptimizationProfile();
    profile->setDimensions("input", OptProfileSelector::kMIN,
                           Dims4{1, 3, 112, 112});
    profile->setDimensions("input", OptProfileSelector::kOPT,
                           Dims4{16, 3, 112, 112});
    profile->setDimensions("input", OptProfileSelector::kMAX,
                           Dims4{32, 3, 112, 112});
    config->addOptimizationProfile(profile);

    // Build engine
    engine.reset(builder->buildEngineWithConfig(*network, *config));

    // Serialize engine for faster loading next time
    save_engine("arcface_fp16.engine");
  }

  // Zero-copy inference directly from GPU face batch
  void infer_batch(float *d_face_batch, int batch_size, float *h_embeddings) {
    // Set batch size dynamically
    context->setBindingDimensions(0, Dims4{batch_size, 3, 112, 112});

    // Setup bindings - input is already in GPU memory!
    void *bindings[] = {d_face_batch, d_embeddings};

    // Enqueue inference
    auto start = std::chrono::high_resolution_clock::now();

    context->enqueueV2(bindings, stream, nullptr);
    cudaStreamSynchronize(stream);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("Face Recognition Inference: %.2f ms for %d faces\n",
           duration.count() / 1000.0f, batch_size);

    // Copy embeddings to host for matching
    cudaMemcpyAsync(h_embeddings, d_embeddings,
                    batch_size * EMBEDDING_SIZE * sizeof(float),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
  }

  // Face matching with cosine similarity
  std::vector<std::pair<std::string, float>> match_faces(float *embeddings,
                                                         int num_faces) {
    std::vector<std::pair<std::string, float>> results;

    for (int i = 0; i < num_faces; i++) {
      float *query_embedding = embeddings + i * EMBEDDING_SIZE;

      std::string best_match = "Unknown";
      float best_similarity = -1.0f;

      // Compare with database
      for (const auto &db_face : face_database) {
        float similarity = cosine_similarity(
            query_embedding, db_face.embedding.data(), EMBEDDING_SIZE);

        if (similarity > best_similarity && similarity > 0.6f) {
          best_similarity = similarity;
          best_match = db_face.name;
        }
      }

      results.push_back({best_match, best_similarity});
    }

    return results;
  }

  // Add face to database
  void add_face_to_database(const std::string &name, float *embedding) {
    FaceEmbedding face;
    face.name = name;
    face.embedding.assign(embedding, embedding + EMBEDDING_SIZE);

    // L2 normalize for cosine similarity
    float norm = 0.0f;
    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      norm += face.embedding[i] * face.embedding[i];
    }
    norm = std::sqrt(norm);

    for (int i = 0; i < EMBEDDING_SIZE; i++) {
      face.embedding[i] /= norm;
    }

    face_database.push_back(face);
  }

private:
  float cosine_similarity(const float *a, const float *b, int size) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;

    for (int i = 0; i < size; i++) {
      dot += a[i] * b[i];
      norm_a += a[i] * a[i];
      norm_b += b[i] * b[i];
    }

    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b));
  }

  void save_engine(const std::string &path) {
    auto serialized = std::unique_ptr<IHostMemory>(engine->serialize());
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char *>(serialized->data()),
               serialized->size());
  }

  void load_engine(const std::string &path) {
    std::ifstream file(path, std::ios::binary);
    std::vector<char> data((std::istreambuf_iterator<char>(file)),
                           std::istreambuf_iterator<char>());

    engine.reset(runtime->deserializeCudaEngine(data.data(), data.size()));
  }

  class Logger : public ILogger {
    void log(Severity severity, const char *msg) noexcept override {
      if (severity <= Severity::kWARNING)
        std::cout << msg << std::endl;
    }
  };
};

// Integrated multi-model pipeline
class MultiModelPipeline {
private:
  FaceRecognitionPipeline face_recognition;
  FaceCropProcessor *face_cropper;

  // Tracking state
  struct TrackedFace {
    int id;
    std::string name;
    cv::Rect2f bbox;
    std::vector<float> embedding;
    int missed_frames;
    float confidence;
  };

  std::vector<TrackedFace> tracked_faces;
  int next_face_id = 0;

public:
  MultiModelPipeline(cudaStream_t stream) {
    face_cropper = new FaceCropProcessor(stream);
  }

  ~MultiModelPipeline() { delete face_cropper; }

  // Main processing pipeline - completely on GPU
  void process_frame(float *d_frame, // Frame already on GPU from decoder
                     BoundingBox *detections, // YOLO detections
                     int num_detections, int frame_width, int frame_height) {
    auto start = std::chrono::high_resolution_clock::now();

    // Step 1: Crop faces on GPU (zero-copy)
    int num_valid_faces;
    float *d_face_batch =
        face_cropper->process_faces(d_frame, detections, num_detections,
                                    frame_width, frame_height, num_valid_faces);

    if (num_valid_faces == 0)
      return;

    // Step 2: Run face recognition (still on GPU)
    float *h_embeddings = new float[num_valid_faces * 512];
    face_recognition.infer_batch(d_face_batch, num_valid_faces, h_embeddings);

    // Step 3: Match and track faces
    update_tracking(detections, h_embeddings, num_valid_faces);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    printf("Total Face Pipeline: %.2f ms for %d faces\n",
           duration.count() / 1000.0f, num_valid_faces);

    delete[] h_embeddings;
  }

  void update_tracking(BoundingBox *detections, float *embeddings,
                       int num_faces) {
    // Simple tracking with embedding matching
    std::vector<bool> matched(num_faces, false);

    // Update existing tracks
    for (auto &track : tracked_faces) {
      float best_similarity = -1.0f;
      int best_idx = -1;

      for (int i = 0; i < num_faces; i++) {
        if (matched[i])
          continue;

        // Check spatial proximity
        float iou = calculate_iou(track.bbox, detections[i]);
        if (iou < 0.3f)
          continue;

        // Check embedding similarity
        float similarity = cosine_similarity(track.embedding.data(),
                                             embeddings + i * 512, 512);

        if (similarity > best_similarity) {
          best_similarity = similarity;
          best_idx = i;
        }
      }

      if (best_idx >= 0 && best_similarity > 0.5f) {
        // Update track
        matched[best_idx] = true;
        track.bbox =
            cv::Rect2f(detections[best_idx].x1, detections[best_idx].y1,
                       detections[best_idx].x2 - detections[best_idx].x1,
                       detections[best_idx].y2 - detections[best_idx].y1);
        track.missed_frames = 0;
        track.confidence = best_similarity;

        // Update embedding with momentum
        for (int j = 0; j < 512; j++) {
          track.embedding[j] =
              0.9f * track.embedding[j] + 0.1f * embeddings[best_idx * 512 + j];
        }
      } else {
        track.missed_frames++;
      }
    }

    // Remove lost tracks
    tracked_faces.erase(std::remove_if(tracked_faces.begin(),
                                       tracked_faces.end(),
                                       [](const TrackedFace &t) {
                                         return t.missed_frames > 10;
                                       }),
                        tracked_faces.end());

    // Add new tracks
    for (int i = 0; i < num_faces; i++) {
      if (!matched[i]) {
        TrackedFace new_track;
        new_track.id = next_face_id++;
        new_track.name = "Person_" + std::to_string(new_track.id);
        new_track.bbox = cv::Rect2f(detections[i].x1, detections[i].y1,
                                    detections[i].x2 - detections[i].x1,
                                    detections[i].y2 - detections[i].y1);
        new_track.embedding.assign(embeddings + i * 512,
                                   embeddings + (i + 1) * 512);
        new_track.missed_frames = 0;
        new_track.confidence = 1.0f;

        tracked_faces.push_back(new_track);
      }
    }
  }

private:
  float calculate_iou(const cv::Rect2f &a, const BoundingBox &b) {
    float x1 = std::max(a.x, b.x1);
    float y1 = std::max(a.y, b.y1);
    float x2 = std::min(a.x + a.width, b.x2);
    float y2 = std::min(a.y + a.height, b.y2);

    if (x2 < x1 || y2 < y1)
      return 0.0f;

    float intersection = (x2 - x1) * (y2 - y1);
    float area_a = a.width * a.height;
    float area_b = (b.x2 - b.x1) * (b.y2 - b.y1);
    float union_area = area_a + area_b - intersection;

    return intersection / union_area;
  }

  float cosine_similarity(const float *a, const float *b, int size) {
    float dot = 0.0f, norm_a = 0.0f, norm_b = 0.0f;
    for (int i = 0; i < size; i++) {
      dot += a[i] * b[i];
      norm_a += a[i] * a[i];
      norm_b += b[i] * b[i];
    }
    return dot / (std::sqrt(norm_a) * std::sqrt(norm_b) + 1e-6f);
  }
};