#include "lm_head.h"
#include "opencl_context.h"
#include <stdexcept>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <CL/cl.h>

namespace cartesia_opencl {

LMHead::LMHead(OpenCLContextManager* ctx, int d_model, int vocab_size)
    : ctx_(ctx)
    , d_model_(d_model)
    , vocab_size_(vocab_size)
    , chunk_size_(0)
    , num_chunks_(0)
    , weights_initialized_(false)
    , output_buffer_(nullptr)
    , output_buffer_size_(0)
{
    if (!ctx_) {
        throw std::runtime_error("OpenCLContext is null");
    }
    
    // Calculate chunk size to fit within device memory limit
    cl_device_id device = ctx_->getDevice();
    cl_ulong max_alloc_size = 0;
    cl_int err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc_size, nullptr);
    if (err != CL_SUCCESS) {
        max_alloc_size = 256 * 1024 * 1024;  // Default to 256 MB if query fails
    }
    
    // Calculate how many vocab tokens fit in one chunk (leave 10% safety margin)
    size_t safe_alloc_size = static_cast<size_t>(max_alloc_size * 0.9);
    chunk_size_ = safe_alloc_size / (d_model * sizeof(float));
    num_chunks_ = (vocab_size + chunk_size_ - 1) / chunk_size_;  // Ceiling division
    
    size_t total_params = static_cast<size_t>(vocab_size) * d_model;
    size_t total_size_mb = (total_params * sizeof(float)) / (1024 * 1024);
    size_t chunk_size_mb = (chunk_size_ * d_model * sizeof(float)) / (1024 * 1024);
    
    std::cout << "  [LMHead] d_model=" << d_model 
              << ", vocab_size=" << vocab_size 
              << ", params=" << total_params << std::endl;
    std::cout << "  [LMHead] Chunking: " << num_chunks_ << " chunks × " 
              << chunk_size_ << " tokens = " << chunk_size_mb << " MB/chunk, "
              << total_size_mb << " MB total" << std::endl;
}

LMHead::~LMHead() {
    for (cl_mem buffer : weights_buffers_) {
        if (buffer) clReleaseMemObject(buffer);
    }
    if (output_buffer_) clReleaseMemObject(output_buffer_);
}

void LMHead::initializeWeights(const std::vector<float>& weights) {
    // Weights are [vocab_size, d_model] but stored row-major
    // For matmul, we need them transposed or use appropriate kernel
    if (weights.size() != static_cast<size_t>(vocab_size_ * d_model_)) {
        throw std::runtime_error("Invalid LM head weights size");
    }
    
    cl_context context = ctx_->getContext();
    cl_int err;
    
    // Split weights into chunks
    weights_buffers_.clear();
    weights_buffers_.reserve(num_chunks_);
    
    for (int chunk_idx = 0; chunk_idx < num_chunks_; ++chunk_idx) {
        // Calculate this chunk's range
        int start_token = chunk_idx * chunk_size_;
        int end_token = std::min(start_token + chunk_size_, vocab_size_);
        int chunk_vocab_size = end_token - start_token;
        
        size_t chunk_buffer_size = chunk_vocab_size * d_model_ * sizeof(float);
        const float* chunk_data = weights.data() + (start_token * d_model_);
        
        cl_mem chunk_buffer = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            chunk_buffer_size,
            (void*)chunk_data,
            &err
        );
        
        if (err != CL_SUCCESS || !chunk_buffer) {
            // Clean up previously created chunks
            for (cl_mem buf : weights_buffers_) {
                if (buf) clReleaseMemObject(buf);
            }
            weights_buffers_.clear();
            
            std::string err_msg = "Failed to create LM head chunk " + std::to_string(chunk_idx) + 
                                ": error " + std::to_string(err);
            if (err == CL_INVALID_BUFFER_SIZE) {
                err_msg += " (CL_INVALID_BUFFER_SIZE - buffer too large)";
                err_msg += "\n  Chunk buffer size: " + std::to_string(chunk_buffer_size) + " bytes";
                err_msg += "\n  Chunk vocab size: " + std::to_string(chunk_vocab_size);
            } else if (err == CL_MEM_OBJECT_ALLOCATION_FAILURE) {
                err_msg += " (CL_MEM_OBJECT_ALLOCATION_FAILURE - out of memory)";
                err_msg += "\n  Chunk buffer size: " + std::to_string(chunk_buffer_size) + " bytes";
            }
            throw std::runtime_error(err_msg);
        }
        
        weights_buffers_.push_back(chunk_buffer);
        std::cout << "  [LMHead] Chunk " << chunk_idx << ": tokens [" << start_token 
                  << ", " << end_token << "), size=" << (chunk_buffer_size / 1024 / 1024) << " MB" << std::endl;
    }
    
    weights_initialized_ = true;
    std::cout << "  [LMHead] ✓ Created " << num_chunks_ << " weight chunks" << std::endl;
}

cl_mem LMHead::forward(cl_mem hidden_states, int batch_size, cl_command_queue queue) {
    if (!weights_initialized_) {
        throw std::runtime_error("LM head weights not initialized");
    }
    
    cl_context context = ctx_->getContext();
    size_t output_size = batch_size * vocab_size_ * sizeof(float);
    
    // Allocate or resize output buffer
    if (!output_buffer_ || output_buffer_size_ < output_size) {
        if (output_buffer_) clReleaseMemObject(output_buffer_);
        
        // Use READ_WRITE so downstream code can read logits from this buffer
        output_buffer_ = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, nullptr, nullptr);
        if (!output_buffer_) {
            throw std::runtime_error("Failed to create LM head output buffer");
        }
        output_buffer_size_ = output_size;
    }
    
    // TODO: Use OpenCL matmul kernel
    // For now, use CPU fallback
    
    // Read hidden states
    std::vector<float> hidden_cpu(batch_size * d_model_);
    cl_int err = clEnqueueReadBuffer(
        queue, hidden_states, CL_TRUE, 0,
        batch_size * d_model_ * sizeof(float), hidden_cpu.data(),
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read hidden states");
    }
    
    // Debug: Check hidden state statistics (first time only)
    static bool debug_hidden_lm = true;
    if (debug_hidden_lm) {
        float min_h = *std::min_element(hidden_cpu.begin(), hidden_cpu.end());
        float max_h = *std::max_element(hidden_cpu.begin(), hidden_cpu.end());
        float sum_h = std::accumulate(hidden_cpu.begin(), hidden_cpu.end(), 0.0f);
        float mean_h = sum_h / hidden_cpu.size();
        std::cout << "  [LMHead Debug] Hidden state input: min=" << min_h 
                  << ", max=" << max_h << ", mean=" << mean_h << std::endl;
        std::cout << "  [LMHead Debug] First 10 hidden values: ";
        for (int i = 0; i < 10 && i < d_model_; ++i) {
            std::cout << hidden_cpu[i] << " ";
        }
        std::cout << std::endl;
        debug_hidden_lm = false;
    }
    
    // Read weights from chunks
    std::vector<std::vector<float>> chunks_cpu(num_chunks_);
    for (int chunk_idx = 0; chunk_idx < num_chunks_; ++chunk_idx) {
        int start_token = chunk_idx * chunk_size_;
        int end_token = std::min(start_token + chunk_size_, vocab_size_);
        int chunk_vocab_size = end_token - start_token;
        
        chunks_cpu[chunk_idx].resize(chunk_vocab_size * d_model_);
        err = clEnqueueReadBuffer(
            queue, weights_buffers_[chunk_idx], CL_TRUE, 0,
            chunk_vocab_size * d_model_ * sizeof(float), chunks_cpu[chunk_idx].data(),
            0, nullptr, nullptr
        );
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to read LM head chunk " + std::to_string(chunk_idx));
        }
    }
    
    // Compute logits: hidden @ weights^T
    // Formula: logits[v] = sum_d (hidden[d] * weights[v, d])
    // where weights are stored row-major: weights[v, d] = weights[v * d_model + d]
    std::vector<float> logits_cpu(batch_size * vocab_size_);
    
    // Debug: Check weight statistics (first time only)
    static bool debug_weights_lm = true;
    if (debug_weights_lm && !chunks_cpu.empty()) {
        float min_w = chunks_cpu[0][0], max_w = chunks_cpu[0][0], sum_w = 0.0f;
        int count = 0;
        for (const auto& chunk : chunks_cpu) {
            for (float w : chunk) {
                min_w = std::min(min_w, w);
                max_w = std::max(max_w, w);
                sum_w += w;
                count++;
            }
        }
        float mean_w = sum_w / count;
        std::cout << "  [LMHead Debug] Weight stats: min=" << min_w 
                  << ", max=" << max_w << ", mean=" << mean_w << std::endl;
        std::cout << "  [LMHead Debug] First 10 weight values: ";
        for (int i = 0; i < 10 && i < static_cast<int>(chunks_cpu[0].size()); ++i) {
            std::cout << chunks_cpu[0][i] << " ";
        }
        std::cout << std::endl;
        debug_weights_lm = false;
    }
    
    for (int b = 0; b < batch_size; ++b) {
        for (int v = 0; v < vocab_size_; ++v) {
            // Determine which chunk this vocab token belongs to
            int chunk_idx = v / chunk_size_;
            int vocab_offset_in_chunk = v % chunk_size_;
            
            float sum = 0.0f;
            for (int d = 0; d < d_model_; ++d) {
                int hidden_idx = b * d_model_ + d;
                int weight_idx = vocab_offset_in_chunk * d_model_ + d;  // Row-major storage
                sum += hidden_cpu[hidden_idx] * chunks_cpu[chunk_idx][weight_idx];
            }
            logits_cpu[b * vocab_size_ + v] = sum;
        }
    }
    
    // Ensure queue is idle before writing logits back (improves stability on some drivers)
    clFinish(queue);

    // Write back to GPU
    err = clEnqueueWriteBuffer(
        queue, output_buffer_, CL_TRUE, 0,
        output_size, logits_cpu.data(),
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS) {
        std::string err_msg = "Failed to write LM head output (err=" + std::to_string(err) + ")";
        throw std::runtime_error(err_msg);
    }
    
    // Increase ref count before returning so callers can safely clRelease
    // without invalidating LMHead's internal buffer reference.
    clRetainMemObject(output_buffer_);
    return output_buffer_;
}

} // namespace cartesia_opencl

