#include "embedding.h"
#include "opencl_context.h"
#include "opencl_utils.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <CL/cl.h>

namespace cartesia_opencl {

// OpenCL kernel for embedding lookup
static const char* embedding_lookup_kernel_source = R"(
__kernel void embedding_lookup(
    __global const int* token_ids,      // [batch_size * seq_len] or [batch_size]
    __global const float* weights,      // [chunk_vocab_size, d_model] (one chunk)
    __global float* output,             // [batch_size * seq_len * d_model] or [batch_size * d_model]
    const int d_model,
    const int num_tokens,               // batch_size * seq_len or batch_size
    const int chunk_start_token,        // Starting token ID for this chunk
    const int chunk_vocab_size          // Number of tokens in this chunk
) {
    const int idx = get_global_id(0);   // Index in output: 0 to num_tokens-1
    
    if (idx >= num_tokens) return;
    
    const int token_id = token_ids[idx];
    
    // Check if token is in this chunk's range
    if (token_id < chunk_start_token || token_id >= chunk_start_token + chunk_vocab_size) {
        // Token not in this chunk, skip (will be handled by another chunk's kernel)
        return;
    }
    
    // Calculate offset in this chunk
    const int token_offset = token_id - chunk_start_token;
    const int weight_base = token_offset * d_model;
    const int output_base = idx * d_model;
    
    // Copy embedding vector
    for (int i = 0; i < d_model; ++i) {
        output[output_base + i] = weights[weight_base + i];
    }
}
)";


EmbeddingLayer::EmbeddingLayer(OpenCLContextManager* ctx, int vocab_size, int d_model)
    : ctx_(ctx)
    , vocab_size_(vocab_size)
    , d_model_(d_model)
    , chunk_size_(0)
    , num_chunks_(0)
    , weights_initialized_(false)
    , program_(nullptr)
    , embedding_kernel_(nullptr)
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
    
    // Calculate how many tokens fit in one chunk (leave 10% safety margin)
    size_t safe_alloc_size = static_cast<size_t>(max_alloc_size * 0.9);
    chunk_size_ = safe_alloc_size / (d_model * sizeof(float));
    num_chunks_ = (vocab_size + chunk_size_ - 1) / chunk_size_;  // Ceiling division
    
    size_t total_params = static_cast<size_t>(vocab_size) * d_model;
    size_t total_size_mb = (total_params * sizeof(float)) / (1024 * 1024);
    size_t chunk_size_mb = (chunk_size_ * d_model * sizeof(float)) / (1024 * 1024);
    
    std::cout << "  [Embedding] vocab_size=" << vocab_size 
              << ", d_model=" << d_model 
              << ", params=" << total_params << std::endl;
    std::cout << "  [Embedding] Chunking: " << num_chunks_ << " chunks × " 
              << chunk_size_ << " tokens = " << chunk_size_mb << " MB/chunk, "
              << total_size_mb << " MB total" << std::endl;
    
    // Build GPU kernel
    buildKernels();
}

EmbeddingLayer::~EmbeddingLayer() {
    // Release kernels before program
    if (embedding_kernel_) clReleaseKernel(embedding_kernel_);
    if (program_) clReleaseProgram(program_);
    
    // Release buffers
    for (cl_mem buffer : weights_buffers_) {
        if (buffer) clReleaseMemObject(buffer);
    }
    if (output_buffer_) clReleaseMemObject(output_buffer_);
}

void EmbeddingLayer::initializeWeights(const std::vector<float>& weights) {
    if (weights.size() != static_cast<size_t>(vocab_size_ * d_model_)) {
        throw std::runtime_error("Invalid embedding weights size");
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
            
            std::string err_msg = "Failed to create embedding chunk " + std::to_string(chunk_idx) + 
                                ": error " + std::to_string(err);
            if (err == CL_INVALID_BUFFER_SIZE) {
                err_msg += " (CL_INVALID_BUFFER_SIZE - buffer too large)";
                err_msg += "\n  Chunk buffer size: " + std::to_string(chunk_buffer_size) + " bytes";
                err_msg += "\n  Chunk vocab size: " + std::to_string(chunk_vocab_size);
            }
            throw std::runtime_error(err_msg);
        }
        
        weights_buffers_.push_back(chunk_buffer);
    }
    
    weights_initialized_ = true;
}

void EmbeddingLayer::buildKernels() {
    auto& ctx_mgr = OpenCLContextManager::getInstance();
    std::vector<std::string> sources = {std::string(embedding_lookup_kernel_source)};
    std::string cache_key = ctx_mgr.generateCacheKey(sources) + "_embedding";
    program_ = ctx_mgr.buildProgram(sources, cache_key);
    embedding_kernel_ = ctx_mgr.getKernel(program_, "embedding_lookup");
    std::cout << "  [Embedding] GPU kernel built successfully" << std::endl;
}

cl_mem EmbeddingLayer::encode(cl_mem token_ids, int batch_size, int seq_len, cl_command_queue queue) {
    if (!weights_initialized_) {
        throw std::runtime_error("Embedding weights not initialized");
    }
    
    cl_context context = ctx_->getContext();
    int num_tokens = batch_size * seq_len;
    size_t output_size = num_tokens * d_model_ * sizeof(float);
    
    // Create output buffer (zero-initialized for safety)
    if (output_buffer_) clReleaseMemObject(output_buffer_);
    cl_int buf_err = CL_SUCCESS;
    output_buffer_ = createAndZeroBuffer(context, queue, output_size, &buf_err);
    if (buf_err != CL_SUCCESS || !output_buffer_) {
        throw std::runtime_error("Failed to create embedding output buffer");
    }
    output_buffer_size_ = output_size;
    
    // Run kernel for each chunk
    for (int chunk_idx = 0; chunk_idx < num_chunks_; ++chunk_idx) {
        int start_token = chunk_idx * chunk_size_;
        int end_token = std::min(start_token + chunk_size_, vocab_size_);
        int chunk_vocab_size = end_token - start_token;
        
        // Set kernel arguments
        cl_int err;
        err = clSetKernelArg(embedding_kernel_, 0, sizeof(cl_mem), &token_ids);
        err |= clSetKernelArg(embedding_kernel_, 1, sizeof(cl_mem), &weights_buffers_[chunk_idx]);
        err |= clSetKernelArg(embedding_kernel_, 2, sizeof(cl_mem), &output_buffer_);
        err |= clSetKernelArg(embedding_kernel_, 3, sizeof(int), &d_model_);
        err |= clSetKernelArg(embedding_kernel_, 4, sizeof(int), &num_tokens);
        err |= clSetKernelArg(embedding_kernel_, 5, sizeof(int), &start_token);
        err |= clSetKernelArg(embedding_kernel_, 6, sizeof(int), &chunk_vocab_size);
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to set embedding kernel args for chunk " + std::to_string(chunk_idx));
        }
        
        // Execute kernel
        size_t global_size = num_tokens;
        err = clEnqueueNDRangeKernel(queue, embedding_kernel_, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue embedding kernel for chunk " + std::to_string(chunk_idx));
        }
    }
    
    // Ensure all kernels complete
    clFinish(queue);
    
    return output_buffer_;
}

cl_mem EmbeddingLayer::encodeStep(cl_mem token_ids, int batch_size, cl_command_queue queue) {
    if (!weights_initialized_) {
        throw std::runtime_error("Embedding weights not initialized");
    }
    
    cl_context context = ctx_->getContext();
    int num_tokens = batch_size;  // Single token per batch
    size_t output_size = num_tokens * d_model_ * sizeof(float);
    
    // Create output buffer (zero-initialized for safety)
    if (output_buffer_) clReleaseMemObject(output_buffer_);
    cl_int buf_err = CL_SUCCESS;
    output_buffer_ = createAndZeroBuffer(context, queue, output_size, &buf_err);
    if (buf_err != CL_SUCCESS || !output_buffer_) {
        throw std::runtime_error("Failed to create embedding output buffer (step)");
    }
    output_buffer_size_ = output_size;
    
    // Run kernel for each chunk (same as encode, just with num_tokens = batch_size)
    for (int chunk_idx = 0; chunk_idx < num_chunks_; ++chunk_idx) {
        int start_token = chunk_idx * chunk_size_;
        int end_token = std::min(start_token + chunk_size_, vocab_size_);
        int chunk_vocab_size = end_token - start_token;
        
        // Set kernel arguments
        cl_int err;
        err = clSetKernelArg(embedding_kernel_, 0, sizeof(cl_mem), &token_ids);
        err |= clSetKernelArg(embedding_kernel_, 1, sizeof(cl_mem), &weights_buffers_[chunk_idx]);
        err |= clSetKernelArg(embedding_kernel_, 2, sizeof(cl_mem), &output_buffer_);
        err |= clSetKernelArg(embedding_kernel_, 3, sizeof(int), &d_model_);
        err |= clSetKernelArg(embedding_kernel_, 4, sizeof(int), &num_tokens);
        err |= clSetKernelArg(embedding_kernel_, 5, sizeof(int), &start_token);
        err |= clSetKernelArg(embedding_kernel_, 6, sizeof(int), &chunk_vocab_size);
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to set embedding kernel args for chunk " + std::to_string(chunk_idx) + " (step)");
        }
        
        // Execute kernel
        size_t global_size = num_tokens;
        err = clEnqueueNDRangeKernel(queue, embedding_kernel_, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
        
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue embedding kernel for chunk " + std::to_string(chunk_idx) + " (step)");
        }
    }
    
    // Ensure all kernels complete
    clFinish(queue);
    
    return output_buffer_;
}

} // namespace cartesia_opencl


