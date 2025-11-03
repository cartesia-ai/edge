#include "embedding.h"
#include "opencl_context.h"
#include <stdexcept>
#include <algorithm>
#include <iostream>
#include <CL/cl.h>

namespace cartesia_opencl {

EmbeddingLayer::EmbeddingLayer(OpenCLContextManager* ctx, int vocab_size, int d_model)
    : ctx_(ctx)
    , vocab_size_(vocab_size)
    , d_model_(d_model)
    , weights_buffer_(nullptr)
    , weights_initialized_(false)
    , output_buffer_(nullptr)
    , output_buffer_size_(0)
{
    if (!ctx_) {
        throw std::runtime_error("OpenCLContext is null");
    }
    
    // Debug output
    size_t num_params = static_cast<size_t>(vocab_size) * d_model;
    size_t buffer_size_mb = (num_params * sizeof(float)) / (1024 * 1024);
    std::cout << "  [Embedding] vocab_size=" << vocab_size 
              << ", d_model=" << d_model 
              << ", params=" << num_params
              << ", buffer_size=" << buffer_size_mb << " MB" << std::endl;
}

EmbeddingLayer::~EmbeddingLayer() {
    if (weights_buffer_) clReleaseMemObject(weights_buffer_);
    if (output_buffer_) clReleaseMemObject(output_buffer_);
}

void EmbeddingLayer::initializeWeights(const std::vector<float>& weights) {
    if (weights.size() != static_cast<size_t>(vocab_size_ * d_model_)) {
        throw std::runtime_error("Invalid embedding weights size");
    }
    
    cl_context context = ctx_->getContext();
    cl_device_id device = ctx_->getDevice();
    cl_int err;
    
    // Check device maximum buffer size
    cl_ulong max_alloc_size = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc_size, nullptr);
    if (err == CL_SUCCESS) {
        size_t buffer_size = weights.size() * sizeof(float);
        if (buffer_size > max_alloc_size) {
            std::string msg = "Embedding buffer size (" + std::to_string(buffer_size) + 
                            " bytes) exceeds device maximum (" + std::to_string(max_alloc_size) + " bytes)";
            throw std::runtime_error(msg);
        }
    }
    
    size_t buffer_size = weights.size() * sizeof(float);
    weights_buffer_ = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        buffer_size,
        (void*)weights.data(),
        &err
    );
    
    if (err != CL_SUCCESS || !weights_buffer_) {
        std::string err_msg = "Failed to create embedding weights buffer: " + std::to_string(err);
        if (err == CL_INVALID_BUFFER_SIZE) {
            err_msg += " (CL_INVALID_BUFFER_SIZE - buffer too large)";
            err_msg += "\n  Buffer size: " + std::to_string(buffer_size) + " bytes";
            err_msg += "\n  Vocab size: " + std::to_string(vocab_size_);
            err_msg += "\n  d_model: " + std::to_string(d_model_);
        }
        throw std::runtime_error(err_msg);
    }
    
    weights_initialized_ = true;
}

cl_mem EmbeddingLayer::encode(cl_mem token_ids, int batch_size, int seq_len, cl_command_queue queue) {
    if (!weights_initialized_) {
        throw std::runtime_error("Embedding weights not initialized");
    }
    
    cl_context context = ctx_->getContext();
    size_t output_size = batch_size * seq_len * d_model_ * sizeof(float);
    
    // Allocate or resize output buffer
    if (!output_buffer_ || output_buffer_size_ != output_size) {
        if (output_buffer_) clReleaseMemObject(output_buffer_);
        
        // Use READ_WRITE for broader driver compatibility
        cl_int create_err = CL_SUCCESS;
        output_buffer_ = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, nullptr, &create_err);
        if (create_err != CL_SUCCESS || !output_buffer_) {
            std::cerr << "\n[Embedding] Failed to create output buffer (encode) err=" << create_err
                      << ", size=" << output_size << std::endl;
            throw std::runtime_error("Failed to create embedding output buffer");
        }
        output_buffer_size_ = output_size;
    }
    
    // For now, use CPU fallback for embedding lookup
    // TODO: Implement OpenCL kernel for efficient embedding lookup
    
    // Read token IDs
    std::vector<int> token_ids_cpu(batch_size * seq_len);
    cl_int err = clEnqueueReadBuffer(
        queue, token_ids, CL_TRUE, 0,
        batch_size * seq_len * sizeof(int), token_ids_cpu.data(),
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read token IDs");
    }
    
    // Read weights
    std::vector<float> weights_cpu(vocab_size_ * d_model_);
    err = clEnqueueReadBuffer(
        queue, weights_buffer_, CL_TRUE, 0,
        vocab_size_ * d_model_ * sizeof(float), weights_cpu.data(),
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read embedding weights");
    }
    
    // Perform embedding lookup on CPU
    std::vector<float> output_cpu(batch_size * seq_len * d_model_);
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int token_id = token_ids_cpu[b * seq_len + s];
            if (token_id < 0 || token_id >= vocab_size_) {
                token_id = 0;  // Fallback to first token
            }
            
            for (int d = 0; d < d_model_; ++d) {
                int out_idx = (b * seq_len + s) * d_model_ + d;
                int weight_idx = token_id * d_model_ + d;
                output_cpu[out_idx] = weights_cpu[weight_idx];
            }
        }
    }
    
    // Ensure prior commands are finished (some drivers require explicit sync)
    clFinish(queue);

    // Prefer map/unmap over WriteBuffer to avoid CL_INVALID_MEM_OBJECT quirks
    void* mapped_ptr = clEnqueueMapBuffer(
        queue, output_buffer_, CL_TRUE, CL_MAP_WRITE, 0, output_size,
        0, nullptr, nullptr, &err
    );
    if (err != CL_SUCCESS || mapped_ptr == nullptr) {
        // Provide detailed diagnostics
        cl_device_id device = ctx_->getDevice();
        cl_ulong max_alloc = 0, global_mem = 0;
        clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem, nullptr);
        std::cerr << "\n[Embedding] Error mapping output buffer" << std::endl;
        std::cerr << "  OpenCL err: " << err << std::endl;
        std::cerr << "  output_size: " << output_size << " bytes" << std::endl;
        std::cerr << "  output_buffer_size_: " << output_buffer_size_ << " bytes" << std::endl;
        std::cerr << "  shape: (batch=" << batch_size << ", seq_len=" << seq_len << ", d_model=" << d_model_ << ")" << std::endl;
        std::cerr << "  device max_alloc: " << (max_alloc / 1024 / 1024) << " MB, global: " << (global_mem / 1024 / 1024) << " MB" << std::endl;
        throw std::runtime_error("Failed to map embedding output buffer");
    }
    std::memcpy(mapped_ptr, output_cpu.data(), output_size);
    err = clEnqueueUnmapMemObject(queue, output_buffer_, mapped_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "\n[Embedding] Error unmapping output buffer, err=" << err << std::endl;
        throw std::runtime_error("Failed to unmap embedding output buffer");
    }
    clFinish(queue);
    
    return output_buffer_;
}

cl_mem EmbeddingLayer::encodeStep(cl_mem token_ids, int batch_size, cl_command_queue queue) {
    if (!weights_initialized_) {
        throw std::runtime_error("Embedding weights not initialized");
    }
    
    cl_context context = ctx_->getContext();
    size_t output_size = batch_size * d_model_ * sizeof(float);
    
    // Allocate or resize output buffer
    // For step, always recreate to avoid stale handles on some drivers
    if (output_buffer_) { clReleaseMemObject(output_buffer_); output_buffer_ = nullptr; }
    if (!output_buffer_ || output_buffer_size_ != output_size) {
        if (output_buffer_) clReleaseMemObject(output_buffer_);
        
        // Use READ_WRITE for broader driver compatibility
        cl_int create_err = CL_SUCCESS;
        output_buffer_ = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, nullptr, &create_err);
        if (create_err != CL_SUCCESS || !output_buffer_) {
            std::cerr << "\n[Embedding] Failed to create output buffer (step) err=" << create_err
                      << ", size=" << output_size << std::endl;
            throw std::runtime_error("Failed to create embedding output buffer");
        }
        output_buffer_size_ = output_size;
    }
    
    // CPU fallback for single token (similar to encode)
    std::vector<int> token_ids_cpu(batch_size);
    cl_int err = clEnqueueReadBuffer(
        queue, token_ids, CL_TRUE, 0,
        batch_size * sizeof(int), token_ids_cpu.data(),
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read token IDs");
    }
    
    std::vector<float> weights_cpu(vocab_size_ * d_model_);
    err = clEnqueueReadBuffer(
        queue, weights_buffer_, CL_TRUE, 0,
        vocab_size_ * d_model_ * sizeof(float), weights_cpu.data(),
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read embedding weights");
    }
    
    std::vector<float> output_cpu(batch_size * d_model_);
    for (int b = 0; b < batch_size; ++b) {
        int token_id = token_ids_cpu[b];
        if (token_id < 0 || token_id >= vocab_size_) {
            token_id = 0;
        }
        
        for (int d = 0; d < d_model_; ++d) {
            int out_idx = b * d_model_ + d;
            int weight_idx = token_id * d_model_ + d;
            output_cpu[out_idx] = weights_cpu[weight_idx];
        }
    }
    
    // Ensure prior commands are finished
    clFinish(queue);

    // Map/unmap path
    void* mapped_ptr = clEnqueueMapBuffer(
        queue, output_buffer_, CL_TRUE, CL_MAP_WRITE, 0, output_size,
        0, nullptr, nullptr, &err
    );
    if (err != CL_SUCCESS || mapped_ptr == nullptr) {
        cl_device_id device = ctx_->getDevice();
        cl_ulong max_alloc = 0, global_mem = 0;
        clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(cl_ulong), &max_alloc, nullptr);
        clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem, nullptr);
        std::cerr << "\n[Embedding] Error mapping output buffer (step)" << std::endl;
        std::cerr << "  OpenCL err: " << err << std::endl;
        std::cerr << "  output_size: " << output_size << " bytes" << std::endl;
        std::cerr << "  output_buffer_size_: " << output_buffer_size_ << " bytes" << std::endl;
        std::cerr << "  shape: (batch=" << batch_size << ", d_model=" << d_model_ << ")" << std::endl;
        std::cerr << "  device max_alloc: " << (max_alloc / 1024 / 1024) << " MB, global: " << (global_mem / 1024 / 1024) << " MB" << std::endl;
        throw std::runtime_error("Failed to map embedding output buffer");
    }
    std::memcpy(mapped_ptr, output_cpu.data(), output_size);
    err = clEnqueueUnmapMemObject(queue, output_buffer_, mapped_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "\n[Embedding] Error unmapping output buffer (step), err=" << err << std::endl;
        throw std::runtime_error("Failed to unmap embedding output buffer");
    }
    clFinish(queue);
    
    return output_buffer_;
}

} // namespace cartesia_opencl

