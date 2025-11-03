#include "lm_head.h"
#include "opencl_context.h"
#include <stdexcept>
#include <vector>
#include <iostream>
#include <CL/cl.h>

namespace cartesia_opencl {

LMHead::LMHead(OpenCLContextManager* ctx, int d_model, int vocab_size)
    : ctx_(ctx)
    , d_model_(d_model)
    , vocab_size_(vocab_size)
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
    std::cout << "  [LMHead] d_model=" << d_model 
              << ", vocab_size=" << vocab_size 
              << ", params=" << num_params
              << ", buffer_size=" << buffer_size_mb << " MB" << std::endl;
}

LMHead::~LMHead() {
    if (weights_buffer_) clReleaseMemObject(weights_buffer_);
    if (output_buffer_) clReleaseMemObject(output_buffer_);
}

void LMHead::initializeWeights(const std::vector<float>& weights) {
    // Weights are [vocab_size, d_model] but stored row-major
    // For matmul, we need them transposed or use appropriate kernel
    if (weights.size() != static_cast<size_t>(vocab_size_ * d_model_)) {
        throw std::runtime_error("Invalid LM head weights size");
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
            std::string msg = "LM head buffer size (" + std::to_string(buffer_size) + 
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
        std::string err_msg = "Failed to create LM head weights buffer: " + std::to_string(err);
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
    
    // Read weights
    std::vector<float> weights_cpu(vocab_size_ * d_model_);
    err = clEnqueueReadBuffer(
        queue, weights_buffer_, CL_TRUE, 0,
        vocab_size_ * d_model_ * sizeof(float), weights_cpu.data(),
        0, nullptr, nullptr
    );
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read LM head weights");
    }
    
    // Compute logits: hidden @ weights^T
    std::vector<float> logits_cpu(batch_size * vocab_size_);
    for (int b = 0; b < batch_size; ++b) {
        for (int v = 0; v < vocab_size_; ++v) {
            float sum = 0.0f;
            for (int d = 0; d < d_model_; ++d) {
                int hidden_idx = b * d_model_ + d;
                int weight_idx = v * d_model_ + d;  // Row-major storage
                sum += hidden_cpu[hidden_idx] * weights_cpu[weight_idx];
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

