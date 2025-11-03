#include "rms_norm_layer.h"
#include "../opencl_context.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <CL/cl.h>

namespace cartesia_opencl {

RMSNormLayer::RMSNormLayer(OpenCLContextManager* ctx, int d_model)
    : ctx_(ctx)
    , d_model_(d_model)
    , program_(nullptr)
    , kernel_(nullptr)
    , weights_buffer_(nullptr)
    , weights_initialized_(false)
    , output_buffer_(nullptr)
    , output_buffer_size_(0)
{
    if (!ctx_) {
        throw std::runtime_error("OpenCLContext is null");
    }
    
    // Debug output
    size_t buffer_size_kb = (d_model * sizeof(float)) / 1024;
    std::cout << "  [RMSNorm] d_model=" << d_model 
              << ", params=" << d_model
              << ", buffer_size=" << buffer_size_kb << " KB" << std::endl;
    
    buildKernels();
}

RMSNormLayer::~RMSNormLayer() {
    if (kernel_) clReleaseKernel(kernel_);
    if (program_) clReleaseProgram(program_);
    if (weights_buffer_) clReleaseMemObject(weights_buffer_);
    if (output_buffer_) clReleaseMemObject(output_buffer_);
}

void RMSNormLayer::buildKernels() {
    // Embedded RMS norm kernel source
    const char* rms_norm_cl_source = R"(
// RMS (Root Mean Square) Normalization kernel
// Normalizes input along the last dimension

__kernel void rms_norm(
    __global const float* input,      // [batch_size, seq_len, d_model] or [batch_size, d_model]
    __global const float* weight,     // [d_model] - scale weights
    __global float* output,           // Same shape as input
    const int d_model,               // Hidden dimension
    const int total_elements,        // Total elements in input (batch_size * seq_len * d_model or batch_size * d_model)
    const float eps                  // Epsilon for numerical stability
) {
    const int idx = get_global_id(0);  // Index of element
    
    if (idx >= total_elements) return;
    
    // Calculate which sequence element this is
    const int seq_idx = idx / d_model;
    const int feat_idx = idx % d_model;
    
    // Calculate mean square within this sequence element
    float mean_square = 0.0f;
    for (int i = 0; i < d_model; ++i) {
        float val = input[seq_idx * d_model + i];
        mean_square += val * val;
    }
    mean_square /= d_model;
    
    // RMS = sqrt(mean_square + eps)
    float rms = sqrt(mean_square + eps);
    
    // Normalize: output = (input / rms) * weight
    output[idx] = (input[idx] / rms) * weight[feat_idx];
}
)";
    
    // Build program
    auto& ctx_mgr = OpenCLContextManager::getInstance();
    std::vector<std::string> sources = {std::string(rms_norm_cl_source)};
    std::string cache_key = ctx_mgr.generateCacheKey(sources);
    program_ = ctx_mgr.buildProgram(sources, cache_key);
    
    // Create kernel
    kernel_ = ctx_mgr.getKernel(program_, "rms_norm");
}

void RMSNormLayer::initializeWeights(const std::vector<float>& weights) {
    if (weights.size() != static_cast<size_t>(d_model_)) {
        throw std::runtime_error("Invalid RMS norm weights size");
    }
    
    cl_context context = ctx_->getContext();
    cl_int err;
    
    weights_buffer_ = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        weights.size() * sizeof(float),
        (void*)weights.data(),
        &err
    );
    
    if (err != CL_SUCCESS || !weights_buffer_) {
        throw std::runtime_error("Failed to create RMS norm weights buffer");
    }
    
    weights_initialized_ = true;
}

cl_mem RMSNormLayer::forward(cl_mem input, int batch_size, int seq_len, cl_command_queue queue) {
    if (!weights_initialized_) {
        throw std::runtime_error("RMS norm weights not initialized");
    }
    
    cl_context context = ctx_->getContext();
    int total_elements = batch_size * seq_len * d_model_;
    size_t output_size = total_elements * sizeof(float);
    
    // Allocate output buffer
    if (!output_buffer_ || output_buffer_size_ < output_size) {
        if (output_buffer_) clReleaseMemObject(output_buffer_);
        output_buffer_ = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, nullptr, nullptr);
        if (!output_buffer_) {
            throw std::runtime_error("Failed to create RMS norm output buffer");
        }
        output_buffer_size_ = output_size;
    }
    
    // Set kernel arguments
    cl_int err;
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &weights_buffer_);
    err |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &output_buffer_);
    err |= clSetKernelArg(kernel_, 3, sizeof(int), &d_model_);
    err |= clSetKernelArg(kernel_, 4, sizeof(int), &total_elements);
    const float eps = 1e-6f;
    err |= clSetKernelArg(kernel_, 5, sizeof(float), &eps);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set RMS norm kernel arguments");
    }
    
    // Execute kernel
    size_t global_size = total_elements;
    err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue RMS norm kernel");
    }
    
    return output_buffer_;
}

cl_mem RMSNormLayer::step(cl_mem input, int batch_size, cl_command_queue queue) {
    // For step, we have [batch_size, d_model] instead of [batch_size, seq_len, d_model]
    // seq_len = 1 in this case
    return forward(input, batch_size, 1, queue);
}

} // namespace cartesia_opencl

