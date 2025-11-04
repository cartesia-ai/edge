#include "swiglu_layer.h"
#include "../opencl_context.h"
#include "linear_layer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <CL/cl.h>

namespace cartesia_opencl {

SwiGLULayer::SwiGLULayer(OpenCLContextManager* ctx, int d_model, int expand)
    : ctx_(ctx)
    , d_model_(d_model)
    , d_inner_(d_model * expand)
    , weights_initialized_(false)
    , program_(nullptr)
    , swish_kernel_(nullptr)
    , combine_kernel_(nullptr)
    , swish_output_(nullptr)
    , intermediate_(nullptr)
    , swish_output_size_(0)
    , intermediate_size_(0)
    , swish_output_step_(nullptr)
    , intermediate_step_(nullptr)
    , swish_output_step_size_(0)
    , intermediate_step_size_(0)
{
    if (!ctx_) {
        throw std::runtime_error("OpenCLContext is null");
    }
    
    // Debug output
    size_t gate_params = static_cast<size_t>(d_inner_) * d_model;
    size_t up_params = static_cast<size_t>(d_inner_) * d_model;
    size_t down_params = static_cast<size_t>(d_model) * d_inner_;
    size_t total_params = gate_params + up_params + down_params;
    size_t buffer_size_mb = (total_params * sizeof(float)) / (1024 * 1024);
    // std::cout << "  [SwiGLU] d_model=" << d_model 
    //           << ", expand=" << expand
    //           << ", d_inner=" << d_inner_
    //           << ", params=" << total_params
    //           << " (gate:" << gate_params << ", up:" << up_params << ", down:" << down_params << ")"
    //           << ", buffer_size=" << buffer_size_mb << " MB" << std::endl;
    
    // Create linear layers
    gate_layer_ = std::make_unique<LinearLayer>(ctx_, d_model, d_inner_, false);
    up_layer_ = std::make_unique<LinearLayer>(ctx_, d_model, d_inner_, false);
    down_layer_ = std::make_unique<LinearLayer>(ctx_, d_inner_, d_model, false);
    
    buildKernels();
}

SwiGLULayer::~SwiGLULayer() {
    if (swish_kernel_) clReleaseKernel(swish_kernel_);
    if (combine_kernel_) clReleaseKernel(combine_kernel_);
    if (program_) clReleaseProgram(program_);
}

void SwiGLULayer::buildKernels() {
    // Embedded SwiGLU kernel source
    const char* swiglu_cl_source = R"(
// Swish (SiLU) activation function: x * sigmoid(x)
__kernel void swish(
    __global const float* input,
    __global float* output,
    const int size
) {
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float x = input[idx];
    
    // Clamp input to prevent Inf/NaN
    if (isnan(x) || isinf(x)) {
        output[idx] = 0.0f;
        return;
    }
    
    // Clamp to reasonable range to prevent exp overflow
    const float max_val = 50.0f;  // exp(-50) is very close to 0, safe for sigmoid
    const float min_val = -50.0f;  // exp(50) would overflow, but we use exp(-x) so -50 means exp(50)
    if (x > max_val) x = max_val;
    if (x < min_val) x = min_val;
    
    // Compute sigmoid with overflow protection
    // For large negative x, exp(-x) can overflow, so use: 1 / (1 + exp(-x)) ≈ 0
    // For large positive x, exp(-x) ≈ 0, so sigmoid ≈ 1
    float sigmoid_x;
    if (x < -50.0f) {
        sigmoid_x = 0.0f;
    } else if (x > 50.0f) {
        sigmoid_x = 1.0f;
    } else {
        float exp_val = exp(-x);
        if (isinf(exp_val) || isnan(exp_val)) {
            // Overflow protection
            sigmoid_x = (x < 0.0f) ? 0.0f : 1.0f;
        } else {
            sigmoid_x = 1.0f / (1.0f + exp_val);
        }
    }
    
    // Compute swish: x * sigmoid(x)
    output[idx] = x * sigmoid_x;
    
    // Final check
    if (isnan(output[idx]) || isinf(output[idx])) {
        output[idx] = 0.0f;
    }
}

// SwiGLU: (Swish(gate) * up) 
// Note: This is typically called after separate linear projections
__kernel void swiglu_combine(
    __global const float* gate,      // [size] - already Swish activated
    __global const float* up,         // [size]
    __global float* output,           // [size]
    const int size
) {
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float gate_val = gate[idx];
    float up_val = up[idx];
    
    // Check for Inf/NaN and clamp
    if (isnan(gate_val) || isinf(gate_val)) {
        gate_val = 0.0f;
    }
    if (isnan(up_val) || isinf(up_val)) {
        up_val = 0.0f;
    }
    
    // Clamp to prevent overflow
    const float max_val = 1e10f;
    const float min_val = -1e10f;
    if (gate_val > max_val) gate_val = max_val;
    if (gate_val < min_val) gate_val = min_val;
    if (up_val > max_val) up_val = max_val;
    if (up_val < min_val) up_val = min_val;
    
    output[idx] = gate_val * up_val;
    
    // Final check
    if (isnan(output[idx]) || isinf(output[idx])) {
        output[idx] = 0.0f;
    }
}
)";
    
    // Build program
    auto& ctx_mgr = OpenCLContextManager::getInstance();
    std::vector<std::string> sources = {std::string(swiglu_cl_source)};
    std::string cache_key = ctx_mgr.generateCacheKey(sources);
    program_ = ctx_mgr.buildProgram(sources, cache_key);
    
    // Create kernels
    swish_kernel_ = ctx_mgr.getKernel(program_, "swish");
    combine_kernel_ = ctx_mgr.getKernel(program_, "swiglu_combine");
}

void SwiGLULayer::initializeWeights(
    const std::vector<float>& gate_weights,
    const std::vector<float>& up_weights,
    const std::vector<float>& down_weights
) {
    if (gate_weights.size() != static_cast<size_t>(d_inner_ * d_model_) ||
        up_weights.size() != static_cast<size_t>(d_inner_ * d_model_) ||
        down_weights.size() != static_cast<size_t>(d_model_ * d_inner_)) {
        throw std::runtime_error("Invalid SwiGLU weights size");
    }
    
    gate_layer_->initializeWeights(gate_weights);
    up_layer_->initializeWeights(up_weights);
    down_layer_->initializeWeights(down_weights);
    
    weights_initialized_ = true;
}

cl_mem SwiGLULayer::applySwish(cl_mem input, int batch_size, int seq_len, cl_command_queue queue) {
    cl_context context = ctx_->getContext();
    int size = batch_size * seq_len * d_inner_;
    
    // Allocate swish output buffer
    if (!swish_output_ || swish_output_size_ < size * sizeof(float)) {
        if (swish_output_) clReleaseMemObject(swish_output_);
        swish_output_ = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float), nullptr, nullptr);
        swish_output_size_ = size * sizeof(float);
    }
    
    // Set kernel arguments
    cl_int err;
    err = clSetKernelArg(swish_kernel_, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(swish_kernel_, 1, sizeof(cl_mem), &swish_output_);
    err |= clSetKernelArg(swish_kernel_, 2, sizeof(int), &size);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set swish kernel arguments");
    }
    
    // Execute
    size_t global_size = size;
    err = clEnqueueNDRangeKernel(queue, swish_kernel_, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue swish kernel");
    }
    
    return swish_output_;
}

cl_mem SwiGLULayer::applySwishStep(cl_mem input, int batch_size, cl_command_queue queue) {
    int size = batch_size * d_inner_;
    
    // Allocate step swish output buffer
    if (!swish_output_step_ || swish_output_step_size_ < size * sizeof(float)) {
        if (swish_output_step_) clReleaseMemObject(swish_output_step_);
        swish_output_step_ = clCreateBuffer(ctx_->getContext(), CL_MEM_WRITE_ONLY, size * sizeof(float), nullptr, nullptr);
        swish_output_step_size_ = size * sizeof(float);
    }
    
    cl_int err;
    err = clSetKernelArg(swish_kernel_, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(swish_kernel_, 1, sizeof(cl_mem), &swish_output_step_);
    err |= clSetKernelArg(swish_kernel_, 2, sizeof(int), &size);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set swish kernel arguments");
    }
    
    size_t global_size = size;
    err = clEnqueueNDRangeKernel(queue, swish_kernel_, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue swish kernel");
    }
    
    return swish_output_step_;
}

cl_mem SwiGLULayer::forward(
    cl_mem input,
    int batch_size,
    int seq_len,
    LayerState* state,
    cl_command_queue queue
) {
    if (!weights_initialized_) {
        throw std::runtime_error("SwiGLU weights not initialized");
    }
    
    // Gate projection: input -> d_inner
    cl_mem gate_out = gate_layer_->forward(input, batch_size, seq_len, queue);
    
    // Up projection: input -> d_inner
    cl_mem up_out = up_layer_->forward(input, batch_size, seq_len, queue);
    
    // Apply Swish to gate
    cl_mem gate_swish = applySwish(gate_out, batch_size, seq_len, queue);
    
    // Combine: gate_swish * up
    cl_context context = ctx_->getContext();
    int size = batch_size * seq_len * d_inner_;
    
    if (!intermediate_ || intermediate_size_ < size * sizeof(float)) {
        if (intermediate_) clReleaseMemObject(intermediate_);
        intermediate_ = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float), nullptr, nullptr);
        intermediate_size_ = size * sizeof(float);
    }
    
    cl_int err;
    err = clSetKernelArg(combine_kernel_, 0, sizeof(cl_mem), &gate_swish);
    err |= clSetKernelArg(combine_kernel_, 1, sizeof(cl_mem), &up_out);
    err |= clSetKernelArg(combine_kernel_, 2, sizeof(cl_mem), &intermediate_);
    err |= clSetKernelArg(combine_kernel_, 3, sizeof(int), &size);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set swiglu_combine kernel arguments");
    }
    
    size_t global_size = size;
    err = clEnqueueNDRangeKernel(queue, combine_kernel_, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue swiglu_combine kernel");
    }
    
    // Down projection: intermediate -> d_model
    cl_mem output = down_layer_->forward(intermediate_, batch_size, seq_len, queue);
    
    return output;
}

cl_mem SwiGLULayer::step(
    cl_mem input,
    int batch_size,
    LayerState* state,
    cl_command_queue queue
) {
    if (!weights_initialized_) {
        throw std::runtime_error("SwiGLU weights not initialized");
    }
    
    // Gate and up projections
    std::cout << " [gate_proj]..." << std::flush;
    cl_mem gate_out = gate_layer_->step(input, batch_size, queue);
    std::cout << " [up_proj]..." << std::flush;
    cl_mem up_out = up_layer_->step(input, batch_size, queue);
    
    // Apply Swish to gate
    std::cout << " [swish]..." << std::flush;
    cl_mem gate_swish = applySwishStep(gate_out, batch_size, queue);
    
    // Combine
    std::cout << " [combine]..." << std::flush;
    cl_context context = ctx_->getContext();
    int size = batch_size * d_inner_;
    
    if (!intermediate_step_ || intermediate_step_size_ < size * sizeof(float)) {
        if (intermediate_step_) clReleaseMemObject(intermediate_step_);
        intermediate_step_ = clCreateBuffer(context, CL_MEM_WRITE_ONLY, size * sizeof(float), nullptr, nullptr);
        intermediate_step_size_ = size * sizeof(float);
    }
    
    cl_int err;
    err = clSetKernelArg(combine_kernel_, 0, sizeof(cl_mem), &gate_swish);
    err |= clSetKernelArg(combine_kernel_, 1, sizeof(cl_mem), &up_out);
    err |= clSetKernelArg(combine_kernel_, 2, sizeof(cl_mem), &intermediate_step_);
    err |= clSetKernelArg(combine_kernel_, 3, sizeof(int), &size);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set swiglu_combine kernel arguments");
    }
    
    size_t global_size = size;
    err = clEnqueueNDRangeKernel(queue, combine_kernel_, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue swiglu_combine kernel");
    }
    
    // Down projection
    std::cout << " [down_proj]..." << std::flush;
    cl_mem output = down_layer_->step(intermediate_step_, batch_size, queue);
    
    return output;
}

} // namespace cartesia_opencl

