#include "residual_block.h"
#include "opencl_context.h"
#include "layers/rms_norm_layer.h"
#include <stdexcept>
#include <CL/cl.h>
#include <cstring>
#include <vector>
#include <iostream>
#include <cmath>

namespace cartesia_opencl {

ResidualBlock::ResidualBlock(
    OpenCLContextManager* ctx,
    Layer* layer,
    int d_model,
    const std::string& norm_point,
    bool stateful
)
    : ctx_(ctx)
    , layer_(layer)
    , d_model_(d_model)
    , norm_point_(norm_point)
    , stateful_(stateful)
{
    if (!ctx_ || !layer_) {
        throw std::runtime_error("Invalid ResidualBlock parameters");
    }
    
    // Initialize norm layer if needed
    if (!norm_point_.empty()) {
        norm_layer_ = std::make_unique<RMSNormLayer>(ctx, d_model);
        // Initialize with ones (will be set from weights later if needed)
        norm_weights_.resize(d_model, 1.0f);
        norm_layer_->initializeWeights(norm_weights_);
    }
}

ResidualBlock::~ResidualBlock() {
    // norm_layer_ will clean itself up
    // Note: We don't delete layer_ - it's managed externally
}

cl_mem ResidualBlock::forward(
    cl_mem input,
    int batch_size,
    int seq_len,
    LayerState* state,
    cl_command_queue queue
) {
    cl_context context = ctx_->getContext();
    cl_mem residual = input;  // Save for residual connection
    
    // Pre-norm
    if (!norm_point_.empty() && norm_point_ == "pre") {
        input = applyNorm(input, batch_size, seq_len, queue);
        
        // Debug: Check pre-norm output (for layer 6 attention only)
        static bool checked_layer6_prenorm = false;
        if (!checked_layer6_prenorm && layer_ && layer_->isStateful()) {
            // This is likely the attention layer (only stateful layer)
            clFinish(queue);  // Ensure norm completes
            
            // Retain the buffer to prevent it from being released
            clRetainMemObject(input);
            
            size_t input_size = batch_size * seq_len * d_model_;
            std::vector<float> prenorm_check(input_size);
            size_t buf_size = 0;
            cl_int info_err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
            
            if (info_err == CL_SUCCESS && buf_size >= input_size * sizeof(float)) {
                cl_int check_err = clEnqueueReadBuffer(queue, input, CL_TRUE, 0,
                    input_size * sizeof(float), prenorm_check.data(), 0, nullptr, nullptr);
                if (check_err == CL_SUCCESS) {
                    int nan_count = 0;
                    int inf_count = 0;
                    std::vector<int> nan_per_token(seq_len, 0);
                    for (size_t i = 0; i < prenorm_check.size(); ++i) {
                        if (std::isnan(prenorm_check[i])) {
                            nan_count++;
                            int token_idx = i / d_model_;
                            if (token_idx < seq_len) {
                                nan_per_token[token_idx]++;
                            }
                        }
                        if (std::isinf(prenorm_check[i])) {
                            inf_count++;
                        }
                    }
                    std::cout << "  [ResidualBlock PreNorm Debug] After pre-norm: " << nan_count 
                              << " NaNs, " << inf_count << " Infs out of " << input_size << " values, buffer_size=" << buf_size << std::endl;
                    std::cout << "  [ResidualBlock PreNorm Debug] NaNs per token: ";
                    for (int i = 0; i < seq_len; ++i) {
                        std::cout << "token" << i << "=" << nan_per_token[i] << "/" << d_model_ << " ";
                    }
                    std::cout << std::endl;
                }
            }
            checked_layer6_prenorm = true;
        }
    }
    
    // Apply layer
    // Ensure all previous operations complete before passing input to layer
    clFinish(queue);
    
    // Debug: Check input right before calling layer forward (for attention layer)
    static bool checked_before_forward = false;
    static std::vector<float> saved_buffer_data;
    static void* saved_buffer_ptr = nullptr;
    if (!checked_before_forward && layer_ && layer_->isStateful()) {
        size_t input_size = batch_size * seq_len * d_model_;
        std::vector<float> before_forward_check(input_size);
        size_t buf_size = 0;
        clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
        cl_int check_err = clEnqueueReadBuffer(queue, input, CL_TRUE, 0,
            input_size * sizeof(float), before_forward_check.data(), 0, nullptr, nullptr);
        if (check_err == CL_SUCCESS) {
            int nan_count = 0;
            for (float val : before_forward_check) {
                if (std::isnan(val)) { nan_count++; }
            }
            std::cout << "  [ResidualBlock BeforeForward] Input right before layer->forward(): " 
                      << nan_count << " NaNs out of " << input_size << " values" << std::endl;
            
            // Save buffer data and pointer for comparison
            saved_buffer_data = before_forward_check;
            saved_buffer_ptr = input;
            
            // Print token 5 values to compare later
            std::cout << "  [ResidualBlock BeforeForward] Saved token 5 first 5 values: ";
            for (int i = 5 * d_model_; i < 5 * d_model_ + 5; ++i) {
                std::cout << before_forward_check[i] << " ";
            }
            std::cout << std::endl;
        }
        checked_before_forward = true;
    }
    
    // If we saved buffer data, check if the pointer is still the same when we call forward
    if (checked_before_forward && layer_ && layer_->isStateful() && input == saved_buffer_ptr) {
        std::cout << "  [ResidualBlock] About to call layer->forward() with SAME buffer pointer" << std::endl;
    } else if (checked_before_forward && layer_ && layer_->isStateful() && input != saved_buffer_ptr) {
        std::cout << "  [ResidualBlock] WARNING: Buffer pointer changed! Was=" << saved_buffer_ptr 
                  << ", Now=" << input << std::endl;
    }
    
    cl_mem output;
    if (stateful_) {
        output = layer_->forward(input, batch_size, seq_len, state, queue);
    } else {
        LayerState dummy_state = LayerState::null();
        output = layer_->forward(input, batch_size, seq_len, &dummy_state, queue);
    }
    
    // Pre-resid norm
    if (!norm_point_.empty() && norm_point_ == "pre_resid") {
        output = applyNorm(output, batch_size, seq_len, queue);
    }
    
    // Residual connection: output = output + residual
    // IMPORTANT: Check if output and residual are the same buffer - if so, we need to create a new buffer
    // This can happen if a layer returns its input buffer without modification
    cl_int err;
    cl_mem residual_buffer = residual;
    cl_mem output_buffer = output;
    
    // Check if output == residual (same buffer pointer)
    if (output_buffer == residual_buffer) {
        // Layer returned input unchanged - create new buffer for output
        cl_context context = ctx_->getContext();
        size_t output_size = batch_size * seq_len * d_model_ * sizeof(float);
        output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, nullptr, &err);
        if (err != CL_SUCCESS || !output_buffer) {
            throw std::runtime_error("Failed to create new output buffer for residual connection");
        }
        // Copy input to output (since layer didn't modify it)
        err = clEnqueueCopyBuffer(queue, residual_buffer, output_buffer, 0, 0, output_size, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(output_buffer);
            throw std::runtime_error("Failed to copy buffer for residual connection");
        }
        clFinish(queue);
    }
    
    // Allocate temporary buffer for residual addition
    size_t output_size = batch_size * seq_len * d_model_ * sizeof(float);
    
    // Read output and residual, add them, write back
    // For now, we'll use a simple CPU fallback (TODO: implement OpenCL kernel)
    std::vector<float> output_cpu(batch_size * seq_len * d_model_);
    std::vector<float> residual_cpu(batch_size * seq_len * d_model_);
    
    clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0, output_size, output_cpu.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, residual_buffer, CL_TRUE, 0, output_size, residual_cpu.data(), 0, nullptr, nullptr);
    
    for (size_t i = 0; i < output_cpu.size(); ++i) {
        output_cpu[i] += residual_cpu[i];
    }
    
    clEnqueueWriteBuffer(queue, output_buffer, CL_TRUE, 0, output_size, output_cpu.data(), 0, nullptr, nullptr);
    
    // Update output to point to the correct buffer
    output = output_buffer;
    
    // Post-norm
    if (!norm_point_.empty() && norm_point_ == "post") {
        output = applyNorm(output, batch_size, seq_len, queue);
    }
    
    return output;
}

cl_mem ResidualBlock::step(
    cl_mem input,
    int batch_size,
    LayerState* state,
    cl_command_queue queue
) {
    cl_mem residual = input;  // Save for residual connection
    
    // Pre-norm
    if (!norm_point_.empty() && norm_point_ == "pre") {
        input = applyNormStep(input, batch_size, queue);
    }
    
    // Apply layer
    cl_mem output;
    if (stateful_) {
        output = layer_->step(input, batch_size, state, queue);
    } else {
        LayerState dummy_state = LayerState::null();
        output = layer_->step(input, batch_size, &dummy_state, queue);
    }
    
    // Pre-resid norm
    if (!norm_point_.empty() && norm_point_ == "pre_resid") {
        output = applyNormStep(output, batch_size, queue);
    }
    
    // Residual connection
    size_t output_size = batch_size * d_model_ * sizeof(float);
    
    // CPU fallback for residual addition (TODO: OpenCL kernel)
    std::vector<float> output_cpu(batch_size * d_model_);
    std::vector<float> residual_cpu(batch_size * d_model_);
    
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, output_size, output_cpu.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, residual, CL_TRUE, 0, output_size, residual_cpu.data(), 0, nullptr, nullptr);
    
    for (size_t i = 0; i < output_cpu.size(); ++i) {
        output_cpu[i] += residual_cpu[i];
    }
    
    clEnqueueWriteBuffer(queue, output, CL_TRUE, 0, output_size, output_cpu.data(), 0, nullptr, nullptr);
    
    // Post-norm
    if (!norm_point_.empty() && norm_point_ == "post") {
        output = applyNormStep(output, batch_size, queue);
    }
    
    return output;
}

cl_mem ResidualBlock::applyNorm(cl_mem input, int batch_size, int seq_len, cl_command_queue queue) {
    if (!norm_layer_) {
        return input;  // No normalization
    }
    return norm_layer_->forward(input, batch_size, seq_len, queue);
}

cl_mem ResidualBlock::applyNormStep(cl_mem input, int batch_size, cl_command_queue queue) {
    if (!norm_layer_) {
        return input;  // No normalization
    }
    return norm_layer_->step(input, batch_size, queue);
}

void ResidualBlock::setNormWeights(const std::vector<float>& weights) {
    if (!norm_layer_) {
        return;  // No norm layer, nothing to set
    }
    if (weights.size() != static_cast<size_t>(d_model_)) {
        throw std::runtime_error("Norm weights size mismatch: expected " + 
                                std::to_string(d_model_) + ", got " + 
                                std::to_string(weights.size()));
    }
    norm_weights_ = weights;
    norm_layer_->initializeWeights(norm_weights_);
}

} // namespace cartesia_opencl

