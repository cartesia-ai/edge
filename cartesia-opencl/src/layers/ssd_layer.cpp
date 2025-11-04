#include "ssd_layer.h"
#include "../opencl_context.h"
#include "linear_layer.h"
#include "rms_norm_layer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <CL/cl.h>
#include <cmath>

namespace cartesia_opencl {

SSDLayer::SSDLayer(
    OpenCLContextManager* ctx,
    int d_model,
    int expand,
    int kernel_size,
    int d_state,
    int d_head,
    int n_groups
)
    : ctx_(ctx)
    , d_model_(d_model)
    , d_inner_(d_model * expand)
    , kernel_size_(kernel_size)
    , d_state_(d_state)
    , d_head_(d_head)
    , n_groups_(n_groups)
    , n_heads_(d_inner_ / d_head)
    , in_proj_dim_(2 * d_inner_ + 2 * d_state_ * n_groups_ + n_heads_)
    , conv_dim_(d_inner_ + 2 * d_state_ * n_groups_)
    , weights_initialized_(false)
    , conv_weight_(nullptr)
    , conv_bias_(nullptr)
    , A_(nullptr)
    , dt_bias_(nullptr)
    , D_(nullptr)
    , in_proj_layer_(nullptr)
    , out_proj_layer_(nullptr)
    , norm_layer_(nullptr)
    , program_(nullptr)
    , ssm_kernel_(nullptr)
    , conv_forward_kernel_(nullptr)
    , conv_update_kernel_(nullptr)
{
    if (!ctx_) {
        throw std::runtime_error("OpenCLContext is null");
    }
    
    if (d_inner_ % d_head_ != 0) {
        throw std::runtime_error("d_inner must be divisible by d_head");
    }
    
    // Debug output
    size_t in_proj_params = static_cast<size_t>(in_proj_dim_) * d_model_;
    size_t conv_params = static_cast<size_t>(conv_dim_) * kernel_size_;
    size_t out_proj_params = static_cast<size_t>(d_model_) * d_inner_;
    size_t ssm_params = n_heads_ + n_heads_ + n_heads_;  // A, dt_bias, D
    size_t total_params = in_proj_params + conv_params + out_proj_params + ssm_params + conv_dim_; // + conv_bias
    size_t buffer_size_mb = (total_params * sizeof(float)) / (1024 * 1024);
    // std::cout << "  [SSD] d_model=" << d_model_
    //           << ", expand=" << (d_inner_ / d_model_)
    //           << ", d_inner=" << d_inner_
    //           << ", kernel_size=" << kernel_size_
    //           << ", d_state=" << d_state_
    //           << ", d_head=" << d_head_
    //           << ", n_heads=" << n_heads_
    //           << ", n_groups=" << n_groups_
    //           << ", in_proj_dim=" << in_proj_dim_
    //           << ", conv_dim=" << conv_dim_
    //           << ", params=" << total_params
    //           << " (in_proj:" << in_proj_params << ", conv:" << conv_params 
    //           << ", out_proj:" << out_proj_params << ", ssm:" << ssm_params << ")"
    //           << ", buffer_size=" << buffer_size_mb << " MB" << std::endl;
    
    // Create linear layers
    in_proj_layer_ = std::make_unique<LinearLayer>(ctx_, d_model_, in_proj_dim_, false);
    out_proj_layer_ = std::make_unique<LinearLayer>(ctx_, d_inner_, d_model_, false);
    
    // Create RMS norm layer
    norm_layer_ = std::make_unique<RMSNormLayer>(ctx_, d_inner_);
    
    buildKernels();
}

SSDLayer::~SSDLayer() {
    if (ssm_kernel_) clReleaseKernel(ssm_kernel_);
    if (conv_forward_kernel_) clReleaseKernel(conv_forward_kernel_);
    if (conv_update_kernel_) clReleaseKernel(conv_update_kernel_);
    if (program_) clReleaseProgram(program_);
    // Linear layers will clean themselves up
    // TODO: Release all buffers
}

void SSDLayer::buildKernels() {
    // Embedded conv1d kernel source
    const char* conv1d_cl_source = R"(
// 1D Convolution kernels for SSD layer

#define SILU(x) ({ \
    float y = 1.0f / (1.0f + exp(-fabs(x))); \
    (x < 0.0f) ? (1.0f - y) * x : y * x; \
})

// Convolution forward pass (for prefill)
// x: [batch_size, n_channels, seq_len]
// w: [n_channels, kernel_size]
// b: [n_channels]
// y: [batch_size, n_channels, seq_len]
__kernel void conv1d_forward_kernel(
    __global const float* x,
    __global const float* w,
    __global const float* b,
    __global float* y,
    const int batch_size,
    const int n_channels,
    const int seq_len,
    const int kernel_size,
    const int swish_activation  // 1 if apply Swish, 0 otherwise
) {
    const int batch_idx = get_global_id(0);
    const int channel_idx = get_global_id(1);
    const int seq_idx = get_global_id(2);
    
    if (batch_idx >= batch_size || channel_idx >= n_channels || seq_idx >= seq_len) return;
    
    // Check bounds: can only compute if we have enough sequence length
    if (seq_idx + kernel_size > seq_len) {
        y[batch_idx * n_channels * seq_len + channel_idx * seq_len + seq_idx] = 0.0f;
        return;
    }
    
    float sum = 0.0f;
    int w_start = channel_idx * kernel_size;
    
    for (int k = 0; k < kernel_size; ++k) {
        int x_idx = batch_idx * n_channels * seq_len + channel_idx * seq_len + seq_idx + k;
        sum += w[w_start + k] * x[x_idx];
    }
    
    sum += b[channel_idx];
    
    if (swish_activation) {
        sum = SILU(sum);
    }
    
    int y_idx = batch_idx * n_channels * seq_len + channel_idx * seq_len + seq_idx;
    y[y_idx] = sum;
}

// Convolution update (for step function)
// x: [batch_size, n_channels] (single token)
// w: [n_channels, kernel_size]
// b: [n_channels]
// state: [batch_size, n_channels, kernel_size - 1] (conv state)
// y: [batch_size, n_channels]
// next_state: [batch_size, n_channels, kernel_size - 1]
__kernel void conv1d_update_kernel(
    __global const float* x,
    __global const float* w,
    __global const float* b,
    __global const float* state,
    __global float* y,
    __global float* next_state,
    const int batch_size,
    const int n_channels,
    const int kernel_size,
    const int swish_activation
) {
    const int batch_idx = get_global_id(0);
    const int channel_idx = get_global_id(1);
    
    if (batch_idx >= batch_size || channel_idx >= n_channels) return;
    
    int x_idx = batch_idx * n_channels + channel_idx;
    int w_start = channel_idx * kernel_size;
    int state_start = batch_idx * n_channels * (kernel_size - 1) + channel_idx * (kernel_size - 1);
    
    float sum = 0.0f;
    
    // Use state for first (kernel_size - 1) elements
    for (int k = 0; k < kernel_size - 1; ++k) {
        sum += w[w_start + k] * state[state_start + k];
    }
    
    // Use current input for last element
    sum += w[w_start + kernel_size - 1] * x[x_idx];
    sum += b[channel_idx];
    
    if (swish_activation) {
        sum = SILU(sum);
    }
    
    y[x_idx] = sum;
    
    // Update state: shift left and append new value
    for (int k = 0; k < kernel_size - 2; ++k) {
        next_state[state_start + k] = state[state_start + k + 1];
    }
    next_state[state_start + kernel_size - 2] = x[x_idx];
}
)";

    // Embedded ssm_update kernel source
    const char* ssm_update_cl_source = R"(
// OpenCL kernel for SSM update operation
// Based on the Metal implementation from cartesia-metal

#define SILU(x) ({ \
    float y = 1.0f / (1.0f + exp(-fabs(x))); \
    (x < 0.0f) ? (1.0f - y) * x : y * x; \
})

#define SOFTPLUS(x) ({ \
    float y = log1p(exp(x)); \
    (x > 20.0f) ? x : y; \
})

__kernel void ssm_update_kernel(
    __global const float* x,
    __global const float* dt,
    __global const float* A,
    __global const float* B,
    __global const float* C,
    __global const float* D,
    __global const float* z,
    __global const float* state,
    __global float* y,
    __global float* next_state,
    const int state_size,
    const int channel_size
) {
    const int batch_idx = get_global_id(0);
    const int channel_idx = get_global_id(1);
    
    const int cb_start_idx = batch_idx * state_size;  // CB are data controlled
    const int x_idx = batch_idx * channel_size + channel_idx;
    const int state_start_idx = x_idx * state_size;

    float this_x = x[x_idx];
    this_x = SILU(this_x); // SILU activation 

    float this_z = z[x_idx];
    this_z = SILU(this_z); // SILU activation 

    float delta = SOFTPLUS(dt[x_idx]);  // Softplus log(1 + exp(dt))

    float temp = 0.0f;
    for (int i = 0; i < state_size; ++i) {
        int cb_idx = cb_start_idx + i;
        int state_idx = state_start_idx + i;
        float this_new_state = state[state_idx] * exp(A[i] * delta) + B[cb_idx] * delta * this_x; 
        next_state[state_idx] = this_new_state;
        temp = temp + this_new_state * C[cb_idx];
    }
    temp = temp + D[channel_idx] * this_x;  // Skip connection
    temp = temp * this_z; // Out gate with z
    y[x_idx] = temp; 
}
)";
    
    // Build program
    auto& ctx_mgr = OpenCLContextManager::getInstance();
    std::vector<std::string> sources = {std::string(conv1d_cl_source), std::string(ssm_update_cl_source)};
    std::string cache_key = ctx_mgr.generateCacheKey(sources);
    program_ = ctx_mgr.buildProgram(sources, cache_key);
    
    // Create kernels
    conv_forward_kernel_ = ctx_mgr.getKernel(program_, "conv1d_forward_kernel");
    conv_update_kernel_ = ctx_mgr.getKernel(program_, "conv1d_update_kernel");
    ssm_kernel_ = ctx_mgr.getKernel(program_, "ssm_update_kernel");
}

void SSDLayer::initializeWeights(
    const std::vector<float>& in_proj_weights,
    const std::vector<float>& conv_weight,
    const std::vector<float>& conv_bias,
    const std::vector<float>& A,
    const std::vector<float>& dt_bias,
    const std::vector<float>& D,
    const std::vector<float>& out_proj_weights
) {
    // Validate sizes
    if (in_proj_weights.size() != static_cast<size_t>(in_proj_dim_ * d_model_)) {
        throw std::runtime_error("Invalid in_proj weights size");
    }
    if (conv_weight.size() != static_cast<size_t>(conv_dim_ * kernel_size_)) {
        throw std::runtime_error("Invalid conv_weight size");
    }
    if (conv_bias.size() != static_cast<size_t>(conv_dim_)) {
        throw std::runtime_error("Invalid conv_bias size");
    }
    if (A.size() != static_cast<size_t>(n_heads_)) {
        throw std::runtime_error("Invalid A size");
    }
    if (dt_bias.size() != static_cast<size_t>(n_heads_)) {
        throw std::runtime_error("Invalid dt_bias size");
    }
    if (D.size() != static_cast<size_t>(n_heads_)) {
        throw std::runtime_error("Invalid D size");
    }
    if (out_proj_weights.size() != static_cast<size_t>(d_model_ * d_inner_)) {
        throw std::runtime_error("Invalid out_proj weights size");
    }
    
    // Initialize linear layers
    in_proj_layer_->initializeWeights(in_proj_weights);
    out_proj_layer_->initializeWeights(out_proj_weights);
    
    // Create OpenCL buffers for conv and SSM weights
    cl_context context = ctx_->getContext();
    cl_int err;
    
    conv_weight_ = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        conv_weight.size() * sizeof(float), (void*)conv_weight.data(), &err
    );
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create conv_weight buffer");
    
    conv_bias_ = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        conv_bias.size() * sizeof(float), (void*)conv_bias.data(), &err
    );
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create conv_bias buffer");
    
    A_ = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        A.size() * sizeof(float), (void*)A.data(), &err
    );
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create A buffer");
    
    dt_bias_ = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        dt_bias.size() * sizeof(float), (void*)dt_bias.data(), &err
    );
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create dt_bias buffer");
    
    D_ = clCreateBuffer(
        context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        D.size() * sizeof(float), (void*)D.data(), &err
    );
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create D buffer");
    
    weights_initialized_ = true;
}

void SSDLayer::splitInProjOutput(
    cl_mem in_proj_output,
    int batch_size,
    int seq_len,
    cl_mem z_out,
    cl_mem xBC_out,
    cl_mem dt_out,
    cl_command_queue queue
) {
    // TODO: Implement OpenCL kernel to split in_proj output
    // For now, use CPU fallback
    // The split is: [d_inner | 2*d_inner + 2*d_state*n_groups | n_heads]
    
    size_t total_size = batch_size * seq_len * in_proj_dim_;
    std::vector<float> in_proj_cpu(total_size);
    clEnqueueReadBuffer(queue, in_proj_output, CL_TRUE, 0, 
                       total_size * sizeof(float), in_proj_cpu.data(), 0, nullptr, nullptr);
    
    // Split on CPU
    std::vector<float> z_cpu(batch_size * seq_len * d_inner_);
    std::vector<float> xBC_cpu(batch_size * seq_len * (2 * d_inner_ + 2 * d_state_ * n_groups_));
    std::vector<float> dt_cpu(batch_size * seq_len * n_heads_);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int base_idx = (b * seq_len + s) * in_proj_dim_;
            
            // z: first d_inner elements
            for (int i = 0; i < d_inner_; ++i) {
                z_cpu[(b * seq_len + s) * d_inner_ + i] = in_proj_cpu[base_idx + i];
            }
            
            // xBC: next (2*d_inner + 2*d_state*n_groups) elements
            int xBC_start = base_idx + d_inner_;
            for (int i = 0; i < 2 * d_inner_ + 2 * d_state_ * n_groups_; ++i) {
                xBC_cpu[(b * seq_len + s) * (2 * d_inner_ + 2 * d_state_ * n_groups_) + i] = 
                    in_proj_cpu[xBC_start + i];
            }
            
            // dt: last n_heads elements
            int dt_start = base_idx + d_inner_ + 2 * d_inner_ + 2 * d_state_ * n_groups_;
            for (int i = 0; i < n_heads_; ++i) {
                dt_cpu[(b * seq_len + s) * n_heads_ + i] = in_proj_cpu[dt_start + i];
            }
        }
    }
    
    // Write back to GPU
    cl_context context = ctx_->getContext();
    cl_int err;
    
    if (!z_out || !xBC_out || !dt_out) {
        throw std::runtime_error("Output buffers not allocated");
    }
    
    clEnqueueWriteBuffer(queue, z_out, CL_TRUE, 0,
                        z_cpu.size() * sizeof(float), z_cpu.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, xBC_out, CL_TRUE, 0,
                        xBC_cpu.size() * sizeof(float), xBC_cpu.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, dt_out, CL_TRUE, 0,
                        dt_cpu.size() * sizeof(float), dt_cpu.data(), 0, nullptr, nullptr);
}

cl_mem SSDLayer::forward(
    cl_mem input,
    int batch_size,
    int seq_len,
    LayerState* state,
    cl_command_queue queue
) {
    if (!weights_initialized_) {
        throw std::runtime_error("SSD layer weights not initialized");
    }
    
    cl_context context = ctx_->getContext();
    
    // Step 1: in_proj: [batch, seq_len, d_model] -> [batch, seq_len, in_proj_dim]
    cl_mem in_proj_out = in_proj_layer_->forward(input, batch_size, seq_len, queue);
    
    // Step 2: Split into z, xBC, dt
    // TODO: Allocate buffers properly
    // For now, placeholder
    cl_mem z_buf = in_proj_out;  // Placeholder
    cl_mem xBC_buf = in_proj_out;  // Placeholder
    cl_mem dt_buf = in_proj_out;   // Placeholder
    
    // Step 3: conv1d on xBC
    // TODO: Implement conv1d_forward with Swish
    
    // Step 4: Split xBC into x, B, C
    // TODO: Implement split
    
    // Step 5: SSM update
    // TODO: Use existing ssm_update_kernel or create ssd_update_kernel
    
    // Step 6: Apply gate: Swish(z) * x, then RMS norm
    // TODO: Implement
    
    // Step 7: out_proj
    // TODO: Implement
    
    // For now, return input unchanged (placeholder)
    return input;
}

cl_mem SSDLayer::step(
    cl_mem input,
    int batch_size,
    LayerState* state,
    cl_command_queue queue
) {
    if (!weights_initialized_) {
        throw std::runtime_error("SSD layer weights not initialized");
    }
    
    // TODO: Implement step function
    // - in_proj step
    // - conv1d_update (maintains conv_state)
    // - SSM update (maintains ssm_state)
    // - Gate and norm
    // - out_proj
    
    return input;  // Placeholder
}

} // namespace cartesia_opencl

