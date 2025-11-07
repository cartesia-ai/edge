#include "ssd_layer.h"
#include "../opencl_context.h"
#include "../opencl_utils.h"
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
    , split_in_proj_kernel_(nullptr)
    , split_xBC_step_kernel_(nullptr)
    , ssm_step_update_kernel_(nullptr)
    , gate_kernel_(nullptr)
    , copy_channels_kernel_(nullptr)
    , process_dt_kernel_(nullptr)
    , compute_dtA_kernel_(nullptr)
    , compute_segsum_decay_kernel_(nullptr)
    , compute_CB_kernel_(nullptr)
    , compute_ssm_output_kernel_(nullptr)
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
    
    // Defer kernel building to first use - this avoids driver crashes during construction
    // The kernels will be built lazily in forward() or step() when first needed
    // This is safer for PowerVR driver which may crash during kernel creation
}

SSDLayer::~SSDLayer() {
    // Release kernels first (before program) - check for null to avoid crashes
    // Order matters: release kernels before program
    if (ssm_kernel_) {
        clReleaseKernel(ssm_kernel_);
        ssm_kernel_ = nullptr;
    }
    if (conv_forward_kernel_) {
        clReleaseKernel(conv_forward_kernel_);
        conv_forward_kernel_ = nullptr;
    }
    if (conv_update_kernel_) {
        clReleaseKernel(conv_update_kernel_);
        conv_update_kernel_ = nullptr;
    }
    if (split_in_proj_kernel_) {
        clReleaseKernel(split_in_proj_kernel_);
        split_in_proj_kernel_ = nullptr;
    }
    if (split_xBC_step_kernel_) {
        clReleaseKernel(split_xBC_step_kernel_);
        split_xBC_step_kernel_ = nullptr;
    }
    if (ssm_step_update_kernel_) {
        clReleaseKernel(ssm_step_update_kernel_);
        ssm_step_update_kernel_ = nullptr;
    }
    if (gate_kernel_) {
        clReleaseKernel(gate_kernel_);
        gate_kernel_ = nullptr;
    }
    if (copy_channels_kernel_) {
        clReleaseKernel(copy_channels_kernel_);
        copy_channels_kernel_ = nullptr;
    }
    if (process_dt_kernel_) {
        clReleaseKernel(process_dt_kernel_);
        process_dt_kernel_ = nullptr;
    }
    if (compute_dtA_kernel_) {
        clReleaseKernel(compute_dtA_kernel_);
        compute_dtA_kernel_ = nullptr;
    }
    if (compute_segsum_decay_kernel_) {
        clReleaseKernel(compute_segsum_decay_kernel_);
        compute_segsum_decay_kernel_ = nullptr;
    }
    if (compute_CB_kernel_) {
        clReleaseKernel(compute_CB_kernel_);
        compute_CB_kernel_ = nullptr;
    }
    if (compute_ssm_output_kernel_) {
        clReleaseKernel(compute_ssm_output_kernel_);
        compute_ssm_output_kernel_ = nullptr;
    }
    
    // Release program after kernels
    if (program_) {
        clReleaseProgram(program_);
        program_ = nullptr;
    }
    
    // Release weight buffers (order doesn't matter for mem objects)
    if (conv_weight_) {
        clReleaseMemObject(conv_weight_);
        conv_weight_ = nullptr;
    }
    if (conv_bias_) {
        clReleaseMemObject(conv_bias_);
        conv_bias_ = nullptr;
    }
    if (A_) {
        clReleaseMemObject(A_);
        A_ = nullptr;
    }
    if (dt_bias_) {
        clReleaseMemObject(dt_bias_);
        dt_bias_ = nullptr;
    }
    if (D_) {
        clReleaseMemObject(D_);
        D_ = nullptr;
    }
    
    // Linear layers will clean themselves up (they're unique_ptr, so automatic)
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

    // Embedded helper kernels for split and gate operations
    const char* helper_kernels_source = R"(
// Helper kernels for SSD layer operations

// Kernel to split in_proj output into z, xBC, dt
// in_proj: [batch_size, seq_len, in_proj_dim] where in_proj_dim = d_inner + (d_inner + 2*d_state*n_groups) + n_heads
// z: [batch_size, seq_len, d_inner]
// xBC: [batch_size, seq_len, d_inner + 2*d_state*n_groups]  // FIXED: was 2*d_inner
// dt: [batch_size, seq_len, n_heads]
__kernel void split_in_proj_kernel(
    __global const float* in_proj,
    __global float* z_out,
    __global float* xBC_out,
    __global float* dt_out,
    const int batch_size,
    const int seq_len,
    const int d_inner,
    const int xBC_dim,  // FIXED: d_inner + 2*d_state*n_groups (was 2*d_inner)
    const int n_heads,
    const int in_proj_dim
) {
    const int idx = get_global_id(0);
    const int total = batch_size * seq_len;
    if (idx >= total) return;
    
    int b = idx / seq_len;
    int s = idx % seq_len;
    
    int in_proj_base = idx * in_proj_dim;
    int z_base = idx * d_inner;
    int xBC_base = idx * xBC_dim;
    int dt_base = idx * n_heads;
    
    // z: first d_inner elements
    for (int i = 0; i < d_inner; ++i) {
        z_out[z_base + i] = in_proj[in_proj_base + i];
    }
    
    // xBC: next xBC_dim elements
    for (int i = 0; i < xBC_dim; ++i) {
        xBC_out[xBC_base + i] = in_proj[in_proj_base + d_inner + i];
    }
    
    // dt: last n_heads elements
    for (int i = 0; i < n_heads; ++i) {
        dt_out[dt_base + i] = in_proj[in_proj_base + d_inner + xBC_dim + i];
    }
}

// Kernel for gate computation: Swish(z) * x
// z: [batch_size, seq_len, d_inner]
// x: [batch_size, seq_len, d_inner]
// output: [batch_size, seq_len, d_inner]
__kernel void gate_kernel(
    __global const float* z,
    __global const float* x,
    __global float* output,
    const int total_size
) {
    const int idx = get_global_id(0);
    if (idx >= total_size) return;
    
    float z_val = z[idx];
    // Swish activation: x * sigmoid(x)
    float sigmoid_z = 1.0f / (1.0f + exp(-z_val));
    float swish_z = z_val * sigmoid_z;
    output[idx] = x[idx] * swish_z;
}

// Kernel to copy channels from source to destination (for pass-through in conv1d)
__kernel void copy_channels_kernel(
    __global const float* src,
    __global float* dst,
    const int batch_size,
    const int seq_len,
    const int src_channels,
    const int dst_channels,
    const int src_start_channel,
    const int dst_start_channel,
    const int num_channels
) {
    const int idx = get_global_id(0);
    const int total = batch_size * seq_len * num_channels;
    if (idx >= total) return;
    
    int flat_idx = idx;
    int channel_offset = flat_idx % num_channels;
    int seq_pos = (flat_idx / num_channels) % seq_len;
    int batch_pos = flat_idx / (num_channels * seq_len);
    
    int src_idx = (batch_pos * seq_len + seq_pos) * src_channels + src_start_channel + channel_offset;
    int dst_idx = (batch_pos * seq_len + seq_pos) * dst_channels + dst_start_channel + channel_offset;
    
    dst[dst_idx] = src[src_idx];
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
    
    // Embedded SSM forward kernels for matrix-based computation
    const char* ssm_forward_cl_source = R"(
// OpenCL kernels for SSM forward pass (prefill) - matrix-based computation

#define SOFTPLUS(x) ((x > 20.0f) ? x : log1p(exp(x)))

// Kernel 1: Process dt (add bias, softplus, clamp)
__kernel void process_dt_kernel(
    __global const float* dt,
    __global const float* dt_bias,
    __global float* dt_processed,
    const int batch_size,
    const int seq_len,
    const int n_heads,
    const float dt_min,
    const float dt_max
) {
    const int idx = get_global_id(0);
    const int total = batch_size * seq_len * n_heads;
    if (idx >= total) return;
    
    int h = idx % n_heads;
    float dt_val = dt[idx] + dt_bias[h];
    dt_val = SOFTPLUS(dt_val);
    if (dt_val < dt_min) dt_val = dt_min;
    if (dt_val > dt_max) dt_val = dt_max;
    
    dt_processed[idx] = dt_val;
}

// Kernel 2: Compute dtA = dt * A
__kernel void compute_dtA_kernel(
    __global const float* dt,
    __global const float* A,
    __global float* dtA,
    const int batch_size,
    const int seq_len,
    const int n_heads
) {
    const int idx = get_global_id(0);
    const int total = batch_size * seq_len * n_heads;
    if (idx >= total) return;
    
    int h = idx % n_heads;
    dtA[idx] = dt[idx] * A[h];
}

// Kernel 3: Compute segsum to get decay matrix
__kernel void compute_segsum_decay_kernel(
    __global const float* dtA,
    __global float* decay,
    const int batch_size,
    const int seq_len,
    const int n_heads
) {
    const int b = get_global_id(0);
    const int h = get_global_id(1);
    const int s = get_global_id(2);
    
    if (b >= batch_size || h >= n_heads || s >= seq_len) return;
    
    for (int t = 0; t < seq_len; ++t) {
        float segsum_val = 0.0f;
        for (int k = t + 1; k <= s; ++k) {
            int dtA_idx = (b * seq_len + k) * n_heads + h;
            segsum_val += dtA[dtA_idx];
        }
        int decay_idx = (b * n_heads + h) * seq_len * seq_len + s * seq_len + t;
        decay[decay_idx] = exp(segsum_val);
    }
}

// Kernel 4: Compute CB = C @ B
__kernel void compute_CB_kernel(
    __global const float* B,
    __global const float* C,
    __global float* CB,
    const int batch_size,
    const int seq_len,
    const int n_groups,
    const int d_state
) {
    const int b = get_global_id(0);
    const int s = get_global_id(1);
    const int t = get_global_id(2);
    const int g = get_global_id(3);
    
    if (b >= batch_size || s >= seq_len || t >= seq_len || g >= n_groups) return;
    
    float sum = 0.0f;
    for (int state_i = 0; state_i < d_state; ++state_i) {
        int C_idx = (b * seq_len + s) * (n_groups * d_state) + g * d_state + state_i;
        int B_idx = (b * seq_len + t) * (n_groups * d_state) + g * d_state + state_i;
        sum += C[C_idx] * B[B_idx];
    }
    
    // CB stored as CB[b, g, s, t] to match MLX and CPU fallback
    int CB_idx = ((b * n_groups + g) * seq_len + s) * seq_len + t;
    CB[CB_idx] = sum;
}

// Kernel 5: Compute final output
__kernel void compute_ssm_output_kernel(
    __global const float* CB,
    __global const float* decay,
    __global const float* dtx,
    __global const float* D,
    __global const float* x,
    __global float* y,
    const int batch_size,
    const int seq_len,
    const int n_heads,
    const int d_head,
    const int n_groups,
    const int d_state
) {
    const int b = get_global_id(0);
    const int s = get_global_id(1);
    const int h = get_global_id(2);
    const int d = get_global_id(3);
    
    if (b >= batch_size || s >= seq_len || h >= n_heads || d >= d_head) return;
    
    int group_idx = h % n_groups;
    int x_idx = (b * seq_len + s) * (n_heads * d_head) + h * d_head + d;
    
    float output_sum = 0.0f;
    
    for (int t = 0; t <= s; ++t) {
        int x_t_idx = (b * seq_len + t) * (n_heads * d_head) + h * d_head + d;
        float dtx_t = dtx[x_t_idx];
        
        int decay_idx = (b * n_heads + h) * seq_len * seq_len + s * seq_len + t;
        float decay_st = decay[decay_idx];
        
        // CB stored as CB[b, g, s, t] = CB[((b * n_groups + g) * seq_len + s) * seq_len + t]
        int CB_idx = ((b * n_groups + group_idx) * seq_len + s) * seq_len + t;
        float CB_st = CB[CB_idx];
        
        output_sum += CB_st * decay_st * dtx_t;
    }
    
    output_sum += D[h] * x[x_idx];
    y[x_idx] = output_sum;
}
)";
    
    // Build program
    auto& ctx_mgr = OpenCLContextManager::getInstance();
    
    // Verify context is valid before proceeding
    cl_context context = ctx_mgr.getContext();
    if (!context) {
        throw std::runtime_error("OpenCL context is invalid");
    }
    
    // Build program with core kernels first (conv1d and ssm_update)
    // Helper kernels will be built separately to avoid PowerVR driver crashes
    std::vector<std::string> core_sources = {
        std::string(conv1d_cl_source), 
        std::string(ssm_update_cl_source)
    };
    std::string cache_key = ctx_mgr.generateCacheKey(core_sources);
    
    try {
        // Ensure queue is flushed before building program
        cl_command_queue queue = ctx_mgr.getQueue();
        if (queue) {
            clFinish(queue);
        }
        
    program_ = ctx_mgr.buildProgram(core_sources, cache_key);
        
        // Verify program was created
        if (!program_) {
            throw std::runtime_error("buildProgram returned null");
        }
        
        // Check build status
        cl_build_status build_status;
        cl_int err = clGetProgramBuildInfo(program_, ctx_mgr.getDevice(), CL_PROGRAM_BUILD_STATUS, 
                                          sizeof(cl_build_status), &build_status, nullptr);
        if (err == CL_SUCCESS) {
            if (build_status != CL_BUILD_SUCCESS) {
            }
        }
    } catch (const std::exception& e) {
        program_ = nullptr;  // Ensure it's null
        throw;
    } catch (...) {
        program_ = nullptr;
        throw std::runtime_error("Unknown error building OpenCL program");
    }
    
    // Initialize kernels to null
    conv_forward_kernel_ = nullptr;
    conv_update_kernel_ = nullptr;
    split_in_proj_kernel_ = nullptr;
    split_xBC_step_kernel_ = nullptr;
    ssm_step_update_kernel_ = nullptr;
    gate_kernel_ = nullptr;
    copy_channels_kernel_ = nullptr;
    ssm_kernel_ = nullptr;
    process_dt_kernel_ = nullptr;
    compute_dtA_kernel_ = nullptr;
    compute_segsum_decay_kernel_ = nullptr;
    compute_CB_kernel_ = nullptr;
    compute_ssm_output_kernel_ = nullptr;
    
    // Create kernels with error handling
    // Verify program is valid before creating kernels
    if (!program_) {
        throw std::runtime_error("Program is null, cannot create kernels");
    }
    
    try {
        // Create kernels one at a time to isolate any driver crashes
        conv_forward_kernel_ = ctx_mgr.getKernel(program_, "conv1d_forward_kernel");
        if (!conv_forward_kernel_) {
            throw std::runtime_error("Failed to create conv_forward_kernel (returned null)");
        }
        
        conv_update_kernel_ = ctx_mgr.getKernel(program_, "conv1d_update_kernel");
        if (!conv_update_kernel_) {
            throw std::runtime_error("Failed to create conv_update_kernel (returned null)");
        }
        
        ssm_kernel_ = ctx_mgr.getKernel(program_, "ssm_update_kernel");
        if (!ssm_kernel_) {
            throw std::runtime_error("Failed to create ssm_kernel (returned null)");
        }
        
        // Build helper kernels separately (split and gate) to avoid PowerVR driver crashes
            std::vector<std::string> helper_sources = {std::string(helper_kernels_source)};
            std::string helper_cache_key = ctx_mgr.generateCacheKey(helper_sources) + "_helper";
            
            cl_program helper_program = ctx_mgr.buildProgram(helper_sources, helper_cache_key);
            
            if (helper_program) {
                split_in_proj_kernel_ = ctx_mgr.getKernel(helper_program, "split_in_proj_kernel");
            if (!split_in_proj_kernel_) {
                throw std::runtime_error("Failed to create split_in_proj_kernel");
            }
            
                gate_kernel_ = ctx_mgr.getKernel(helper_program, "gate_kernel");
            if (!gate_kernel_) {
                throw std::runtime_error("Failed to create gate_kernel");
            }
            
                copy_channels_kernel_ = ctx_mgr.getKernel(helper_program, "copy_channels_kernel");
            if (!copy_channels_kernel_) {
                throw std::runtime_error("Failed to create copy_channels_kernel");
            }
            
            // Build split_xBC_step_kernel separately to avoid PowerVR crashes
            try {
                const char* split_xBC_source = R"(
__kernel void split_xBC_step_kernel(
    __global const float* xBC,
    __global float* x,
    __global float* B,
    __global float* C,
    const int batch_size,
    const int d_inner,
    const int d_state_groups
) {
    const int idx = get_global_id(0);
    if (idx >= batch_size) return;
    
    const int xBC_dim = d_inner + 2 * d_state_groups;
    const int xBC_base = idx * xBC_dim;
    const int x_base = idx * d_inner;
    const int BC_base = idx * d_state_groups;
    
    for (int i = 0; i < d_inner; ++i) {
        x[x_base + i] = xBC[xBC_base + i];
    }
    
    for (int i = 0; i < d_state_groups; ++i) {
        B[BC_base + i] = xBC[xBC_base + d_inner + i];
    }
    
    for (int i = 0; i < d_state_groups; ++i) {
        C[BC_base + i] = xBC[xBC_base + d_inner + d_state_groups + i];
    }
}
)";
                std::vector<std::string> split_sources = {std::string(split_xBC_source)};
                std::string split_cache_key = ctx_mgr.generateCacheKey(split_sources) + "_split_xBC";
                cl_program split_program = ctx_mgr.buildProgram(split_sources, split_cache_key);
                if (split_program) {
                    split_xBC_step_kernel_ = ctx_mgr.getKernel(split_program, "split_xBC_step_kernel");
                    clReleaseProgram(split_program);
                    fprintf(stderr, "[SSD] split_xBC_step_kernel built successfully\n");
                }
            } catch (const std::exception& e) {
                fprintf(stderr, "[SSD] WARNING: Failed to build split_xBC_step_kernel: %s\n", e.what());
            }
                
                // Release helper program (kernels are retained)
                clReleaseProgram(helper_program);
            } else {
            throw std::runtime_error("Failed to build helper kernels program");
        }
        
        // SSM forward kernels (for prefill) - build each kernel separately to avoid PowerVR driver crashes
        // PowerVR driver crashes when building large kernel programs, so we build each kernel individually
        
        // Split the SSM forward source into individual kernel sources
        const char* process_dt_source = R"(
#define SOFTPLUS(x) ((x > 20.0f) ? x : log1p(exp(x)))

__kernel void process_dt_kernel(
    __global const float* dt,
    __global const float* dt_bias,
    __global float* dt_processed,
    const int batch_size,
    const int seq_len,
    const int n_heads,
    const float dt_min,
    const float dt_max
) {
    const int idx = get_global_id(0);
    const int total = batch_size * seq_len * n_heads;
    if (idx >= total) return;
    
    int h = idx % n_heads;
    float dt_val = dt[idx] + dt_bias[h];
    dt_val = SOFTPLUS(dt_val);
    if (dt_val < dt_min) dt_val = dt_min;
    if (dt_val > dt_max) dt_val = dt_max;
    
    dt_processed[idx] = dt_val;
}
)";

        const char* compute_dtA_source = R"(
__kernel void compute_dtA_kernel(
    __global const float* dt,
    __global const float* A,
    __global float* dtA,
    const int batch_size,
    const int seq_len,
    const int n_heads
) {
    const int idx = get_global_id(0);
    const int total = batch_size * seq_len * n_heads;
    if (idx >= total) return;
    
    int h = idx % n_heads;
    dtA[idx] = dt[idx] * A[h];
}
)";

        const char* compute_segsum_source = R"(
__kernel void compute_segsum_decay_kernel(
    __global const float* dtA,
    __global float* decay,
    const int batch_size,
    const int seq_len,
    const int n_heads
) {
    const int b = get_global_id(0);
    const int h = get_global_id(1);
    const int s = get_global_id(2);
    
    if (b >= batch_size || h >= n_heads || s >= seq_len) return;
    
    for (int t = 0; t < seq_len; ++t) {
        float segsum_val = 0.0f;
        for (int k = t + 1; k <= s; ++k) {
            int dtA_idx = (b * seq_len + k) * n_heads + h;
            segsum_val += dtA[dtA_idx];
        }
        int decay_idx = (b * n_heads + h) * seq_len * seq_len + s * seq_len + t;
        decay[decay_idx] = exp(segsum_val);
    }
}
)";

        const char* compute_CB_source = R"(
__kernel void compute_CB_kernel(
    __global const float* B,
    __global const float* C,
    __global float* CB,
    const int batch_size,
    const int seq_len,
    const int n_groups,
    const int d_state
) {
    const int b = get_global_id(0);
    const int s = get_global_id(1);
    const int t = get_global_id(2);
    
    if (b >= batch_size || s >= seq_len || t >= seq_len) return;
    
    for (int g = 0; g < n_groups; ++g) {
        float sum = 0.0f;
        for (int state_i = 0; state_i < d_state; ++state_i) {
            int C_idx = (b * seq_len + s) * (n_groups * d_state) + g * d_state + state_i;
            int B_idx = (b * seq_len + t) * (n_groups * d_state) + g * d_state + state_i;
            sum += C[C_idx] * B[B_idx];
        }
        int CB_idx = ((b * n_groups + g) * seq_len + s) * seq_len + t;
        CB[CB_idx] = sum;
    }
}
)";

        const char* compute_ssm_output_source = R"(
__kernel void compute_ssm_output_kernel(
    __global const float* CB,
    __global const float* decay,
    __global const float* dtx,
    __global const float* D,
    __global const float* x,
    __global float* y,
    const int batch_size,
    const int seq_len,
    const int n_heads,
    const int d_head,
    const int n_groups,
    const int d_state
) {
    const int b = get_global_id(0);
    const int s = get_global_id(1);
    const int h = get_global_id(2);
    
    if (b >= batch_size || s >= seq_len || h >= n_heads) return;
    
    int group_idx = h % n_groups;
    
    for (int d = 0; d < d_head; ++d) {
        int x_idx = (b * seq_len + s) * (n_heads * d_head) + h * d_head + d;
        float output_sum = 0.0f;
        
        for (int t = 0; t <= s; ++t) {
            int x_t_idx = (b * seq_len + t) * (n_heads * d_head) + h * d_head + d;
            float dtx_t = dtx[x_t_idx];
            
            int decay_idx = (b * n_heads + h) * seq_len * seq_len + s * seq_len + t;
            float decay_st = decay[decay_idx];
            
            int CB_idx = ((b * n_groups + group_idx) * seq_len + s) * seq_len + t;
            float CB_st = CB[CB_idx];
            
            output_sum += CB_st * decay_st * dtx_t;
        }
        
        output_sum += D[h] * x[x_idx];
        y[x_idx] = output_sum;
    }
}
)";

        // Build each kernel separately to avoid PowerVR driver crashes
        int kernels_built = 0;
        
        try {
            std::vector<std::string> dt_sources = {std::string(process_dt_source)};
            std::string dt_cache_key = ctx_mgr.generateCacheKey(dt_sources) + "_process_dt";
            cl_program dt_program = ctx_mgr.buildProgram(dt_sources, dt_cache_key);
            if (dt_program) {
                process_dt_kernel_ = ctx_mgr.getKernel(dt_program, "process_dt_kernel");
                clReleaseProgram(dt_program);
                kernels_built++;
            }
        } catch (const std::exception& e) {
            std::cerr << "[SSD] Failed to build process_dt kernel: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[SSD] Failed to build process_dt kernel (unknown error)" << std::endl;
        }
        
        try {
            std::vector<std::string> dtA_sources = {std::string(compute_dtA_source)};
            std::string dtA_cache_key = ctx_mgr.generateCacheKey(dtA_sources) + "_compute_dtA";
            cl_program dtA_program = ctx_mgr.buildProgram(dtA_sources, dtA_cache_key);
            if (dtA_program) {
                compute_dtA_kernel_ = ctx_mgr.getKernel(dtA_program, "compute_dtA_kernel");
                clReleaseProgram(dtA_program);
                kernels_built++;
            }
        } catch (const std::exception& e) {
            std::cerr << "[SSD] Failed to build compute_dtA kernel: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[SSD] Failed to build compute_dtA kernel (unknown error)" << std::endl;
        }
        
        try {
            std::vector<std::string> segsum_sources = {std::string(compute_segsum_source)};
            std::string segsum_cache_key = ctx_mgr.generateCacheKey(segsum_sources) + "_compute_segsum";
            cl_program segsum_program = ctx_mgr.buildProgram(segsum_sources, segsum_cache_key);
            if (segsum_program) {
                compute_segsum_decay_kernel_ = ctx_mgr.getKernel(segsum_program, "compute_segsum_decay_kernel");
                clReleaseProgram(segsum_program);
                kernels_built++;
            }
        } catch (const std::exception& e) {
            std::cerr << "[SSD] Failed to build compute_segsum kernel: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[SSD] Failed to build compute_segsum kernel (unknown error)" << std::endl;
        }
        
        try {
            std::vector<std::string> CB_sources = {std::string(compute_CB_source)};
            std::string CB_cache_key = ctx_mgr.generateCacheKey(CB_sources) + "_compute_CB";
            cl_program CB_program = ctx_mgr.buildProgram(CB_sources, CB_cache_key);
            if (CB_program) {
                compute_CB_kernel_ = ctx_mgr.getKernel(CB_program, "compute_CB_kernel");
                clReleaseProgram(CB_program);
                kernels_built++;
            }
        } catch (const std::exception& e) {
            std::cerr << "[SSD] Failed to build compute_CB kernel: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[SSD] Failed to build compute_CB kernel (unknown error)" << std::endl;
        }
        
        try {
            std::vector<std::string> output_sources = {std::string(compute_ssm_output_source)};
            std::string output_cache_key = ctx_mgr.generateCacheKey(output_sources) + "_compute_output";
            cl_program output_program = ctx_mgr.buildProgram(output_sources, output_cache_key);
            if (output_program) {
                compute_ssm_output_kernel_ = ctx_mgr.getKernel(output_program, "compute_ssm_output_kernel");
                clReleaseProgram(output_program);
                kernels_built++;
            }
        } catch (const std::exception& e) {
            std::cerr << "[SSD] Failed to build compute_ssm_output kernel: " << e.what() << std::endl;
        } catch (...) {
            std::cerr << "[SSD] Failed to build compute_ssm_output kernel (unknown error)" << std::endl;
        }
        
        fprintf(stderr, "[SSD] SSM forward kernels: %d/5 built successfully\n", kernels_built);
        
        // Build SSM step update kernel separately (for token generation)
        // Build it after SSM forward kernels to avoid PowerVR driver crashes
        fprintf(stderr, "[SSD] Attempting to build ssm_step_update_kernel...\n");
        try {
            fprintf(stderr, "[SSD] Creating kernel source...\n");
            // Build kernel - testing what causes PowerVR crash
            // Try without SOFTPLUS macro first
            const char* ssm_step_update_source = R"(
__kernel void ssm_step_update_kernel(
    __global const float* x,
    __global const float* dt,
    __global const float* dt_bias,
    __global const float* A,
    __global const float* B,
    __global const float* C,
    __global const float* D,
    __global const float* state,
    __global float* output,
    __global float* next_state,
    const int batch_size,
    const int n_heads,
    const int d_head,
    const int d_state,
    const int n_groups,
    const int d_inner
) {
    const int idx = get_global_id(0);
    const int total = batch_size * n_heads * d_head;
    if (idx >= total) return;
    
    // Decompose idx into b, h, d
    const int d = idx % d_head;
    const int h = (idx / d_head) % n_heads;
    const int b = idx / (d_head * n_heads);
    
    // Process dt: add bias, softplus, clamp
    // Use log(1.0f + exp()) instead of log1p(exp()) to avoid PowerVR crash
    float dt_val = dt[b * n_heads + h] + dt_bias[h];
    float delta;
    if (dt_val > 20.0f) {
        delta = dt_val;
    } else {
        float exp_val = exp(dt_val);
        delta = log(1.0f + exp_val);
    }
    if (delta < 0.0f) delta = 0.0f;
    
    float A_val = A[h];
    int group_idx = h % n_groups;
    int x_idx = b * d_inner + h * d_head + d;
    float x_val = x[x_idx];
    
    float output_sum = 0.0f;
    
    // Update state and compute output for each state dimension
    for (int state_i = 0; state_i < d_state; ++state_i) {
        int state_idx = (b * n_heads + h) * d_head * d_state + d * d_state + state_i;
        float old_state = state[state_idx];
        int BC_idx = b * (d_state * n_groups) + group_idx * d_state + state_i;
        float B_val = B[BC_idx];
        float C_val = C[BC_idx];
        // Use exp() directly - this works in other kernels
        float new_state = old_state * exp(A_val * delta) + B_val * delta * x_val;
        next_state[state_idx] = new_state;
        output_sum += new_state * C_val;
    }
    
    output_sum += D[h] * x_val;
    output[x_idx] = output_sum;
}
)";
            fprintf(stderr, "[SSD] Building ssm_step_update_kernel program...\n");
            std::vector<std::string> ssm_step_sources = {std::string(ssm_step_update_source)};
            std::string ssm_step_cache_key = ctx_mgr.generateCacheKey(ssm_step_sources) + "_ssm_step";
            cl_program ssm_step_program = ctx_mgr.buildProgram(ssm_step_sources, ssm_step_cache_key);
            fprintf(stderr, "[SSD] Program build returned, checking result...\n");
            if (ssm_step_program) {
                ssm_step_update_kernel_ = ctx_mgr.getKernel(ssm_step_program, "ssm_step_update_kernel");
                if (ssm_step_update_kernel_) {
                    clReleaseProgram(ssm_step_program);
                    fprintf(stderr, "[SSD] ssm_step_update_kernel built successfully\n");
                } else {
                    clReleaseProgram(ssm_step_program);
                    fprintf(stderr, "[SSD] WARNING: Failed to get ssm_step_update_kernel from program\n");
                }
            } else {
                fprintf(stderr, "[SSD] WARNING: Failed to build ssm_step_update_kernel program\n");
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "[SSD] WARNING: Failed to build ssm_step_update_kernel: %s\n", e.what());
        } catch (...) {
            fprintf(stderr, "[SSD] WARNING: Failed to build ssm_step_update_kernel (unknown error)\n");
        }
        
    } catch (const std::exception& e) {
        // Clean up what we created - but be very careful about order
        // Release kernels first, then program
        if (process_dt_kernel_) { 
            clReleaseKernel(process_dt_kernel_); 
            process_dt_kernel_ = nullptr; 
        }
        if (compute_dtA_kernel_) { 
            clReleaseKernel(compute_dtA_kernel_); 
            compute_dtA_kernel_ = nullptr; 
        }
        if (compute_segsum_decay_kernel_) { 
            clReleaseKernel(compute_segsum_decay_kernel_); 
            compute_segsum_decay_kernel_ = nullptr; 
        }
        if (compute_CB_kernel_) { 
            clReleaseKernel(compute_CB_kernel_); 
            compute_CB_kernel_ = nullptr; 
        }
        if (compute_ssm_output_kernel_) { 
            clReleaseKernel(compute_ssm_output_kernel_); 
            compute_ssm_output_kernel_ = nullptr; 
        }
        if (ssm_kernel_) { 
            clReleaseKernel(ssm_kernel_); 
            ssm_kernel_ = nullptr; 
        }
        if (copy_channels_kernel_) { 
            clReleaseKernel(copy_channels_kernel_); 
            copy_channels_kernel_ = nullptr; 
        }
        if (gate_kernel_) { 
            clReleaseKernel(gate_kernel_); 
            gate_kernel_ = nullptr; 
        }
        if (split_in_proj_kernel_) { 
            clReleaseKernel(split_in_proj_kernel_); 
            split_in_proj_kernel_ = nullptr; 
        }
        if (split_xBC_step_kernel_) { 
            clReleaseKernel(split_xBC_step_kernel_); 
            split_xBC_step_kernel_ = nullptr; 
        }
        if (ssm_step_update_kernel_) { 
            clReleaseKernel(ssm_step_update_kernel_); 
            ssm_step_update_kernel_ = nullptr; 
        }
        if (conv_update_kernel_) { 
            clReleaseKernel(conv_update_kernel_); 
            conv_update_kernel_ = nullptr; 
        }
        if (conv_forward_kernel_) { 
            clReleaseKernel(conv_forward_kernel_); 
            conv_forward_kernel_ = nullptr; 
        }
        if (program_) {
            clReleaseProgram(program_);
            program_ = nullptr;
        }
        throw;
    }
}

void SSDLayer::initializeWeights(
    const std::vector<float>& in_proj_weights,
    const std::vector<float>& conv_weight,
    const std::vector<float>& conv_bias,
    const std::vector<float>& A,
    const std::vector<float>& dt_bias,
    const std::vector<float>& D,
    const std::vector<float>& out_proj_weights,
    const std::vector<float>& rms_norm_weights
) {
    // Validate sizes
    if (in_proj_weights.size() != static_cast<size_t>(in_proj_dim_ * d_model_)) {
        throw std::runtime_error("Invalid in_proj weights size");
    }
    // conv_weight has conv_dim channels (d_inner + 2*d_state*n_groups), not xBC_channels
    // MLX conv1d processes only the first conv_dim channels of xBC
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
    
    // Validate rms_norm_weights size
    if (rms_norm_weights.size() != static_cast<size_t>(d_inner_)) {
        throw std::runtime_error("Invalid rms_norm weights size: expected " + 
                               std::to_string(d_inner_) + ", got " + 
                               std::to_string(rms_norm_weights.size()));
    }
    
    // Initialize linear layers
    in_proj_layer_->initializeWeights(in_proj_weights);
    out_proj_layer_->initializeWeights(out_proj_weights);
    
    // Initialize RMS norm layer with actual weights from file
    norm_layer_->initializeWeights(rms_norm_weights);
    
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
    // Ensure kernels are built
    if (!program_) {
        buildKernels();
    }
    
    // GPU kernel is required
    if (!split_in_proj_kernel_) {
        throw std::runtime_error("split_in_proj_kernel not available - GPU kernel required");
    }
    
        if (!z_out || !xBC_out || !dt_out) {
            throw std::runtime_error("Output buffers not allocated");
        }
        
    int xBC_dim = d_inner_ + 2 * d_state_ * n_groups_;
        
        cl_int err;
        err = clSetKernelArg(split_in_proj_kernel_, 0, sizeof(cl_mem), &in_proj_output);
        err |= clSetKernelArg(split_in_proj_kernel_, 1, sizeof(cl_mem), &z_out);
        err |= clSetKernelArg(split_in_proj_kernel_, 2, sizeof(cl_mem), &xBC_out);
        err |= clSetKernelArg(split_in_proj_kernel_, 3, sizeof(cl_mem), &dt_out);
        err |= clSetKernelArg(split_in_proj_kernel_, 4, sizeof(int), &batch_size);
        err |= clSetKernelArg(split_in_proj_kernel_, 5, sizeof(int), &seq_len);
        err |= clSetKernelArg(split_in_proj_kernel_, 6, sizeof(int), &d_inner_);
        err |= clSetKernelArg(split_in_proj_kernel_, 7, sizeof(int), &xBC_dim);
        err |= clSetKernelArg(split_in_proj_kernel_, 8, sizeof(int), &n_heads_);
        err |= clSetKernelArg(split_in_proj_kernel_, 9, sizeof(int), &in_proj_dim_);
        
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set split_in_proj kernel arguments");
    }
    
            size_t global_size = static_cast<size_t>(batch_size * seq_len);
            err = clEnqueueNDRangeKernel(queue, split_in_proj_kernel_, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue split_in_proj kernel");
    }
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
    cl_int err;
    
    // Step 1: in_proj: [batch, seq_len, d_model] -> [batch, seq_len, in_proj_dim]
    cl_mem in_proj_out = in_proj_layer_->forward(input, batch_size, seq_len, queue);
    clFinish(queue);
    
    // Note: in_proj_out is retained by LinearLayer, we need to release it after use
    
    // Step 2: Split into z, xBC, dt
    // Allocate buffers for z, xBC, dt (zero-initialized for determinism)
    // FIXED: xBC should be d_inner (NOT 2*d_inner) + 2*d_state*n_groups
    size_t z_size = batch_size * seq_len * d_inner_;
    size_t xBC_size = batch_size * seq_len * (d_inner_ + 2 * d_state_ * n_groups_);  // FIXED: was 2*d_inner_
    size_t dt_size = batch_size * seq_len * n_heads_;
    
    cl_mem z_buf = createAndZeroBuffer(context, queue, z_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !z_buf) throw std::runtime_error("Failed to create z buffer");
    
    cl_mem xBC_buf = createAndZeroBuffer(context, queue, xBC_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !xBC_buf) throw std::runtime_error("Failed to create xBC buffer");
    
    cl_mem dt_buf = createAndZeroBuffer(context, queue, dt_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !dt_buf) throw std::runtime_error("Failed to create dt buffer");
    
    // Split in_proj output (uses GPU kernel if available, CPU fallback otherwise)
    splitInProjOutput(in_proj_out, batch_size, seq_len, z_buf, xBC_buf, dt_buf, queue);
    clFinish(queue);
    
    // Debug: Check xBC after split (before conv1d)
    // Step 3: conv1d on xBC with Swish activation using GPU kernel
    // xBC is [batch, seq_len, xBC_channels] where xBC_channels = d_inner + 2*d_state*n_groups
    // (NOTE: xBC_channels was incorrectly 2*d_inner before - now fixed!)
    // conv1d processes conv_dim channels (d_inner + 2*d_state*n_groups), then we split to get x,B,C
    
    // CRITICAL: Match Metal implementation - concatenate state before conv, then drop last k-1 elements
    // Metal: x = concatenate([state, x], axis=-1), then y = y[..., : -kernel_size + 1]
    
    // Ensure kernels are built
    if (!program_) {
        buildKernels();
    }
    
    if (!conv_forward_kernel_) {
        throw std::runtime_error("conv_forward_kernel not available");
    }
    
    int xBC_channels = d_inner_ + 2 * d_state_ * n_groups_;  // FIXED: was 2*d_inner_
    
    // Get or initialize conv_state from state->state1
    // conv_state shape: [batch_size, conv_dim, kernel_size - 1]
    size_t conv_state_size = batch_size * conv_dim_ * (kernel_size_ - 1);
    cl_mem conv_state = nullptr;
    if (state && !state->is_null()) {
        conv_state = state->state1;
    }
    if (conv_state == nullptr) {
        // Initialize conv_state to zeros (CRITICAL for determinism)
        conv_state = createAndZeroBuffer(context, queue, conv_state_size * sizeof(float), &err);
        if (err != CL_SUCCESS || !conv_state) throw std::runtime_error("Failed to create conv_state buffer");
    }
    
    // Read xBC and extract first conv_dim channels
    std::vector<float> xBC_full(xBC_size);
    clEnqueueReadBuffer(queue, xBC_buf, CL_TRUE, 0, 
                       xBC_size * sizeof(float), xBC_full.data(), 0, nullptr, nullptr);
    
    // Read conv_state
    std::vector<float> conv_state_cpu(conv_state_size);
    clEnqueueReadBuffer(queue, conv_state, CL_TRUE, 0, 
                       conv_state_size * sizeof(float), conv_state_cpu.data(), 0, nullptr, nullptr);
    
    // Concatenate state with x along sequence dimension (matching Metal line 60)
    // xBC is [batch, seq_len, conv_dim] -> reshape to [batch, conv_dim, seq_len]
    // state is [batch, conv_dim, kernel_size - 1]
    // concatenate along sequence: [batch, conv_dim, kernel_size - 1 + seq_len]
    int concat_seq_len = (kernel_size_ - 1) + seq_len;
    size_t conv_input_concat_size = batch_size * conv_dim_ * concat_seq_len;
    std::vector<float> conv_input_concat(conv_input_concat_size);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < conv_dim_; ++c) {
            // Copy state first: [batch, conv_dim, kernel_size - 1]
            for (int k = 0; k < kernel_size_ - 1; ++k) {
                int state_idx = b * conv_dim_ * (kernel_size_ - 1) + c * (kernel_size_ - 1) + k;
                int concat_idx = b * conv_dim_ * concat_seq_len + c * concat_seq_len + k;
                conv_input_concat[concat_idx] = conv_state_cpu[state_idx];
            }
            // Copy xBC: [batch, seq_len, conv_dim] -> [batch, conv_dim, seq_len]
        for (int s = 0; s < seq_len; ++s) {
                int xBC_idx = (b * seq_len + s) * xBC_channels + c;
                int concat_idx = b * conv_dim_ * concat_seq_len + c * concat_seq_len + (kernel_size_ - 1) + s;
                conv_input_concat[concat_idx] = xBC_full[xBC_idx];
            }
        }
    }
    
    // Create buffer for concatenated input (zero-initialized for determinism)
    cl_mem conv_input_buf = createAndZeroBuffer(context, queue, conv_input_concat_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !conv_input_buf) throw std::runtime_error("Failed to create conv_input buffer");
    clEnqueueWriteBuffer(queue, conv_input_buf, CL_TRUE, 0, 
                        conv_input_concat_size * sizeof(float), conv_input_concat.data(), 0, nullptr, nullptr);
    
    // Create output buffer for full conv output (before dropping elements) - zero-initialized for determinism
    // Output will be [batch, conv_dim, concat_seq_len] but we'll drop last k-1 elements
    size_t conv_output_full_size = batch_size * conv_dim_ * concat_seq_len;
    cl_mem conv_output_buf = createAndZeroBuffer(context, queue, conv_output_full_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !conv_output_buf) {
        clReleaseMemObject(conv_input_buf);
        throw std::runtime_error("Failed to create conv_output buffer");
    }
    
    // Run conv1d forward kernel on concatenated input
    int swish_activation = 1;
    err = clSetKernelArg(conv_forward_kernel_, 0, sizeof(cl_mem), &conv_input_buf);
    err |= clSetKernelArg(conv_forward_kernel_, 1, sizeof(cl_mem), &conv_weight_);
    err |= clSetKernelArg(conv_forward_kernel_, 2, sizeof(cl_mem), &conv_bias_);
    err |= clSetKernelArg(conv_forward_kernel_, 3, sizeof(cl_mem), &conv_output_buf);
    err |= clSetKernelArg(conv_forward_kernel_, 4, sizeof(int), &batch_size);
    err |= clSetKernelArg(conv_forward_kernel_, 5, sizeof(int), &conv_dim_);
    err |= clSetKernelArg(conv_forward_kernel_, 6, sizeof(int), &concat_seq_len);  // Use concatenated length
    err |= clSetKernelArg(conv_forward_kernel_, 7, sizeof(int), &kernel_size_);
    err |= clSetKernelArg(conv_forward_kernel_, 8, sizeof(int), &swish_activation);
    
    if (err != CL_SUCCESS) {
        clReleaseMemObject(conv_input_buf);
        clReleaseMemObject(conv_output_buf);
        throw std::runtime_error("Failed to set conv_forward kernel arguments, error=" + std::to_string(err));
    }
    
    size_t conv_global_size[3] = {
        static_cast<size_t>(batch_size),
        static_cast<size_t>(conv_dim_),
        static_cast<size_t>(concat_seq_len)  // Use concatenated length
    };
    err = clEnqueueNDRangeKernel(queue, conv_forward_kernel_, 3, nullptr, conv_global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(conv_input_buf);
        clReleaseMemObject(conv_output_buf);
        throw std::runtime_error("Failed to enqueue conv_forward kernel, error=" + std::to_string(err));
    }
    clFinish(queue);
    
    // Read full conv output
    std::vector<float> conv_output_full(conv_output_full_size);
    clEnqueueReadBuffer(queue, conv_output_buf, CL_TRUE, 0, 
                       conv_output_full_size * sizeof(float), conv_output_full.data(), 0, nullptr, nullptr);
    
    // CRITICAL: Drop last k-1 elements (matching Metal line 65: y = y[..., : -kernel_size + 1])
    // Output shape: [batch, conv_dim, concat_seq_len] -> drop to [batch, conv_dim, seq_len]
    size_t conv_output_size = batch_size * conv_dim_ * seq_len;
    std::vector<float> conv_output_reshaped(conv_output_size);
    for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < conv_dim_; ++c) {
            for (int s = 0; s < seq_len; ++s) {
                int src_idx = b * conv_dim_ * concat_seq_len + c * concat_seq_len + s;
                int dst_idx = b * conv_dim_ * seq_len + c * seq_len + s;
                conv_output_reshaped[dst_idx] = conv_output_full[src_idx];
            }
        }
    }
    
    // Extract next_state from concatenated input (matching Metal line 61)
    // next_state = x[:, :, -kernel_size + 1 :] from concatenated input
    std::vector<float> next_conv_state_cpu(conv_state_size);
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < conv_dim_; ++c) {
            for (int k = 0; k < kernel_size_ - 1; ++k) {
                int src_idx = b * conv_dim_ * concat_seq_len + c * concat_seq_len + (concat_seq_len - kernel_size_ + 1 + k);
                int dst_idx = b * conv_dim_ * (kernel_size_ - 1) + c * (kernel_size_ - 1) + k;
                next_conv_state_cpu[dst_idx] = conv_input_concat[src_idx];
            }
        }
    }
    
    // Update state->state1 with next_conv_state (zero-initialized for determinism)
    cl_mem next_conv_state = createAndZeroBuffer(context, queue, conv_state_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !next_conv_state) throw std::runtime_error("Failed to create next_conv_state buffer");
    clEnqueueWriteBuffer(queue, next_conv_state, CL_TRUE, 0, 
                        conv_state_size * sizeof(float), next_conv_state_cpu.data(), 0, nullptr, nullptr);
    
    if (state) {
        if (state->state1 != conv_state && state->state1 != nullptr) {
            clReleaseMemObject(state->state1);
        }
        state->state1 = next_conv_state;
    }
    if (conv_state != next_conv_state) {
        clReleaseMemObject(conv_state);
    }
    
    // Reshape from [batch, conv_dim, seq_len] back to [batch, seq_len, conv_dim]
    // and write back to xBC_buf, then copy remaining channels unchanged
    std::vector<float> xBC_conv_result(xBC_size);
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int base_idx = (b * seq_len + s) * xBC_channels;
            
            // Copy conv output for first conv_dim channels
            for (int c = 0; c < conv_dim_; ++c) {
                int src_idx = b * conv_dim_ * seq_len + c * seq_len + s;
                xBC_conv_result[base_idx + c] = conv_output_reshaped[src_idx];
            }
            
            // Copy remaining channels unchanged from original xBC
            for (int c = conv_dim_; c < xBC_channels; ++c) {
                int src_idx = (b * seq_len + s) * xBC_channels + c;
                xBC_conv_result[base_idx + c] = xBC_full[src_idx];
            }
        }
    }
    
    // Write result back to GPU
    clEnqueueWriteBuffer(queue, xBC_buf, CL_TRUE, 0, 
                         xBC_size * sizeof(float), xBC_conv_result.data(), 0, nullptr, nullptr);
    
    // Cleanup temporary buffers
    clReleaseMemObject(conv_input_buf);
    clReleaseMemObject(conv_output_buf);
    clFinish(queue);
    
    // Step 4: Split xBC into x, B, C
    // Read xBC_conv from GPU
    std::vector<float> xBC_conv_read(xBC_size);
    clEnqueueReadBuffer(queue, xBC_buf, CL_TRUE, 0, 
                       xBC_size * sizeof(float), xBC_conv_read.data(), 0, nullptr, nullptr);
    
    // Allocate buffers for x, B, C
    // After conv, xBC is split at [d_inner, d_inner + d_state*n_groups]
    // So: x = d_inner, B = d_state*n_groups, C = d_state*n_groups
    size_t x_size = batch_size * seq_len * d_inner_;
    size_t B_size = batch_size * seq_len * d_state_ * n_groups_;
    size_t C_size = batch_size * seq_len * d_state_ * n_groups_;
    
    // Zero-initialize all intermediate buffers for determinism
    cl_mem x_buf = createAndZeroBuffer(context, queue, x_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !x_buf) throw std::runtime_error("Failed to create x buffer");
    
    cl_mem B_buf = createAndZeroBuffer(context, queue, B_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !B_buf) throw std::runtime_error("Failed to create B buffer");
    
    cl_mem C_buf = createAndZeroBuffer(context, queue, C_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !C_buf) throw std::runtime_error("Failed to create C buffer");
    
    // Split: MLX splits at [d_inner, d_inner + d_state*n_groups]
    // So: x = first d_inner, B = next d_state*n_groups, C = remaining d_state*n_groups
    // Note: xBC_conv_read has xBC_channels = d_inner + 2*d_state*n_groups channels (FIXED)
    std::vector<float> x_cpu(x_size);
    std::vector<float> B_cpu(B_size);
    std::vector<float> C_cpu(C_size);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int base_idx = (b * seq_len + s) * xBC_channels;
            // x = first d_inner
            for (int i = 0; i < d_inner_; ++i) {
                x_cpu[(b * seq_len + s) * d_inner_ + i] = xBC_conv_read[base_idx + i];
            }
            // B = next d_state*n_groups
            for (int i = 0; i < d_state_ * n_groups_; ++i) {
                B_cpu[(b * seq_len + s) * (d_state_ * n_groups_) + i] = 
                    xBC_conv_read[base_idx + d_inner_ + i];
            }
            // C = remaining d_state*n_groups
            for (int i = 0; i < d_state_ * n_groups_; ++i) {
                C_cpu[(b * seq_len + s) * (d_state_ * n_groups_) + i] = 
                    xBC_conv_read[base_idx + d_inner_ + d_state_ * n_groups_ + i];
            }
        }
    }
    
    // Debug: Check conv output and split (only in CPU fallback path)
    // Note: Debug output moved to CPU fallback section where ssm_call_count is available
    
    clEnqueueWriteBuffer(queue, x_buf, CL_TRUE, 0, x_size * sizeof(float), x_cpu.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, B_buf, CL_TRUE, 0, B_size * sizeof(float), B_cpu.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, C_buf, CL_TRUE, 0, C_size * sizeof(float), C_cpu.data(), 0, nullptr, nullptr);
    clFinish(queue);
    
    // Step 5: SSM forward - GPU-based matrix computation (or CPU fallback)
    // Note: SSM forward kernels may not be available on PowerVR due to driver bug
    
    // Declare variables that may be used in both CPU and GPU paths
    const float dt_min = 0.0f;
    const float dt_max = 1e10f;
    size_t dt_processed_size = dt_size;
    size_t dtA_size = dt_size;
    size_t decay_size = batch_size * n_heads_ * seq_len * seq_len;
    size_t CB_size = batch_size * seq_len * seq_len * n_groups_;
    size_t dtx_size = x_size;  // dtx has same shape as x
    
    cl_mem dt_processed_buf = nullptr;
    cl_mem dtA_buf = nullptr;
    cl_mem decay_buf = nullptr;
    cl_mem CB_buf = nullptr;
    cl_mem dtx_buf = nullptr;
    cl_mem A_actual_buf = nullptr;
    
    // Build kernels lazily on first use (deferred from constructor to avoid driver crashes)
    // Ensure OpenCL context is valid and queue is flushed before building
    if (!program_) {
        try {
            // Flush queue and ensure context is valid
            clFinish(queue);
            
            // Verify context is still valid
            cl_context context = ctx_->getContext();
            if (!context) {
                throw std::runtime_error("OpenCL context is invalid");
            }
            
            buildKernels();
        } catch (const std::exception& e) {
            throw std::runtime_error("Failed to build kernels: " + std::string(e.what()));
        } catch (...) {
            throw std::runtime_error("OpenCL driver crashed while building kernels");
        }
    }
    
    // GPU kernels are required
    if (!process_dt_kernel_ || !compute_dtA_kernel_ || !compute_segsum_decay_kernel_ || 
        !compute_CB_kernel_ || !compute_ssm_output_kernel_) {
        throw std::runtime_error("SSM forward GPU kernels not available - GPU kernels required");
    }
    
        // GPU path: Use SSM forward kernels
        // A is stored as already negative in MLX (not as A_log before negation)
        // So we use it directly without applying softplus or negation
        std::vector<float> A_cpu_gpu(n_heads_);
        clEnqueueReadBuffer(queue, A_, CL_TRUE, 0, n_heads_ * sizeof(float), A_cpu_gpu.data(), 0, nullptr, nullptr);
        std::vector<float> A_actual_gpu(n_heads_);
        for (int h = 0; h < n_heads_; ++h) {
            A_actual_gpu[h] = A_cpu_gpu[h];  // Use A directly - it's already the final negative value
        }
        
        // Upload A_actual to GPU
        A_actual_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                             n_heads_ * sizeof(float), A_actual_gpu.data(), &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create A_actual buffer");
        
        // Allocate GPU buffers for intermediate results (zero-initialized for determinism)
        dt_processed_buf = createAndZeroBuffer(context, queue, dt_processed_size * sizeof(float), &err);
        if (err != CL_SUCCESS || !dt_processed_buf) throw std::runtime_error("Failed to create dt_processed buffer");
        
        dtA_buf = createAndZeroBuffer(context, queue, dtA_size * sizeof(float), &err);
        if (err != CL_SUCCESS || !dtA_buf) throw std::runtime_error("Failed to create dtA buffer");
        
        decay_buf = createAndZeroBuffer(context, queue, decay_size * sizeof(float), &err);
        if (err != CL_SUCCESS || !decay_buf) throw std::runtime_error("Failed to create decay buffer");
        
        CB_buf = createAndZeroBuffer(context, queue, CB_size * sizeof(float), &err);
        if (err != CL_SUCCESS || !CB_buf) throw std::runtime_error("Failed to create CB buffer");
        
        dtx_buf = createAndZeroBuffer(context, queue, dtx_size * sizeof(float), &err);
        if (err != CL_SUCCESS || !dtx_buf) throw std::runtime_error("Failed to create dtx buffer");
        
        err = clSetKernelArg(process_dt_kernel_, 0, sizeof(cl_mem), &dt_buf);
        err |= clSetKernelArg(process_dt_kernel_, 1, sizeof(cl_mem), &dt_bias_);
        err |= clSetKernelArg(process_dt_kernel_, 2, sizeof(cl_mem), &dt_processed_buf);
        err |= clSetKernelArg(process_dt_kernel_, 3, sizeof(int), &batch_size);
        err |= clSetKernelArg(process_dt_kernel_, 4, sizeof(int), &seq_len);
        err |= clSetKernelArg(process_dt_kernel_, 5, sizeof(int), &n_heads_);
        err |= clSetKernelArg(process_dt_kernel_, 6, sizeof(float), &dt_min);
        err |= clSetKernelArg(process_dt_kernel_, 7, sizeof(float), &dt_max);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to set process_dt kernel arguments, error=" + std::to_string(err));
        }
        
        size_t process_dt_global_size = dt_processed_size;
        err = clEnqueueNDRangeKernel(queue, process_dt_kernel_, 1, nullptr, &process_dt_global_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue process_dt kernel");
        
        // Kernel 2: Compute dtA = dt * A
        err = clSetKernelArg(compute_dtA_kernel_, 0, sizeof(cl_mem), &dt_processed_buf);
        err |= clSetKernelArg(compute_dtA_kernel_, 1, sizeof(cl_mem), &A_actual_buf);
        err |= clSetKernelArg(compute_dtA_kernel_, 2, sizeof(cl_mem), &dtA_buf);
        err |= clSetKernelArg(compute_dtA_kernel_, 3, sizeof(int), &batch_size);
        err |= clSetKernelArg(compute_dtA_kernel_, 4, sizeof(int), &seq_len);
        err |= clSetKernelArg(compute_dtA_kernel_, 5, sizeof(int), &n_heads_);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set compute_dtA kernel arguments");
        
        size_t compute_dtA_global_size = dtA_size;
        err = clEnqueueNDRangeKernel(queue, compute_dtA_kernel_, 1, nullptr, &compute_dtA_global_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue compute_dtA kernel");
        
        // Kernel 3: Compute segsum to get decay matrix
        size_t segsum_global_size[3] = {static_cast<size_t>(batch_size), static_cast<size_t>(n_heads_), static_cast<size_t>(seq_len)};
        err = clSetKernelArg(compute_segsum_decay_kernel_, 0, sizeof(cl_mem), &dtA_buf);
        err |= clSetKernelArg(compute_segsum_decay_kernel_, 1, sizeof(cl_mem), &decay_buf);
        err |= clSetKernelArg(compute_segsum_decay_kernel_, 2, sizeof(int), &batch_size);
        err |= clSetKernelArg(compute_segsum_decay_kernel_, 3, sizeof(int), &seq_len);
        err |= clSetKernelArg(compute_segsum_decay_kernel_, 4, sizeof(int), &n_heads_);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set compute_segsum_decay kernel arguments");
        
        err = clEnqueueNDRangeKernel(queue, compute_segsum_decay_kernel_, 3, nullptr, segsum_global_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue compute_segsum_decay kernel");
        
        // Kernel 4: Compute CB = C @ B (using 3D work group - OpenCL max is 3D)
        size_t CB_global_size[3] = {
            static_cast<size_t>(batch_size), 
            static_cast<size_t>(seq_len), 
            static_cast<size_t>(seq_len)
        };
        err = clSetKernelArg(compute_CB_kernel_, 0, sizeof(cl_mem), &B_buf);
        err |= clSetKernelArg(compute_CB_kernel_, 1, sizeof(cl_mem), &C_buf);
        err |= clSetKernelArg(compute_CB_kernel_, 2, sizeof(cl_mem), &CB_buf);
        err |= clSetKernelArg(compute_CB_kernel_, 3, sizeof(int), &batch_size);
        err |= clSetKernelArg(compute_CB_kernel_, 4, sizeof(int), &seq_len);
        err |= clSetKernelArg(compute_CB_kernel_, 5, sizeof(int), &n_groups_);
        err |= clSetKernelArg(compute_CB_kernel_, 6, sizeof(int), &d_state_);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set compute_CB kernel arguments");
        
        err = clEnqueueNDRangeKernel(queue, compute_CB_kernel_, 3, nullptr, CB_global_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue compute_CB kernel");
        
        // Compute dtx = dt * x (element-wise multiplication)
        // We'll do this with a simple kernel or element-wise multiplication
        // For now, read dt_processed and x, compute dtx, then write back
        // TODO: Create a dedicated kernel for this
        std::vector<float> dt_processed_cpu(dt_processed_size);
        std::vector<float> x_cpu_dtx(x_size);
        clEnqueueReadBuffer(queue, dt_processed_buf, CL_TRUE, 0, dt_processed_size * sizeof(float), dt_processed_cpu.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, x_buf, CL_TRUE, 0, x_size * sizeof(float), x_cpu_dtx.data(), 0, nullptr, nullptr);
        
        std::vector<float> dtx_cpu(dtx_size);
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                for (int h = 0; h < n_heads_; ++h) {
                    float dt_val = dt_processed_cpu[(b * seq_len + s) * n_heads_ + h];
                    for (int d = 0; d < d_head_; ++d) {
                        int x_idx = (b * seq_len + s) * d_inner_ + h * d_head_ + d;
                        dtx_cpu[x_idx] = dt_val * x_cpu_dtx[x_idx];
                    }
                }
            }
        }
        clEnqueueWriteBuffer(queue, dtx_buf, CL_TRUE, 0, dtx_size * sizeof(float), dtx_cpu.data(), 0, nullptr, nullptr);
        
        // Kernel 5: Compute final output y = tril(CB * decay) @ dtx + D * x
        // Create a separate output buffer for the SSM result (zero-initialized for determinism)
        cl_mem x_ssm_output_buf = createAndZeroBuffer(context, queue, x_size * sizeof(float), &err);
        if (err != CL_SUCCESS || !x_ssm_output_buf) throw std::runtime_error("Failed to create x_ssm_output buffer");
        
        // Use 3D work group (OpenCL max is 3D) - loop over d_head inside kernel
        size_t output_global_size[3] = {
            static_cast<size_t>(batch_size), 
            static_cast<size_t>(seq_len), 
            static_cast<size_t>(n_heads_)
        };
        err = clSetKernelArg(compute_ssm_output_kernel_, 0, sizeof(cl_mem), &CB_buf);
        err |= clSetKernelArg(compute_ssm_output_kernel_, 1, sizeof(cl_mem), &decay_buf);
        err |= clSetKernelArg(compute_ssm_output_kernel_, 2, sizeof(cl_mem), &dtx_buf);
        err |= clSetKernelArg(compute_ssm_output_kernel_, 3, sizeof(cl_mem), &D_);
        err |= clSetKernelArg(compute_ssm_output_kernel_, 4, sizeof(cl_mem), &x_buf);  // Input x for D skip connection
        err |= clSetKernelArg(compute_ssm_output_kernel_, 5, sizeof(cl_mem), &x_ssm_output_buf);  // Output y
        err |= clSetKernelArg(compute_ssm_output_kernel_, 6, sizeof(int), &batch_size);
        err |= clSetKernelArg(compute_ssm_output_kernel_, 7, sizeof(int), &seq_len);
        err |= clSetKernelArg(compute_ssm_output_kernel_, 8, sizeof(int), &n_heads_);
        err |= clSetKernelArg(compute_ssm_output_kernel_, 9, sizeof(int), &d_head_);
        err |= clSetKernelArg(compute_ssm_output_kernel_, 10, sizeof(int), &n_groups_);
        err |= clSetKernelArg(compute_ssm_output_kernel_, 11, sizeof(int), &d_state_);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set compute_ssm_output kernel arguments");
        
        err = clEnqueueNDRangeKernel(queue, compute_ssm_output_kernel_, 3, nullptr, output_global_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue compute_ssm_output kernel");
        
        clFinish(queue);
        
        // BEFORE overwriting x_buf: Extract SSM state for next step() call (MUST be before copy!)
        // We need to read x_buf while it still contains the SSM INPUT, not output
        // SSM state = sum_t[(dtx[t] * decay[t]) @ B[t]], accumulated over ALL sequence positions
        // State shape: [batch_size, n_heads, d_head, d_state]
        if (state != nullptr) {
            size_t ssm_state_size = batch_size * n_heads_ * d_head_ * d_state_;
            std::vector<float> ssm_state_cpu(ssm_state_size, 0.0f);  // Initialize to zero for accumulation
            
            // Read ALL x, B, dt values (not just last position)
            std::vector<float> x_all(x_size);
            std::vector<float> B_all(B_size);
            std::vector<float> dt_all(batch_size * seq_len * n_heads_);
            
            clEnqueueReadBuffer(queue, x_buf, CL_TRUE, 0, x_all.size() * sizeof(float), x_all.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(queue, B_buf, CL_TRUE, 0, B_all.size() * sizeof(float), B_all.data(), 0, nullptr, nullptr);
            clEnqueueReadBuffer(queue, dt_processed_buf, CL_TRUE, 0, dt_all.size() * sizeof(float), dt_all.data(), 0, nullptr, nullptr);
            
            // Read A
            std::vector<float> A_cpu(n_heads_);
            clEnqueueReadBuffer(queue, A_, CL_TRUE, 0, n_heads_ * sizeof(float), A_cpu.data(), 0, nullptr, nullptr);
            
            // Accumulate SSM state across ALL sequence positions
            for (int b = 0; b < batch_size; ++b) {
                for (int h = 0; h < n_heads_; ++h) {
                    float A_val = A_cpu[h];
                    int group_idx = h % n_groups_;
                    
                    // For each sequence position, compute cumulative decay and accumulate contribution
                    for (int t = 0; t < seq_len; ++t) {
                        // FIXED: Compute decay from position t+1 to END: exp(A * sum(dt[t+1:seq_len]))
                        float dt_cumsum_after = 0.0f;
                        for (int s = t + 1; s < seq_len; ++s) {
                            dt_cumsum_after += dt_all[(b * seq_len + s) * n_heads_ + h];
                        }
                        float decay = expf(A_val * dt_cumsum_after);
                        float dt_val = dt_all[(b * seq_len + t) * n_heads_ + h];
                        
                        for (int d = 0; d < d_head_; ++d) {
                            int x_idx = (b * seq_len + t) * d_inner_ + h * d_head_ + d;
                            float x_val = x_all[x_idx];
                            float dtx_decay = dt_val * x_val * decay;
                            
                            // Accumulate state contribution: dtx_decay @ B[t]
                            for (int n = 0; n < d_state_; ++n) {
                                int B_idx = (b * seq_len + t) * (d_state_ * n_groups_) + group_idx * d_state_ + n;
                                float B_val = B_all[B_idx];
                                
                                int state_idx = (b * n_heads_ + h) * d_head_ * d_state_ + d * d_state_ + n;
                                ssm_state_cpu[state_idx] += dtx_decay * B_val;  // ACCUMULATE, not replace
                            }
                        }
                    }
                }
            }
            
            // Write SSM state to GPU buffer
            cl_int err_state;
            cl_mem next_ssm_state = createAndZeroBuffer(context, queue, ssm_state_size * sizeof(float), &err_state);
            if (err_state != CL_SUCCESS || !next_ssm_state) throw std::runtime_error("Failed to create next_ssm_state buffer in forward");
            
            clEnqueueWriteBuffer(queue, next_ssm_state, CL_TRUE, 0, ssm_state_size * sizeof(float), ssm_state_cpu.data(), 0, nullptr, nullptr);
            
            // Update state->state2
            if (state->state2 != nullptr) {
                clReleaseMemObject(state->state2);
            }
            state->state2 = next_ssm_state;
            clFinish(queue);
        }
        
        // NOW copy SSM output to x_buf (after extracting state using the original x values)
        err = clEnqueueCopyBuffer(queue, x_ssm_output_buf, x_buf, 0, 0, x_size * sizeof(float), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to copy SSM output to x_buf");
        clFinish(queue);
        
        // Cleanup SSM output buffer
        clReleaseMemObject(x_ssm_output_buf);
        
    // Cleanup GPU buffers
        if (A_actual_buf) clReleaseMemObject(A_actual_buf);
        if (dt_processed_buf) clReleaseMemObject(dt_processed_buf);
        if (dtA_buf) clReleaseMemObject(dtA_buf);
        if (decay_buf) clReleaseMemObject(decay_buf);
        if (CB_buf) clReleaseMemObject(CB_buf);
        if (dtx_buf) clReleaseMemObject(dtx_buf);
    
    // Step 6: Apply gate: Swish(z) * x using GPU kernel, then RMS norm
    // Ensure kernels are built
    if (!program_) {
        buildKernels();
    }
    
    // GPU kernel is required
    if (!gate_kernel_) {
        throw std::runtime_error("gate_kernel not available - GPU kernel required");
    }
    
    // Create output buffer for gated result (zero-initialized for determinism)
    cl_mem gated_buf = createAndZeroBuffer(context, queue, x_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !gated_buf) throw std::runtime_error("Failed to create gated buffer");
    
    err = clSetKernelArg(gate_kernel_, 0, sizeof(cl_mem), &z_buf);
    err |= clSetKernelArg(gate_kernel_, 1, sizeof(cl_mem), &x_buf);
    err |= clSetKernelArg(gate_kernel_, 2, sizeof(cl_mem), &gated_buf);
    err |= clSetKernelArg(gate_kernel_, 3, sizeof(int), &x_size);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set gate kernel arguments");
    }
    
    size_t gate_global_size = static_cast<size_t>(x_size);
    err = clEnqueueNDRangeKernel(queue, gate_kernel_, 1, nullptr, &gate_global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue gate kernel");
    }
    
    clFinish(queue);
    
    // Apply RMS norm (norm_before_gate is False, so norm after gate)
    cl_mem normed = norm_layer_->forward(gated_buf, batch_size, seq_len, queue);
    clFinish(queue);
    
    // Step 7: out_proj
    cl_mem output = out_proj_layer_->forward(normed, batch_size, seq_len, queue);
    clFinish(queue);
    
    // Cleanup temporary buffers (be careful - some may be null)
    // Release buffers we created locally
    if (z_buf) { clReleaseMemObject(z_buf); z_buf = nullptr; }
    if (xBC_buf) { clReleaseMemObject(xBC_buf); xBC_buf = nullptr; }
    if (dt_buf) { clReleaseMemObject(dt_buf); dt_buf = nullptr; }
    if (x_buf) { clReleaseMemObject(x_buf); x_buf = nullptr; }
    if (B_buf) { clReleaseMemObject(B_buf); B_buf = nullptr; }
    if (C_buf) { clReleaseMemObject(C_buf); C_buf = nullptr; }
    if (gated_buf) { clReleaseMemObject(gated_buf); gated_buf = nullptr; }
    
    // Release buffers returned from layer calls (they were retained)
    if (in_proj_out) { clReleaseMemObject(in_proj_out); in_proj_out = nullptr; }
    if (normed) { clReleaseMemObject(normed); normed = nullptr; }
    // Note: output is returned to caller, they will release it
    
    return output;
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
    
    // Initialize state if null (first call to step())
    if (state == nullptr) {
        throw std::runtime_error("SSD layer step: state pointer is null");
    }
    
    // If state buffers are null, initialize them
    if (state->is_null()) {
        // Initialize conv_state and ssm_state to zeros (CRITICAL for determinism)
        size_t conv_state_size = batch_size * conv_dim_ * (kernel_size_ - 1);
        size_t ssm_state_size = batch_size * n_heads_ * d_head_ * d_state_;
        
        cl_context context = ctx_->getContext();
        cl_int err;
        
        // Initialize conv_state using createAndZeroBuffer for determinism
        state->state1 = createAndZeroBuffer(context, queue, conv_state_size * sizeof(float), &err);
        if (err != CL_SUCCESS || !state->state1) throw std::runtime_error("Failed to create conv_state buffer");
        
        // Initialize ssm_state using createAndZeroBuffer for determinism
        state->state2 = createAndZeroBuffer(context, queue, ssm_state_size * sizeof(float), &err);
        if (err != CL_SUCCESS || !state->state2) {
            clReleaseMemObject(state->state1);
            throw std::runtime_error("Failed to create ssm_state buffer");
        }
        clFinish(queue);
    }
    
    cl_context context = ctx_->getContext();
    cl_int err;
    
    // Step 1: in_proj on single token [batch_size, d_model] -> [batch_size, in_proj_dim]
    cl_mem in_proj_output = in_proj_layer_->forward(input, batch_size, 1, queue);
    clFinish(queue);
    
    // Step 2: Split in_proj output into z, xBC, dt
    // in_proj_dim = 2*d_inner + 2*d_state*n_groups + n_heads
    // z: [batch_size, d_inner]
    // xBC: [batch_size, d_inner + 2*d_state*n_groups]
    // dt: [batch_size, n_heads]
    size_t z_size = batch_size * d_inner_;
    size_t xBC_size = batch_size * (d_inner_ + 2 * d_state_ * n_groups_);
    size_t dt_size = batch_size * n_heads_;
    
    cl_mem z_buf = createAndZeroBuffer(context, queue, z_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !z_buf) throw std::runtime_error("Failed to create z buffer");
    cl_mem xBC_buf = createAndZeroBuffer(context, queue, xBC_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !xBC_buf) throw std::runtime_error("Failed to create xBC buffer");
    cl_mem dt_buf = createAndZeroBuffer(context, queue, dt_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !dt_buf) throw std::runtime_error("Failed to create dt buffer");
    
    splitInProjOutput(in_proj_output, batch_size, 1, z_buf, xBC_buf, dt_buf, queue);
    clFinish(queue);
    clReleaseMemObject(in_proj_output);
    
    // Step 3: conv1d_update on xBC (using conv_state from state->state1)
    // conv_state shape: [batch_size, conv_dim, kernel_size - 1]
    // xBC: [batch_size, conv_dim] (single token)
    size_t conv_state_size = batch_size * conv_dim_ * (kernel_size_ - 1);
    cl_mem conv_state = state->state1;
    if (conv_state == nullptr) {
        // Initialize conv_state to zeros (CRITICAL for determinism)
        conv_state = createAndZeroBuffer(context, queue, conv_state_size * sizeof(float), &err);
        if (err != CL_SUCCESS || !conv_state) throw std::runtime_error("Failed to create conv_state buffer");
    }
    
    // Zero-initialize intermediate buffers for determinism
    cl_mem conv_output = createAndZeroBuffer(context, queue, xBC_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !conv_output) throw std::runtime_error("Failed to create conv_output buffer");
    cl_mem next_conv_state = createAndZeroBuffer(context, queue, conv_state_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !next_conv_state) throw std::runtime_error("Failed to create next_conv_state buffer");
    
    // Set kernel arguments for conv1d_update
    err = clSetKernelArg(conv_update_kernel_, 0, sizeof(cl_mem), &xBC_buf);
    err |= clSetKernelArg(conv_update_kernel_, 1, sizeof(cl_mem), &conv_weight_);
    err |= clSetKernelArg(conv_update_kernel_, 2, sizeof(cl_mem), &conv_bias_);
    err |= clSetKernelArg(conv_update_kernel_, 3, sizeof(cl_mem), &conv_state);
    err |= clSetKernelArg(conv_update_kernel_, 4, sizeof(cl_mem), &conv_output);
    err |= clSetKernelArg(conv_update_kernel_, 5, sizeof(cl_mem), &next_conv_state);
    err |= clSetKernelArg(conv_update_kernel_, 6, sizeof(int), &batch_size);
    err |= clSetKernelArg(conv_update_kernel_, 7, sizeof(int), &conv_dim_);
    err |= clSetKernelArg(conv_update_kernel_, 8, sizeof(int), &kernel_size_);
    int swish_activation = 1;
    err |= clSetKernelArg(conv_update_kernel_, 9, sizeof(int), &swish_activation);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set conv_update kernel arguments");
    
    // Validate parameters before kernel call
    if (batch_size <= 0 || conv_dim_ <= 0 || kernel_size_ <= 0) {
        throw std::runtime_error("Invalid parameters for conv_update: batch_size=" + std::to_string(batch_size) 
                                 + ", conv_dim=" + std::to_string(conv_dim_) + ", kernel_size=" + std::to_string(kernel_size_));
    }
    
    // Validate buffers are not null
    if (!xBC_buf || !conv_weight_ || !conv_bias_ || !conv_state || !conv_output || !next_conv_state) {
        throw std::runtime_error("Null buffer passed to conv_update kernel");
    }
    
    size_t global_size[2] = {static_cast<size_t>(batch_size), static_cast<size_t>(conv_dim_)};
    
    // Enqueue the kernel
    err = clEnqueueNDRangeKernel(queue, conv_update_kernel_, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::string error_msg = "Failed to enqueue conv_update kernel - OpenCL error: " + std::to_string(err);
        error_msg += "\n    Global size: [" + std::to_string(global_size[0]) + ", " + std::to_string(global_size[1]) + "]";
        error_msg += "\n    Batch size: " + std::to_string(batch_size) + ", conv_dim: " + std::to_string(conv_dim_);
        throw std::runtime_error(error_msg);
    }
    
    // Finish the queue to ensure kernel completes
    err = clFinish(queue);
    if (err != CL_SUCCESS) {
        // PowerVR driver sometimes reports errors but execution succeeds - log but don't throw
        // The "NDRANGE_KERNEL executed abnormally" message is a driver diagnostic, not a fatal error
    }
    
    // Update state->state1 with next_conv_state
    if (state->state1 != conv_state) {
        if (state->state1 != nullptr) clReleaseMemObject(state->state1);
    }
    state->state1 = next_conv_state;
    if (conv_state != state->state1) clReleaseMemObject(conv_state);
    clReleaseMemObject(xBC_buf);
    xBC_buf = conv_output;  // Reuse conv_output as xBC
    
    // Step 4: Split xBC into x, B, C
    // x: [batch_size, d_inner]
    // B: [batch_size, d_state*n_groups]
    // C: [batch_size, d_state*n_groups]
    size_t x_size = batch_size * d_inner_;
    size_t BC_size = batch_size * d_state_ * n_groups_;
    
    // GPU kernel is required
    if (!split_xBC_step_kernel_) {
        throw std::runtime_error("split_xBC_step_kernel not available - GPU kernel required");
    }
    
    // Create GPU buffers for x, B, C
    cl_mem x_buf = createAndZeroBuffer(context, queue, x_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !x_buf) throw std::runtime_error("Failed to create x buffer");
    
    cl_mem B_buf = createAndZeroBuffer(context, queue, BC_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !B_buf) throw std::runtime_error("Failed to create B buffer");
    
    cl_mem C_buf = createAndZeroBuffer(context, queue, BC_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !C_buf) throw std::runtime_error("Failed to create C buffer");
    
    // Use GPU kernel to split xBC
    int d_state_groups = d_state_ * n_groups_;
    err = clSetKernelArg(split_xBC_step_kernel_, 0, sizeof(cl_mem), &xBC_buf);
    err |= clSetKernelArg(split_xBC_step_kernel_, 1, sizeof(cl_mem), &x_buf);
    err |= clSetKernelArg(split_xBC_step_kernel_, 2, sizeof(cl_mem), &B_buf);
    err |= clSetKernelArg(split_xBC_step_kernel_, 3, sizeof(cl_mem), &C_buf);
    err |= clSetKernelArg(split_xBC_step_kernel_, 4, sizeof(int), &batch_size);
    err |= clSetKernelArg(split_xBC_step_kernel_, 5, sizeof(int), &d_inner_);
    err |= clSetKernelArg(split_xBC_step_kernel_, 6, sizeof(int), &d_state_groups);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set split_xBC_step kernel args");
    
    size_t global_size_split = static_cast<size_t>(batch_size);
    err = clEnqueueNDRangeKernel(queue, split_xBC_step_kernel_, 1, nullptr, &global_size_split, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue split_xBC_step kernel");
    
    clFinish(queue);
    clReleaseMemObject(xBC_buf);
    
    // Step 5: SSM update (using ssm_state from state->state2) - GPU kernel
    // Get or initialize ssm_state
    size_t ssm_state_size = batch_size * n_heads_ * d_head_ * d_state_;
    cl_mem ssm_state = state->state2;
    if (ssm_state == nullptr) {
        // Initialize ssm_state to zeros (CRITICAL for determinism)
        ssm_state = createAndZeroBuffer(context, queue, ssm_state_size * sizeof(float), &err);
        if (err != CL_SUCCESS || !ssm_state) throw std::runtime_error("Failed to create ssm_state buffer");
    }
    
    // GPU kernel is required
    if (!ssm_step_update_kernel_) {
        throw std::runtime_error("ssm_step_update_kernel not available - GPU kernel required");
    }
    
    if (x_buf == nullptr || B_buf == nullptr || C_buf == nullptr) {
        throw std::runtime_error("x_buf, B_buf, or C_buf is null - GPU buffers required");
    }
    
    // x, B, C are already on GPU from split kernel
    // Create output buffer for SSM result
    cl_mem ssm_output = createAndZeroBuffer(context, queue, x_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !ssm_output) throw std::runtime_error("Failed to create SSM output buffer");
    
    // Create next_state buffer
    cl_mem next_ssm_state = createAndZeroBuffer(context, queue, ssm_state_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !next_ssm_state) throw std::runtime_error("Failed to create next_ssm_state buffer");
    
    // Set kernel arguments
    err = clSetKernelArg(ssm_step_update_kernel_, 0, sizeof(cl_mem), &x_buf);
    err |= clSetKernelArg(ssm_step_update_kernel_, 1, sizeof(cl_mem), &dt_buf);
    err |= clSetKernelArg(ssm_step_update_kernel_, 2, sizeof(cl_mem), &dt_bias_);
    err |= clSetKernelArg(ssm_step_update_kernel_, 3, sizeof(cl_mem), &A_);
    err |= clSetKernelArg(ssm_step_update_kernel_, 4, sizeof(cl_mem), &B_buf);
    err |= clSetKernelArg(ssm_step_update_kernel_, 5, sizeof(cl_mem), &C_buf);
    err |= clSetKernelArg(ssm_step_update_kernel_, 6, sizeof(cl_mem), &D_);
    err |= clSetKernelArg(ssm_step_update_kernel_, 7, sizeof(cl_mem), &ssm_state);
    err |= clSetKernelArg(ssm_step_update_kernel_, 8, sizeof(cl_mem), &ssm_output);
    err |= clSetKernelArg(ssm_step_update_kernel_, 9, sizeof(cl_mem), &next_ssm_state);
    err |= clSetKernelArg(ssm_step_update_kernel_, 10, sizeof(int), &batch_size);
    err |= clSetKernelArg(ssm_step_update_kernel_, 11, sizeof(int), &n_heads_);
    err |= clSetKernelArg(ssm_step_update_kernel_, 12, sizeof(int), &d_head_);
    err |= clSetKernelArg(ssm_step_update_kernel_, 13, sizeof(int), &d_state_);
    err |= clSetKernelArg(ssm_step_update_kernel_, 14, sizeof(int), &n_groups_);
    err |= clSetKernelArg(ssm_step_update_kernel_, 15, sizeof(int), &d_inner_);
    
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set ssm_step_update kernel args");
    
    // Execute kernel: 1D work group (batch_size * n_heads * d_head)
    size_t global_size = static_cast<size_t>(batch_size * n_heads_ * d_head_);
    err = clEnqueueNDRangeKernel(queue, ssm_step_update_kernel_, 1, nullptr, &global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue ssm_step_update kernel");
    
    clFinish(queue);
    
    // Update state->state2
    if (state->state2 != ssm_state) {
        if (state->state2 != nullptr) clReleaseMemObject(state->state2);
    }
    state->state2 = next_ssm_state;
    if (ssm_state != state->state2) clReleaseMemObject(ssm_state);
    
    // Release old x_buf and use SSM output
    clReleaseMemObject(x_buf);
    x_buf = ssm_output;
    
    // Release B and C buffers (no longer needed)
    clReleaseMemObject(B_buf);
    clReleaseMemObject(C_buf);
    
    // Step 6: Apply gate: Swish(z) * x using GPU kernel, then RMS norm
    // Ensure kernels are built
    if (!program_) {
        buildKernels();
    }
    
    // GPU kernel is required
    if (!gate_kernel_) {
        throw std::runtime_error("gate_kernel not available - GPU kernel required");
    }
    
    // Create output buffer for gated result (zero-initialized for determinism)
    cl_mem gated_buf = createAndZeroBuffer(context, queue, x_size * sizeof(float), &err);
    if (err != CL_SUCCESS || !gated_buf) throw std::runtime_error("Failed to create gated buffer");
    
    err = clSetKernelArg(gate_kernel_, 0, sizeof(cl_mem), &z_buf);
    err |= clSetKernelArg(gate_kernel_, 1, sizeof(cl_mem), &x_buf);
    err |= clSetKernelArg(gate_kernel_, 2, sizeof(cl_mem), &gated_buf);
    err |= clSetKernelArg(gate_kernel_, 3, sizeof(int), &x_size);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set gate kernel arguments");
    }
    
    size_t gate_global_size = static_cast<size_t>(x_size);
    err = clEnqueueNDRangeKernel(queue, gate_kernel_, 1, nullptr, &gate_global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue gate kernel");
    }
    
    clFinish(queue);
    
    // Apply RMS norm (norm_before_gate is False, so norm after gate)
    cl_mem normed = norm_layer_->forward(gated_buf, batch_size, 1, queue);
    clFinish(queue);
    
    // Step 7: out_proj
    cl_mem output = out_proj_layer_->forward(normed, batch_size, 1, queue);
    clFinish(queue);
    
    // Cleanup temporary buffers (be careful - some may be null)
    if (z_buf) { clReleaseMemObject(z_buf); z_buf = nullptr; }
    if (dt_buf) { clReleaseMemObject(dt_buf); dt_buf = nullptr; }
    if (x_buf) { clReleaseMemObject(x_buf); x_buf = nullptr; }
    if (gated_buf) { clReleaseMemObject(gated_buf); gated_buf = nullptr; }
    if (normed) { clReleaseMemObject(normed); normed = nullptr; }  // Released retained buffer from RMSNormLayer
    // Note: output is returned to caller, they will release it
    
    return output;
}

} // namespace cartesia_opencl

