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
    
    int CB_idx = ((b * seq_len + s) * seq_len + t) * n_groups + g;
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
        
        int CB_idx = ((b * seq_len + s) * seq_len + t) * n_groups + group_idx;
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
    
    // Build program with just the required kernels first
    // SSM forward kernels will be added separately if needed
    std::vector<std::string> sources = {
        std::string(conv1d_cl_source), 
        std::string(ssm_update_cl_source)
    };
    std::string cache_key = ctx_mgr.generateCacheKey(sources);
    
    try {
        // Ensure queue is flushed before building program
        cl_command_queue queue = ctx_mgr.getQueue();
        if (queue) {
            clFinish(queue);
        }
        
        std::cout << "[SSDLayer] Building OpenCL program with " << sources.size() << " source(s)..." << std::flush;
        program_ = ctx_mgr.buildProgram(sources, cache_key);
        std::cout << " ✓" << std::endl;
        
        // Verify program was created
        if (!program_) {
            throw std::runtime_error("buildProgram returned null");
        }
        
        // Check build status
        cl_build_status build_status;
        cl_int err = clGetProgramBuildInfo(program_, ctx_mgr.getDevice(), CL_PROGRAM_BUILD_STATUS, 
                                          sizeof(cl_build_status), &build_status, nullptr);
        if (err == CL_SUCCESS) {
            if (build_status == CL_BUILD_SUCCESS) {
                std::cout << "[SSDLayer] Program build successful" << std::endl;
            } else {
                std::cerr << "[SSDLayer] Warning: Program build status = " << build_status << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[SSDLayer] Error building program: " << e.what() << std::endl;
        program_ = nullptr;  // Ensure it's null
        throw;
    } catch (...) {
        std::cerr << "[SSDLayer] Unknown error building program" << std::endl;
        program_ = nullptr;
        throw std::runtime_error("Unknown error building OpenCL program");
    }
    
    // Initialize kernels to null
    conv_forward_kernel_ = nullptr;
    conv_update_kernel_ = nullptr;
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
        std::cout << "[SSDLayer] Creating required kernels..." << std::flush;
        
        // Create kernels one at a time to isolate any driver crashes
        std::cout << "\n  [SSDLayer] Creating conv_forward_kernel..." << std::flush;
        conv_forward_kernel_ = ctx_mgr.getKernel(program_, "conv1d_forward_kernel");
        if (!conv_forward_kernel_) {
            throw std::runtime_error("Failed to create conv_forward_kernel (returned null)");
        }
        std::cout << " ✓" << std::flush;
        
        std::cout << "\n  [SSDLayer] Creating conv_update_kernel..." << std::flush;
        conv_update_kernel_ = ctx_mgr.getKernel(program_, "conv1d_update_kernel");
        if (!conv_update_kernel_) {
            throw std::runtime_error("Failed to create conv_update_kernel (returned null)");
        }
        std::cout << " ✓" << std::flush;
        
        std::cout << "\n  [SSDLayer] Creating ssm_kernel..." << std::flush;
        ssm_kernel_ = ctx_mgr.getKernel(program_, "ssm_update_kernel");
        if (!ssm_kernel_) {
            throw std::runtime_error("Failed to create ssm_kernel (returned null)");
        }
        std::cout << " ✓" << std::flush;
        
        std::cout << "\n[SSDLayer] Required kernels created successfully" << std::endl;
        
        // SSM forward kernels (for prefill) - build a separate program for these
        // WARNING: PowerVR OpenCL driver crashes (segfault) when building these kernels
        // This is a known driver bug. We skip them entirely to avoid crashes.
        // The model will use CPU fallback for prefill SSM computation.
        std::cout << "[SSDLayer] Skipping SSM forward kernels (PowerVR driver bug causes segfault)" << std::endl;
        std::cout << "[SSDLayer] Will use CPU fallback for prefill SSM computation" << std::endl;
        
        // DO NOT attempt to build SSM forward kernels on PowerVR - it causes driver crash
        // Uncomment below to test on other drivers, but keep commented for PowerVR
        /*
        try {
            std::vector<std::string> ssm_forward_sources = {std::string(ssm_forward_cl_source)};
            std::string ssm_forward_cache_key = ctx_mgr.generateCacheKey(ssm_forward_sources) + "_ssm_forward";
            
            std::cout << "\n  [SSDLayer] Building SSM forward program..." << std::flush;
            cl_program ssm_forward_program = ctx_mgr.buildProgram(ssm_forward_sources, ssm_forward_cache_key);
            std::cout << " ✓" << std::flush;
            
            if (ssm_forward_program) {
                std::cout << "\n  [SSDLayer] Creating process_dt_kernel..." << std::flush;
                process_dt_kernel_ = ctx_mgr.getKernel(ssm_forward_program, "process_dt_kernel");
                std::cout << " ✓" << std::flush;
                
                std::cout << "\n  [SSDLayer] Creating compute_dtA_kernel..." << std::flush;
                compute_dtA_kernel_ = ctx_mgr.getKernel(ssm_forward_program, "compute_dtA_kernel");
                std::cout << " ✓" << std::flush;
                
                std::cout << "\n  [SSDLayer] Creating compute_segsum_decay_kernel..." << std::flush;
                compute_segsum_decay_kernel_ = ctx_mgr.getKernel(ssm_forward_program, "compute_segsum_decay_kernel");
                std::cout << " ✓" << std::flush;
                
                std::cout << "\n  [SSDLayer] Creating compute_CB_kernel..." << std::flush;
                compute_CB_kernel_ = ctx_mgr.getKernel(ssm_forward_program, "compute_CB_kernel");
                std::cout << " ✓" << std::flush;
                
                std::cout << "\n  [SSDLayer] Creating compute_ssm_output_kernel..." << std::flush;
                compute_ssm_output_kernel_ = ctx_mgr.getKernel(ssm_forward_program, "compute_ssm_output_kernel");
                std::cout << " ✓" << std::flush;
                
                // Store the program for cleanup (we'll need to add a member variable for this)
                // For now, we'll just keep the kernels and release the program
                clReleaseProgram(ssm_forward_program);
                
                std::cout << "\n[SSDLayer] All SSM forward kernels created successfully" << std::endl;
            } else {
                std::cerr << "\n[SSDLayer] Warning: SSM forward program is null" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "\n[SSDLayer] Warning: Failed to create SSM forward kernels (will use CPU fallback): " << e.what() << std::endl;
            // Keep kernels as nullptr - forward() will check and use CPU fallback
        } catch (...) {
            std::cerr << "\n[SSDLayer] Warning: Unknown error creating SSM forward kernels (possible driver crash)" << std::endl;
            // Keep kernels as nullptr
        }
        */
    } catch (const std::exception& e) {
        std::cerr << "[SSDLayer] Error creating kernels: " << e.what() << std::endl;
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
    
    // Initialize RMS norm layer with ones (default initialization)
    std::vector<float> norm_weights(d_inner_, 1.0f);
    norm_layer_->initializeWeights(norm_weights);
    
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
    cl_int err;
    
    // Step 1: in_proj: [batch, seq_len, d_model] -> [batch, seq_len, in_proj_dim]
    cl_mem in_proj_out = in_proj_layer_->forward(input, batch_size, seq_len, queue);
    clFinish(queue);
    
    // Note: in_proj_out is retained by LinearLayer, we need to release it after use
    
    // Step 2: Split into z, xBC, dt
    // Allocate buffers for z, xBC, dt
    size_t z_size = batch_size * seq_len * d_inner_;
    size_t xBC_size = batch_size * seq_len * (2 * d_inner_ + 2 * d_state_ * n_groups_);
    size_t dt_size = batch_size * seq_len * n_heads_;
    
    cl_mem z_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, z_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create z buffer");
    
    cl_mem xBC_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, xBC_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create xBC buffer");
    
    cl_mem dt_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dt_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create dt buffer");
    
    // Split in_proj output
    splitInProjOutput(in_proj_out, batch_size, seq_len, z_buf, xBC_buf, dt_buf, queue);
    clFinish(queue);
    
    // Step 3: conv1d on xBC with Swish activation
    // xBC is [batch, seq_len, conv_dim], need to reshape to [batch, conv_dim, seq_len] for conv
    // Use CPU fallback for now
    // Read xBC from the already-split buffer
    std::vector<float> xBC_cpu(xBC_size);
    clEnqueueReadBuffer(queue, xBC_buf, CL_TRUE, 0, 
                       xBC_size * sizeof(float), xBC_cpu.data(), 0, nullptr, nullptr);
    
    // Read conv weights and bias
    std::vector<float> conv_weight_cpu(conv_dim_ * kernel_size_);
    std::vector<float> conv_bias_cpu(conv_dim_);
    clEnqueueReadBuffer(queue, conv_weight_, CL_TRUE, 0, 
                       conv_weight_cpu.size() * sizeof(float), conv_weight_cpu.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, conv_bias_, CL_TRUE, 0, 
                       conv_bias_cpu.size() * sizeof(float), conv_bias_cpu.data(), 0, nullptr, nullptr);
    
    // Apply conv1d: xBC is [batch, seq_len, conv_dim], reshape to [batch, conv_dim, seq_len] for conv
    // Then apply conv1d with Swish, result back to [batch, seq_len, conv_dim]
    std::vector<float> xBC_conv_cpu(xBC_size);
    for (int b = 0; b < batch_size; ++b) {
        for (int c = 0; c < conv_dim_; ++c) {
            for (int s = 0; s < seq_len; ++s) {
                float sum = 0.0f;
                for (int k = 0; k < kernel_size_ && s + k < seq_len; ++k) {
                    int xBC_idx = b * seq_len * conv_dim_ + (s + k) * conv_dim_ + c;
                    int w_idx = c * kernel_size_ + k;
                    sum += conv_weight_cpu[w_idx] * xBC_cpu[xBC_idx];
                }
                sum += conv_bias_cpu[c];
                // Swish activation
                float sigmoid = 1.0f / (1.0f + expf(-sum));
                sum = sum * sigmoid;
                xBC_conv_cpu[b * seq_len * conv_dim_ + s * conv_dim_ + c] = sum;
            }
        }
    }
    
    // Write xBC_conv back to GPU
    clEnqueueWriteBuffer(queue, xBC_buf, CL_TRUE, 0, 
                         xBC_conv_cpu.size() * sizeof(float), xBC_conv_cpu.data(), 0, nullptr, nullptr);
    clFinish(queue);
    
    // Step 4: Split xBC into x, B, C
    // Read xBC_conv
    std::vector<float> xBC_conv_read(xBC_size);
    clEnqueueReadBuffer(queue, xBC_buf, CL_TRUE, 0, 
                       xBC_size * sizeof(float), xBC_conv_read.data(), 0, nullptr, nullptr);
    
    // Allocate buffers for x, B, C
    // After conv, xBC is split at [d_inner, d_inner + d_state*n_groups]
    // So: x = d_inner, B = d_state*n_groups, C = d_state*n_groups
    size_t x_size = batch_size * seq_len * d_inner_;
    size_t B_size = batch_size * seq_len * d_state_ * n_groups_;
    size_t C_size = batch_size * seq_len * d_state_ * n_groups_;
    
    cl_mem x_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, x_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create x buffer");
    
    cl_mem B_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, B_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create B buffer");
    
    cl_mem C_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, C_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create C buffer");
    
    // Split: MLX splits at [d_inner, d_inner + d_state*n_groups]
    // So: x = first d_inner, B = next d_state*n_groups, C = remaining d_state*n_groups
    std::vector<float> x_cpu(x_size);
    std::vector<float> B_cpu(B_size);
    std::vector<float> C_cpu(C_size);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            int base_idx = (b * seq_len + s) * conv_dim_;
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
            
            // Note: SSM forward kernels may be nullptr if they were skipped (e.g., PowerVR driver bug)
            // This is OK - forward() will check and use CPU fallback
        } catch (const std::exception& e) {
            std::cerr << "[SSDLayer] Error building kernels in forward(): " << e.what() << std::endl;
            std::cerr << "[SSDLayer] This may be a PowerVR driver issue. Will use CPU fallback for SSM forward." << std::endl;
            // Don't throw - allow CPU fallback to be used
            // SSM forward kernels will be nullptr, which is fine
        } catch (...) {
            std::cerr << "[SSDLayer] Unknown error building kernels (possible driver crash)" << std::endl;
            throw std::runtime_error("OpenCL driver crashed while building kernels");
        }
    }
    
    // Check if SSM forward kernels are available
    // If not (e.g., PowerVR driver bug), use CPU fallback
    if (!process_dt_kernel_) {
        // CPU fallback for SSM forward computation (matrix-based, matching MLX)
        std::cout << "[SSDLayer] Using CPU fallback for SSM forward (GPU kernels not available)" << std::endl;
        
        // Read necessary data from GPU
        std::vector<float> dt_cpu(dt_size);
        std::vector<float> x_cpu(x_size);
        std::vector<float> B_cpu(B_size);
        std::vector<float> C_cpu(C_size);
        std::vector<float> A_log_cpu(n_heads_);
        std::vector<float> dt_bias_cpu(n_heads_);
        std::vector<float> D_cpu(n_heads_);
        
        clEnqueueReadBuffer(queue, dt_buf, CL_TRUE, 0, dt_size * sizeof(float), dt_cpu.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, x_buf, CL_TRUE, 0, x_size * sizeof(float), x_cpu.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, B_buf, CL_TRUE, 0, B_size * sizeof(float), B_cpu.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, C_buf, CL_TRUE, 0, C_size * sizeof(float), C_cpu.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, A_, CL_TRUE, 0, n_heads_ * sizeof(float), A_log_cpu.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, dt_bias_, CL_TRUE, 0, n_heads_ * sizeof(float), dt_bias_cpu.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue, D_, CL_TRUE, 0, n_heads_ * sizeof(float), D_cpu.data(), 0, nullptr, nullptr);
        
        // Convert A_log to A_actual
        std::vector<float> A_actual(n_heads_);
        for (int h = 0; h < n_heads_; ++h) {
            float a_log = A_log_cpu[h];
            float softplus = (a_log > 20.0f) ? a_log : log1pf(expf(a_log));
            A_actual[h] = -softplus;
        }
        
        // Process dt: add bias, softplus, clamp
        std::vector<float> dt_processed(dt_size);
        const float dt_min = 0.0f;
        const float dt_max = 1e10f;
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                for (int h = 0; h < n_heads_; ++h) {
                    int idx = (b * seq_len + s) * n_heads_ + h;
                    float dt_val = dt_cpu[idx] + dt_bias_cpu[h];
                    dt_val = (dt_val > 20.0f) ? dt_val : log1pf(expf(dt_val));  // softplus
                    if (dt_val < dt_min) dt_val = dt_min;
                    if (dt_val > dt_max) dt_val = dt_max;
                    dt_processed[idx] = dt_val;
                }
            }
        }
        
        // Compute dtA = dt * A
        std::vector<float> dtA(dt_size);
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                for (int h = 0; h < n_heads_; ++h) {
                    int idx = (b * seq_len + s) * n_heads_ + h;
                    dtA[idx] = dt_processed[idx] * A_actual[h];
                }
            }
        }
        
        // Compute segsum decay matrix: decay[s,t] = exp(sum from k=t+1 to s of dtA[k])
        size_t decay_size = batch_size * n_heads_ * seq_len * seq_len;
        std::vector<float> decay(decay_size);
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < n_heads_; ++h) {
                for (int s = 0; s < seq_len; ++s) {
                    for (int t = 0; t < seq_len; ++t) {
                        float segsum = 0.0f;
                        for (int k = t + 1; k <= s; ++k) {
                            int dtA_idx = (b * seq_len + k) * n_heads_ + h;
                            segsum += dtA[dtA_idx];
                        }
                        int decay_idx = (b * n_heads_ + h) * seq_len * seq_len + s * seq_len + t;
                        decay[decay_idx] = expf(segsum);
                    }
                }
            }
        }
        
        // Compute CB = C @ B (matrix multiplication per group)
        size_t CB_size = batch_size * seq_len * seq_len * n_groups_;
        std::vector<float> CB(CB_size);
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                for (int t = 0; t < seq_len; ++t) {
                    for (int g = 0; g < n_groups_; ++g) {
                        float sum = 0.0f;
                        for (int state_i = 0; state_i < d_state_; ++state_i) {
                            int C_idx = (b * seq_len + s) * (n_groups_ * d_state_) + g * d_state_ + state_i;
                            int B_idx = (b * seq_len + t) * (n_groups_ * d_state_) + g * d_state_ + state_i;
                            sum += C_cpu[C_idx] * B_cpu[B_idx];
                        }
                        int CB_idx = ((b * seq_len + s) * seq_len + t) * n_groups_ + g;
                        CB[CB_idx] = sum;
                    }
                }
            }
        }
        
        // Compute dtx = dt * x (element-wise)
        std::vector<float> dtx(x_size);
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                for (int h = 0; h < n_heads_; ++h) {
                    float dt_val = dt_processed[(b * seq_len + s) * n_heads_ + h];
                    for (int d = 0; d < d_head_; ++d) {
                        int x_idx = (b * seq_len + s) * d_inner_ + h * d_head_ + d;
                        dtx[x_idx] = dt_val * x_cpu[x_idx];
                    }
                }
            }
        }
        
        // Compute final output: y = tril(CB * decay) @ dtx + D * x
        std::vector<float> y_cpu(x_size);
        for (int b = 0; b < batch_size; ++b) {
            for (int s = 0; s < seq_len; ++s) {
                for (int h = 0; h < n_heads_; ++h) {
                    int group_idx = h % n_groups_;
                    for (int d = 0; d < d_head_; ++d) {
                        int x_idx = (b * seq_len + s) * d_inner_ + h * d_head_ + d;
                        float output_sum = 0.0f;
                        
                        // tril(CB * decay) @ dtx
                        for (int t = 0; t <= s; ++t) {
                            int x_t_idx = (b * seq_len + t) * d_inner_ + h * d_head_ + d;
                            float dtx_t = dtx[x_t_idx];
                            
                            int decay_idx = (b * n_heads_ + h) * seq_len * seq_len + s * seq_len + t;
                            float decay_st = decay[decay_idx];
                            
                            int CB_idx = ((b * seq_len + s) * seq_len + t) * n_groups_ + group_idx;
                            float CB_st = CB[CB_idx];
                            
                            output_sum += CB_st * decay_st * dtx_t;
                        }
                        
                        // Add D * x
                        output_sum += D_cpu[h] * x_cpu[x_idx];
                        y_cpu[x_idx] = output_sum;
                    }
                }
            }
        }
        
        // Write result back to GPU
        clEnqueueWriteBuffer(queue, x_buf, CL_TRUE, 0, x_size * sizeof(float), y_cpu.data(), 0, nullptr, nullptr);
        clFinish(queue);
        
        // CPU fallback complete - continue with gate and norm
        // (skip GPU kernel cleanup since we didn't allocate those buffers)
    } else {
        // GPU path: Use SSM forward kernels
        // Convert A_log to A_actual on CPU (small operation)
        std::vector<float> A_cpu_gpu(n_heads_);
        clEnqueueReadBuffer(queue, A_, CL_TRUE, 0, n_heads_ * sizeof(float), A_cpu_gpu.data(), 0, nullptr, nullptr);
        std::vector<float> A_actual_gpu(n_heads_);
        for (int h = 0; h < n_heads_; ++h) {
            float a_log = A_cpu_gpu[h];
            float softplus = (a_log > 20.0f) ? a_log : log1pf(expf(a_log));
            A_actual_gpu[h] = -softplus;
        }
        
        // Upload A_actual to GPU
        A_actual_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                             n_heads_ * sizeof(float), A_actual_gpu.data(), &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create A_actual buffer");
        
        // Allocate GPU buffers for intermediate results
        dt_processed_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dt_processed_size * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create dt_processed buffer");
        
        dtA_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dtA_size * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create dtA buffer");
        
        decay_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, decay_size * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create decay buffer");
        
        CB_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, CB_size * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create CB buffer");
        
        dtx_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dtx_size * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create dtx buffer");
        
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
        
        // Kernel 4: Compute CB = C @ B
        size_t CB_global_size[4] = {
            static_cast<size_t>(batch_size), 
            static_cast<size_t>(seq_len), 
            static_cast<size_t>(seq_len), 
            static_cast<size_t>(n_groups_)
        };
        err = clSetKernelArg(compute_CB_kernel_, 0, sizeof(cl_mem), &B_buf);
        err |= clSetKernelArg(compute_CB_kernel_, 1, sizeof(cl_mem), &C_buf);
        err |= clSetKernelArg(compute_CB_kernel_, 2, sizeof(cl_mem), &CB_buf);
        err |= clSetKernelArg(compute_CB_kernel_, 3, sizeof(int), &batch_size);
        err |= clSetKernelArg(compute_CB_kernel_, 4, sizeof(int), &seq_len);
        err |= clSetKernelArg(compute_CB_kernel_, 5, sizeof(int), &n_groups_);
        err |= clSetKernelArg(compute_CB_kernel_, 6, sizeof(int), &d_state_);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to set compute_CB kernel arguments");
        
        err = clEnqueueNDRangeKernel(queue, compute_CB_kernel_, 4, nullptr, CB_global_size, nullptr, 0, nullptr, nullptr);
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
        // Create a separate output buffer for the SSM result
        cl_mem x_ssm_output_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, x_size * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create x_ssm_output buffer");
        
        size_t output_global_size[4] = {
            static_cast<size_t>(batch_size), 
            static_cast<size_t>(seq_len), 
            static_cast<size_t>(n_heads_), 
            static_cast<size_t>(d_head_)
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
        
        err = clEnqueueNDRangeKernel(queue, compute_ssm_output_kernel_, 4, nullptr, output_global_size, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue compute_ssm_output kernel");
        
        clFinish(queue);
        
        // Copy SSM output to x_buf for subsequent steps
        err = clEnqueueCopyBuffer(queue, x_ssm_output_buf, x_buf, 0, 0, x_size * sizeof(float), 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to copy SSM output to x_buf");
        clFinish(queue);
        
        // Cleanup SSM output buffer
        clReleaseMemObject(x_ssm_output_buf);
        
        // Cleanup GPU buffers (only if GPU path was used)
        if (A_actual_buf) clReleaseMemObject(A_actual_buf);
        if (dt_processed_buf) clReleaseMemObject(dt_processed_buf);
        if (dtA_buf) clReleaseMemObject(dtA_buf);
        if (decay_buf) clReleaseMemObject(decay_buf);
        if (CB_buf) clReleaseMemObject(CB_buf);
        if (dtx_buf) clReleaseMemObject(dtx_buf);
    }
    
    // Step 6: Apply gate: Swish(z) * x, then RMS norm
    // Read z and x from GPU
    std::vector<float> z_cpu(z_size);
    std::vector<float> x_cpu_gated(x_size);
    clEnqueueReadBuffer(queue, z_buf, CL_TRUE, 0, z_size * sizeof(float), z_cpu.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, x_buf, CL_TRUE, 0, x_size * sizeof(float), x_cpu_gated.data(), 0, nullptr, nullptr);
    
    // Compute Swish(z) * x
    std::vector<float> gated_cpu(x_size);
    for (int i = 0; i < x_size; ++i) {
        float z_val = z_cpu[i];
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float swish_z = z_val * sigmoid_z;
        gated_cpu[i] = x_cpu_gated[i] * swish_z;
    }
    
    // Write gated result to x_buf for RMS norm
    cl_mem gated_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, x_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create gated buffer");
    clEnqueueWriteBuffer(queue, gated_buf, CL_TRUE, 0, x_size * sizeof(float), gated_cpu.data(), 0, nullptr, nullptr);
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
        // Initialize conv_state and ssm_state to zeros
        size_t conv_state_size = batch_size * conv_dim_ * (kernel_size_ - 1);
        size_t ssm_state_size = batch_size * n_heads_ * d_head_ * d_state_;
        
        cl_context context = ctx_->getContext();
        cl_int err;
        
        // Initialize conv_state
        state->state1 = clCreateBuffer(context, CL_MEM_READ_WRITE, conv_state_size * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create conv_state buffer");
        std::vector<float> zeros_conv(conv_state_size, 0.0f);
        clEnqueueWriteBuffer(queue, state->state1, CL_TRUE, 0, conv_state_size * sizeof(float), zeros_conv.data(), 0, nullptr, nullptr);
        
        // Initialize ssm_state
        state->state2 = clCreateBuffer(context, CL_MEM_READ_WRITE, ssm_state_size * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) {
            clReleaseMemObject(state->state1);
            throw std::runtime_error("Failed to create ssm_state buffer");
        }
        std::vector<float> zeros_ssm(ssm_state_size, 0.0f);
        clEnqueueWriteBuffer(queue, state->state2, CL_TRUE, 0, ssm_state_size * sizeof(float), zeros_ssm.data(), 0, nullptr, nullptr);
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
    
    cl_mem z_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, z_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create z buffer");
    cl_mem xBC_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, xBC_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create xBC buffer");
    cl_mem dt_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, dt_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create dt buffer");
    
    splitInProjOutput(in_proj_output, batch_size, 1, z_buf, xBC_buf, dt_buf, queue);
    clFinish(queue);
    clReleaseMemObject(in_proj_output);
    
    // Step 3: conv1d_update on xBC (using conv_state from state->state1)
    // conv_state shape: [batch_size, conv_dim, kernel_size - 1]
    // xBC: [batch_size, conv_dim] (single token)
    size_t conv_state_size = batch_size * conv_dim_ * (kernel_size_ - 1);
    cl_mem conv_state = state->state1;
    if (conv_state == nullptr) {
        // Initialize conv_state to zeros
        conv_state = clCreateBuffer(context, CL_MEM_READ_WRITE, conv_state_size * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create conv_state buffer");
        std::vector<float> zeros(conv_state_size, 0.0f);
        clEnqueueWriteBuffer(queue, conv_state, CL_TRUE, 0, conv_state_size * sizeof(float), zeros.data(), 0, nullptr, nullptr);
    }
    
    cl_mem conv_output = clCreateBuffer(context, CL_MEM_READ_WRITE, xBC_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create conv_output buffer");
    cl_mem next_conv_state = clCreateBuffer(context, CL_MEM_READ_WRITE, conv_state_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create next_conv_state buffer");
    
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
    
    size_t global_size[2] = {static_cast<size_t>(batch_size), static_cast<size_t>(conv_dim_)};
    err = clEnqueueNDRangeKernel(queue, conv_update_kernel_, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue conv_update kernel");
    clFinish(queue);
    
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
    
    // Read xBC to CPU for splitting
    std::vector<float> xBC_cpu(xBC_size);
    clEnqueueReadBuffer(queue, xBC_buf, CL_TRUE, 0, xBC_size * sizeof(float), xBC_cpu.data(), 0, nullptr, nullptr);
    
    std::vector<float> x_cpu(x_size);
    std::vector<float> B_cpu(BC_size);
    std::vector<float> C_cpu(BC_size);
    
    for (int b = 0; b < batch_size; ++b) {
        int xBC_base = b * (d_inner_ + 2 * d_state_ * n_groups_);
        int x_base = b * d_inner_;
        int BC_base = b * d_state_ * n_groups_;
        
        // Copy x
        for (int i = 0; i < d_inner_; ++i) {
            x_cpu[x_base + i] = xBC_cpu[xBC_base + i];
        }
        
        // Copy B
        for (int i = 0; i < d_state_ * n_groups_; ++i) {
            B_cpu[BC_base + i] = xBC_cpu[xBC_base + d_inner_ + i];
        }
        
        // Copy C
        for (int i = 0; i < d_state_ * n_groups_; ++i) {
            C_cpu[BC_base + i] = xBC_cpu[xBC_base + d_inner_ + d_state_ * n_groups_ + i];
        }
    }
    
    clReleaseMemObject(xBC_buf);
    
    // Step 5: SSM update (using ssm_state from state->state2)
    // Read dt, A, D, dt_bias
    std::vector<float> dt_cpu(dt_size);
    std::vector<float> A_cpu(n_heads_);
    std::vector<float> D_cpu(n_heads_);
    std::vector<float> dt_bias_cpu(n_heads_);
    
    clEnqueueReadBuffer(queue, dt_buf, CL_TRUE, 0, dt_size * sizeof(float), dt_cpu.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, A_, CL_TRUE, 0, n_heads_ * sizeof(float), A_cpu.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, D_, CL_TRUE, 0, n_heads_ * sizeof(float), D_cpu.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, dt_bias_, CL_TRUE, 0, n_heads_ * sizeof(float), dt_bias_cpu.data(), 0, nullptr, nullptr);
    
    // Convert A_log to A: A = exp(A_log)
    for (int h = 0; h < n_heads_; ++h) {
        {
            float a_log = A_cpu[h];
            float softplus = (a_log > 20.0f) ? a_log : log1pf(expf(a_log));
            A_cpu[h] = -softplus;
        }
    }
    
    // Process dt: add bias, apply softplus, clamp
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < n_heads_; ++h) {
            int dt_idx = b * n_heads_ + h;
            float dt_val = dt_cpu[dt_idx] + dt_bias_cpu[h];
            
            // Apply softplus: log(1 + exp(dt))
            if (dt_val > 20.0f) {
                dt_val = dt_val;
            } else {
                dt_val = log1pf(expf(dt_val));
            }
            
            // Clamp dt (dt_min=0.0, dt_max=inf)
            if (dt_val < 0.0f) dt_val = 0.0f;
            // dt_max is inf, so no upper clamp needed
            
            dt_cpu[dt_idx] = dt_val;
        }
    }
    
    // Get or initialize ssm_state
    // ssm_state shape: [batch_size, n_heads, d_head, d_state]
    size_t ssm_state_size = batch_size * n_heads_ * d_head_ * d_state_;
    cl_mem ssm_state = state->state2;
    if (ssm_state == nullptr) {
        // Initialize ssm_state to zeros
        ssm_state = clCreateBuffer(context, CL_MEM_READ_WRITE, ssm_state_size * sizeof(float), nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create ssm_state buffer");
        std::vector<float> zeros(ssm_state_size, 0.0f);
        clEnqueueWriteBuffer(queue, ssm_state, CL_TRUE, 0, ssm_state_size * sizeof(float), zeros.data(), 0, nullptr, nullptr);
    }
    
    // Read ssm_state to CPU
    std::vector<float> ssm_state_cpu(ssm_state_size);
    clEnqueueReadBuffer(queue, ssm_state, CL_TRUE, 0, ssm_state_size * sizeof(float), ssm_state_cpu.data(), 0, nullptr, nullptr);
    
    // Perform SSM update for single token
    // x is [batch_size, n_heads, d_head] after reshape
    // For each batch, head, and head_dim, update state and compute output
    std::vector<float> x_ssm_cpu(x_size);
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < n_heads_; ++h) {
            float delta = dt_cpu[b * n_heads_ + h];
            float A_val = A_cpu[h];
            
            for (int d = 0; d < d_head_; ++d) {
                int x_idx = b * d_inner_ + h * d_head_ + d;
                float x_val = x_cpu[x_idx];
                
                // Get group index for this head
                int group_idx = h % n_groups_;
                
                float output_sum = 0.0f;
                
                // Update state and compute output for each state dimension
                for (int state_i = 0; state_i < d_state_; ++state_i) {
                    int state_idx = (b * n_heads_ + h) * d_head_ * d_state_ + d * d_state_ + state_i;
                    float old_state = ssm_state_cpu[state_idx];
                    
                    // Get B and C values
                    int B_idx = b * (d_state_ * n_groups_) + group_idx * d_state_ + state_i;
                    float B_val = B_cpu[B_idx];
                    int C_idx = B_idx;  // Same indexing
                    float C_val = C_cpu[C_idx];
                    
                    // State update: new_state = old_state * exp(A * delta) + B * delta * x
                    float new_state = old_state * expf(A_val * delta) + B_val * delta * x_val;
                    ssm_state_cpu[state_idx] = new_state;
                    
                    // Output contribution: new_state * C
                    output_sum += new_state * C_val;
                }
                
                // Add D skip connection: D * x
                output_sum += D_cpu[h] * x_val;
                
                x_ssm_cpu[x_idx] = output_sum;
            }
        }
    }
    
    // Write updated ssm_state back
    cl_mem next_ssm_state = clCreateBuffer(context, CL_MEM_READ_WRITE, ssm_state_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create next_ssm_state buffer");
    clEnqueueWriteBuffer(queue, next_ssm_state, CL_TRUE, 0, ssm_state_size * sizeof(float), ssm_state_cpu.data(), 0, nullptr, nullptr);
    
    // Update state->state2
    if (state->state2 != ssm_state) {
        if (state->state2 != nullptr) clReleaseMemObject(state->state2);
    }
    state->state2 = next_ssm_state;
    if (ssm_state != state->state2) clReleaseMemObject(ssm_state);
    
    // Write x_ssm to GPU
    cl_mem x_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, x_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create x buffer");
    clEnqueueWriteBuffer(queue, x_buf, CL_TRUE, 0, x_size * sizeof(float), x_ssm_cpu.data(), 0, nullptr, nullptr);
    clFinish(queue);
    
    // Step 6: Apply gate: Swish(z) * x, then RMS norm
    // Read z
    std::vector<float> z_cpu(z_size);
    clEnqueueReadBuffer(queue, z_buf, CL_TRUE, 0, z_size * sizeof(float), z_cpu.data(), 0, nullptr, nullptr);
    
    // Compute Swish(z) * x
    std::vector<float> gated_cpu(x_size);
    for (int i = 0; i < x_size; ++i) {
        float z_val = z_cpu[i];
        float sigmoid_z = 1.0f / (1.0f + expf(-z_val));
        float swish_z = z_val * sigmoid_z;
        gated_cpu[i] = x_ssm_cpu[i] * swish_z;
    }
    
    // Write gated result to GPU
    cl_mem gated_buf = clCreateBuffer(context, CL_MEM_READ_WRITE, x_size * sizeof(float), nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create gated buffer");
    clEnqueueWriteBuffer(queue, gated_buf, CL_TRUE, 0, x_size * sizeof(float), gated_cpu.data(), 0, nullptr, nullptr);
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

