#include "rms_norm_layer.h"
#include "../opencl_context.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <cmath>
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
    // size_t buffer_size_kb = (d_model * sizeof(float)) / 1024;
    // std::cout << "  [RMSNorm] d_model=" << d_model 
    //           << ", params=" << d_model
    //           << ", buffer_size=" << buffer_size_kb << " KB" << std::endl;
    
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
    
    // Calculate mean square within this sequence element (match MLX: no clamping, 32-bit precision)
    float mean_square = 0.0f;
    for (int i = 0; i < d_model; ++i) {
        float val = input[seq_idx * d_model + i];
        mean_square += val * val;
    }
    mean_square /= d_model;
    
    // RMS = sqrt(mean_square + eps)
    float rms = sqrt(mean_square + eps);
    
    // Normalize: output = (input / rms) * weight (match MLX exactly)
    float weight_val = weight[feat_idx];
    output[idx] = (input[idx] / rms) * weight_val;
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
    
    // Create a NEW output buffer for each call to avoid stale data issues
    // Use CL_MEM_READ_WRITE so we can explicitly initialize it to zero
    cl_mem output_buffer = nullptr;
    output_buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, nullptr, nullptr);
    if (!output_buffer) {
        throw std::runtime_error("Failed to create RMS norm output buffer");
    }
    
    // Explicitly zero the buffer using a simple kernel to ensure it works
    // clEnqueueFillBuffer might not be reliable on all devices
    static cl_kernel zero_kernel = nullptr;
    static bool zero_kernel_built = false;
    
    if (!zero_kernel_built) {
        const char* zero_kernel_source = R"(
__kernel void zero_buffer(__global float* buffer, const int size) {
    int idx = get_global_id(0);
    if (idx < size) {
        buffer[idx] = 0.0f;
    }
}
)";
        auto& ctx_mgr = OpenCLContextManager::getInstance();
        std::vector<std::string> sources = {std::string(zero_kernel_source)};
        std::string cache_key = ctx_mgr.generateCacheKey(sources) + "_zero";
        cl_program zero_program = ctx_mgr.buildProgram(sources, cache_key);
        zero_kernel = ctx_mgr.getKernel(zero_program, "zero_buffer");
        zero_kernel_built = true;
    }
    
    if (zero_kernel) {
        cl_int zero_err = clSetKernelArg(zero_kernel, 0, sizeof(cl_mem), &output_buffer);
        zero_err |= clSetKernelArg(zero_kernel, 1, sizeof(int), &total_elements);
        if (zero_err == CL_SUCCESS) {
            size_t zero_global = ((total_elements + 63) / 64) * 64;  // Round to multiple of 64
            size_t zero_local = 64;
            cl_int launch_err = clEnqueueNDRangeKernel(queue, zero_kernel, 1, nullptr, 
                                                       &zero_global, &zero_local, 0, nullptr, nullptr);
            if (launch_err == CL_SUCCESS) {
                clFinish(queue);  // Ensure zeroing completes
                
                // Debug: Verify zero kernel worked for layer 6
                static int zero_check_count = 0;
                zero_check_count++;
                if (zero_check_count == 7) {  // Layer 6
                    std::vector<float> zero_check(total_elements);
                    cl_int check_err = clEnqueueReadBuffer(queue, output_buffer, CL_TRUE, 0,
                        output_size, zero_check.data(), 0, nullptr, nullptr);
                    if (check_err == CL_SUCCESS) {
                        int non_zero_count = 0;
                        int nan_count = 0;
                        for (size_t i = 0; i < zero_check.size(); ++i) {
                            if (zero_check[i] != 0.0f) {
                                non_zero_count++;
                                if (std::isnan(zero_check[i])) nan_count++;
                            }
                        }
                        std::cout << "  [RMSNorm Zero Check] After zero kernel: " << non_zero_count 
                                  << " non-zero values, " << nan_count << " NaNs out of " 
                                  << total_elements << " values" << std::endl;
                    }
                }
            }
        }
    }
    
    // If we had a previous buffer, release it (but keep the size for comparison)
    if (output_buffer_) {
        clReleaseMemObject(output_buffer_);
    }
    output_buffer_ = output_buffer;
    output_buffer_size_ = output_size;
    
    // Set kernel arguments
    cl_int err;
    err = clSetKernelArg(kernel_, 0, sizeof(cl_mem), &input);
    err |= clSetKernelArg(kernel_, 1, sizeof(cl_mem), &weights_buffer_);
    err |= clSetKernelArg(kernel_, 2, sizeof(cl_mem), &output_buffer_);
    err |= clSetKernelArg(kernel_, 3, sizeof(int), &d_model_);
    err |= clSetKernelArg(kernel_, 4, sizeof(int), &total_elements);
    const float eps = 1e-5f;  // Match MLX default epsilon (was 1e-6)
    err |= clSetKernelArg(kernel_, 5, sizeof(float), &eps);
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set RMS norm kernel arguments");
    }
    
    // Execute kernel
    size_t global_size = total_elements;
    
    // Use a local work-group size that evenly divides the global size
    // 7168 = 7 * 1024, so we can use 64, 128, 256, or 512 as local size
    // Using 64 to ensure maximum compatibility
    size_t local_size = 64;
    
    // Round global_size up to be a multiple of local_size (OpenCL requirement)
    size_t rounded_global = ((global_size + local_size - 1) / local_size) * local_size;
    
    // Debug: Check device work-group size limits for layer 6
    static int kernel_launch_count = 0;
    kernel_launch_count++;
    if (kernel_launch_count == 7) {  // Layer 6
        size_t max_work_group_size = 0;
        cl_int wg_err = clGetKernelWorkGroupInfo(kernel_, ctx_->getDevice(), CL_KERNEL_WORK_GROUP_SIZE,
                                                 sizeof(size_t), &max_work_group_size, nullptr);
        if (wg_err == CL_SUCCESS && max_work_group_size < local_size) {
            local_size = max_work_group_size;
            rounded_global = ((global_size + local_size - 1) / local_size) * local_size;
        }
        if (wg_err == CL_SUCCESS) {
            std::cout << "  [RMSNorm Layer 6 Kernel Debug] Max work-group size: " << max_work_group_size 
                      << ", Using local size: " << local_size 
                      << ", Global size: " << global_size 
                      << " (rounded to: " << rounded_global << ")" << std::endl;
        }
    }
    
    // Launch with explicit local size to ensure all work items execute
    err = clEnqueueNDRangeKernel(queue, kernel_, 1, nullptr, &rounded_global, &local_size, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue RMS norm kernel, error=" + std::to_string(err));
    }
    
    // Ensure kernel completes before returning buffer
    clFinish(queue);
    
    // Retain buffer before returning to ensure it stays valid
    clRetainMemObject(output_buffer_);
    
    // Debug: Check RMSNorm input, weights, and output for layer 6
    static int rms_check_count = 0;
    rms_check_count++;
    if (rms_check_count == 7) {  // Layer 6's RMSNorm (after layers 0-5)
        // First check the INPUT buffer
        std::vector<float> rms_input(total_elements);
        cl_int input_check_err = clEnqueueReadBuffer(queue, input, CL_TRUE, 0,
            output_size, rms_input.data(), 0, nullptr, nullptr);
        if (input_check_err == CL_SUCCESS) {
            int input_nan_count = 0;
            std::vector<int> input_nan_per_token(seq_len, 0);
            std::vector<float> token_sum(seq_len, 0.0f);
            std::vector<float> token_sum_sq(seq_len, 0.0f);
            for (size_t i = 0; i < rms_input.size(); ++i) {
                int token_idx = i / d_model_;
                if (token_idx < seq_len) {
                    float val = rms_input[i];
                    if (std::isnan(val)) {
                        input_nan_count++;
                        input_nan_per_token[token_idx]++;
                    } else if (std::isinf(val)) {
                        input_nan_count++;  // Treat Inf as problematic
                        input_nan_per_token[token_idx]++;
                    } else {
                        token_sum[token_idx] += val;
                        token_sum_sq[token_idx] += val * val;
                    }
                }
            }
            std::cout << "  [RMSNorm Layer 6 Input Debug] Input has " << input_nan_count 
                      << " NaN/Inf out of " << total_elements << " values" << std::endl;
            std::cout << "  [RMSNorm Layer 6 Input Debug] Input NaN/Inf per token: ";
            for (int i = 0; i < seq_len; ++i) {
                std::cout << "token" << i << "=" << input_nan_per_token[i] << "/" << d_model_ << " ";
            }
            std::cout << std::endl;
            std::cout << "  [RMSNorm Layer 6 Input Debug] Token sum_sq (mean_sq proxy): ";
            for (int i = 0; i < seq_len; ++i) {
                float mean_sq_proxy = token_sum_sq[i] / d_model_;
                std::cout << "token" << i << "=" << mean_sq_proxy << " ";
            }
            std::cout << std::endl;
        }
        
        // Check weights for NaN
        std::vector<float> weights_check(d_model_);
        size_t weights_size = d_model_ * sizeof(float);
        cl_int weights_check_err = clEnqueueReadBuffer(queue, weights_buffer_, CL_TRUE, 0,
            weights_size, weights_check.data(), 0, nullptr, nullptr);
        if (weights_check_err == CL_SUCCESS) {
            int weights_nan_count = 0;
            for (float w : weights_check) {
                if (std::isnan(w) || std::isinf(w)) {
                    weights_nan_count++;
                }
            }
            std::cout << "  [RMSNorm Layer 6 Weights Debug] Weights have " << weights_nan_count 
                      << " NaN/Inf out of " << d_model_ << " values" << std::endl;
        }
        
        // Then check the OUTPUT buffer
        std::vector<float> rms_output(total_elements);
        cl_int check_err = clEnqueueReadBuffer(queue, output_buffer_, CL_TRUE, 0,
            output_size, rms_output.data(), 0, nullptr, nullptr);
        if (check_err == CL_SUCCESS) {
            int nan_count = 0;
            std::vector<int> nan_per_token(seq_len, 0);
            for (size_t i = 0; i < rms_output.size(); ++i) {
                if (std::isnan(rms_output[i])) {
                    nan_count++;
                    int token_idx = i / d_model_;
                    if (token_idx < seq_len) {
                        nan_per_token[token_idx]++;
                    }
                }
            }
            std::cout << "  [RMSNorm Layer 6 Debug] Output: " << nan_count 
                      << " NaNs out of " << total_elements << " values" << std::endl;
            std::cout << "  [RMSNorm Layer 6 Debug] NaNs per token: ";
            for (int i = 0; i < seq_len; ++i) {
                std::cout << "token" << i << "=" << nan_per_token[i] << "/" << d_model_ << " ";
            }
            std::cout << std::endl;
            
            // Check token 5 specifically
            std::cout << "  [RMSNorm Layer 6 Debug] Token 5 first 5 values: ";
            for (int i = 5 * d_model_; i < 5 * d_model_ + 5; ++i) {
                std::cout << rms_output[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "  [RMSNorm Layer 6 Debug] Token 0 first 5 values: ";
            for (int i = 0; i < 5; ++i) {
                std::cout << rms_output[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    return output_buffer_;
}

cl_mem RMSNormLayer::step(cl_mem input, int batch_size, cl_command_queue queue) {
    // For step, we have [batch_size, d_model] instead of [batch_size, seq_len, d_model]
    // seq_len = 1 in this case
    return forward(input, batch_size, 1, queue);
}

} // namespace cartesia_opencl

