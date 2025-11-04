#include "linear_layer.h"
#include "../opencl_context.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <CL/cl.h>

namespace cartesia_opencl {

LinearLayer::LinearLayer(OpenCLContextManager* ctx, int input_dim, int output_dim, bool has_bias)
    : ctx_(ctx)
    , input_dim_(input_dim)
    , output_dim_(output_dim)
    , has_bias_(has_bias)
    , program_(nullptr)
    , matmul_kernel_(nullptr)
    , matvec_kernel_(nullptr)
    , weights_buffer_(nullptr)
    , bias_buffer_(nullptr)
    , weights_initialized_(false)
    , output_buffer_(nullptr)
    , output_buffer_size_(0)
{
    if (!ctx_) {
        throw std::runtime_error("OpenCLContext is null");
    }
    
    // Debug output
    size_t num_params = static_cast<size_t>(output_dim) * input_dim;
    size_t buffer_size_mb = (num_params * sizeof(float)) / (1024 * 1024);
    // std::cout << "  [Linear] input_dim=" << input_dim 
    //           << ", output_dim=" << output_dim 
    //           << ", params=" << num_params;
    // if (has_bias) {
    //     std::cout << " (with bias: +" << output_dim << " params)";
    // }
    // std::cout << ", buffer_size=" << buffer_size_mb << " MB" << std::endl;
    
    buildKernels();
}

LinearLayer::~LinearLayer() {
    if (matmul_kernel_) clReleaseKernel(matmul_kernel_);
    if (matvec_kernel_) clReleaseKernel(matvec_kernel_);
    if (program_) clReleaseProgram(program_);
    if (weights_buffer_) clReleaseMemObject(weights_buffer_);
    if (bias_buffer_) clReleaseMemObject(bias_buffer_);
    if (output_buffer_) clReleaseMemObject(output_buffer_);
}

void LinearLayer::buildKernels() {
    // Embedded linear kernel source
    const char* linear_cl_source = R"(
// Matrix multiplication: C = A * B
// A: [m, k], B: [k, n], C: [m, n]
// For batched: A: [batch, m, k], B: [k, n], C: [batch, m, n]

__kernel void matmul(
    __global const float* A,          // Input matrix [m, k] or [batch, m, k]
    __global const float* B,          // Weight matrix [k, n]
    __global float* C,                // Output matrix [m, n] or [batch, m, n]
    const int m,                      // Rows in A / output
    const int k,                      // Columns in A, rows in B
    const int n,                      // Columns in B / output
    const int batch_size              // Batch size (1 if not batched)
) {
    const int batch_idx = get_global_id(0) / m;
    const int row = get_global_id(0) % m;
    const int col = get_global_id(1);
    
    if (batch_idx >= batch_size || row >= m || col >= n) return;
    
    float sum = 0.0f;
    const float max_val = 1e10f;  // Prevent overflow
    const float min_val = -1e10f;
    
    for (int i = 0; i < k; ++i) {
        int a_idx = batch_idx * m * k + row * k + i;
        int b_idx = i * n + col;
        
        float a_val = A[a_idx];
        float b_val = B[b_idx];
        
        // Clamp to prevent Inf/NaN and overflow
        if (isnan(a_val) || isinf(a_val)) a_val = 0.0f;
        if (isnan(b_val) || isinf(b_val)) b_val = 0.0f;
        if (a_val > max_val) a_val = max_val;
        if (a_val < min_val) a_val = min_val;
        if (b_val > max_val) b_val = max_val;
        if (b_val < min_val) b_val = min_val;
        
        float product = a_val * b_val;
        // Clamp product to prevent sum overflow
        if (product > max_val) product = max_val;
        if (product < min_val) product = min_val;
        
        sum += product;
        
        // Clamp sum periodically to prevent accumulation overflow
        if (sum > max_val) sum = max_val;
        if (sum < min_val) sum = min_val;
    }
    
    int c_idx = batch_idx * m * n + row * n + col;
    
    // Final clamp before writing
    if (isnan(sum) || isinf(sum)) {
        sum = 0.0f;
    } else {
        if (sum > max_val) sum = max_val;
        if (sum < min_val) sum = min_val;
    }
    
    C[c_idx] = sum;
}

// Matrix-vector multiplication: y = A * x (for step function)
// A: [m, n], x: [n], y: [m]
__kernel void matvec(
    __global const float* A,          // Weight matrix [m, n]
    __global const float* x,          // Input vector [n]
    __global float* y,                // Output vector [m]
    const int m,                      // Rows in A
    const int n                       // Columns in A, size of x
) {
    const int row = get_global_id(0);
    if (row >= m) return;
    
    float sum = 0.0f;
    const float max_val = 1e10f;  // Prevent overflow
    const float min_val = -1e10f;
    
    for (int i = 0; i < n; ++i) {
        float a_val = A[row * n + i];
        float x_val = x[i];
        
        // Clamp to prevent Inf/NaN and overflow
        if (isnan(a_val) || isinf(a_val)) a_val = 0.0f;
        if (isnan(x_val) || isinf(x_val)) x_val = 0.0f;
        if (a_val > max_val) a_val = max_val;
        if (a_val < min_val) a_val = min_val;
        if (x_val > max_val) x_val = max_val;
        if (x_val < min_val) x_val = min_val;
        
        float product = a_val * x_val;
        // Clamp product to prevent sum overflow
        if (product > max_val) product = max_val;
        if (product < min_val) product = min_val;
        
        sum += product;
        
        // Clamp sum periodically to prevent accumulation overflow
        if (sum > max_val) sum = max_val;
        if (sum < min_val) sum = min_val;
    }
    
    // Final clamp before writing
    if (isnan(sum) || isinf(sum)) {
        sum = 0.0f;
    } else {
        if (sum > max_val) sum = max_val;
        if (sum < min_val) sum = min_val;
    }
    
    y[row] = sum;
}
)";
    
    // Build program
    auto& ctx_mgr = OpenCLContextManager::getInstance();
    std::vector<std::string> sources = {std::string(linear_cl_source)};
    std::string cache_key = ctx_mgr.generateCacheKey(sources);
    program_ = ctx_mgr.buildProgram(sources, cache_key);
    
    // Create kernels
    matmul_kernel_ = ctx_mgr.getKernel(program_, "matmul");
    matvec_kernel_ = ctx_mgr.getKernel(program_, "matvec");
}

void LinearLayer::initializeWeights(const std::vector<float>& weights, const std::vector<float>& bias) {
    if (weights.size() != static_cast<size_t>(output_dim_ * input_dim_)) {
        throw std::runtime_error("Invalid linear layer weights size");
    }
    
    if (has_bias_) {
        if (bias.size() != static_cast<size_t>(output_dim_)) {
            throw std::runtime_error("Invalid linear layer bias size");
        }
    }
    
    // Transpose weights: MLX exports as (output_dim, input_dim) but OpenCL kernel expects [input_dim, output_dim]
    // The matmul kernel does: b_idx = i * output_dim + col where i is input_dim
    // So B is stored as [input_dim, output_dim]
    std::vector<float> transposed_weights(input_dim_ * output_dim_);
    for (int i = 0; i < input_dim_; ++i) {
        for (int j = 0; j < output_dim_; ++j) {
            // MLX format: weights[j * input_dim_ + i] is element at (output_dim=j, input_dim=i)
            // OpenCL format: transposed_weights[i * output_dim_ + j] should be element at (input_dim=i, output_dim=j)
            transposed_weights[i * output_dim_ + j] = weights[j * input_dim_ + i];
        }
    }
    
    cl_context context = ctx_->getContext();
    cl_int err;
    
    // Check device capabilities before creating buffer
    size_t buffer_size = transposed_weights.size() * sizeof(float);
    cl_device_id device = ctx_->getDevice();
    
    // Get device memory info
    size_t max_alloc_size = 0;
    cl_ulong global_mem_size = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &max_alloc_size, nullptr);
    clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, nullptr);
    
    size_t buffer_size_mb = buffer_size / (1024 * 1024);
    size_t max_alloc_mb = max_alloc_size / (1024 * 1024);
    size_t global_mem_mb = global_mem_size / (1024 * 1024);
    
    if (err == CL_SUCCESS && buffer_size > max_alloc_size) {
        throw std::runtime_error(
            "Linear layer weights buffer (" + std::to_string(buffer_size_mb) + " MB) "
            "exceeds device max allocation size (" + std::to_string(max_alloc_mb) + " MB). "
            "Layer dimensions: " + std::to_string(input_dim_) + "x" + std::to_string(output_dim_) + ". "
            "Device global memory: " + std::to_string(global_mem_mb) + " MB"
        );
    }
    
    // Warn if buffer is getting large
    if (buffer_size_mb > 30) {
        std::cerr << "\n    [Linear] Warning: Creating large buffer (" << buffer_size_mb << " MB, "
                  << input_dim_ << "x" << output_dim_ << "). "
                  << "Device max alloc: " << max_alloc_mb << " MB, "
                  << "Global mem: " << global_mem_mb << " MB" << std::flush;
    }
    
    weights_buffer_ = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        buffer_size,
        (void*)transposed_weights.data(),
        &err
    );
    
    if (err != CL_SUCCESS || !weights_buffer_) {
        std::string error_msg = "Failed to create linear layer weights buffer (";
        error_msg += std::to_string(buffer_size_mb) + " MB, ";
        error_msg += std::to_string(input_dim_) + "x" + std::to_string(output_dim_) + ")";
        error_msg += " - OpenCL error: " + std::to_string(err);
        if (err == CL_INVALID_BUFFER_SIZE) {
            error_msg += " (CL_INVALID_BUFFER_SIZE - buffer too large or device memory exhausted)";
            error_msg += "\n    Device limits: max_alloc=" + std::to_string(max_alloc_mb) + " MB, ";
            error_msg += "global_mem=" + std::to_string(global_mem_mb) + " MB";
            error_msg += "\n    This may indicate cumulative memory usage from previous layers has exhausted device memory.";
        } else if (err == CL_OUT_OF_HOST_MEMORY) {
            error_msg += " (CL_OUT_OF_HOST_MEMORY - host memory exhausted)";
        } else if (err == CL_OUT_OF_RESOURCES) {
            error_msg += " (CL_OUT_OF_RESOURCES - device resources exhausted)";
        }
        throw std::runtime_error(error_msg);
    }
    
    if (buffer_size_mb > 30) {
        std::cerr << " ✓" << std::endl;
    }
    
    if (has_bias_ && !bias.empty()) {
        bias_buffer_ = clCreateBuffer(
            context,
            CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
            bias.size() * sizeof(float),
            (void*)bias.data(),
            &err
        );
        
        if (err != CL_SUCCESS || !bias_buffer_) {
            throw std::runtime_error("Failed to create linear layer bias buffer");
        }
    }
    
    weights_initialized_ = true;
}

cl_mem LinearLayer::forward(cl_mem input, int batch_size, int seq_len, cl_command_queue queue) {
    if (!weights_initialized_) {
        throw std::runtime_error("Linear layer weights not initialized");
    }
    
    cl_context context = ctx_->getContext();
    int total_rows = batch_size * seq_len;
    size_t output_size = total_rows * output_dim_ * sizeof(float);
    
    // Allocate output buffer (always recreate to avoid stale handles on some drivers)
    if (output_buffer_) { clReleaseMemObject(output_buffer_); output_buffer_ = nullptr; }
    {
        cl_int buf_err = CL_SUCCESS;
        output_buffer_ = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, nullptr, &buf_err);
        if (buf_err != CL_SUCCESS || !output_buffer_) {
            std::stringstream ss; ss << "Failed to create linear layer output buffer (forward), err=" << buf_err
            << ", size=" << output_size;
            throw std::runtime_error(ss.str());
        }
        output_buffer_size_ = output_size;
    }
    
    // Set kernel arguments for matmul
    // A: input [batch*seq_len, input_dim], B: weights [input_dim, output_dim], C: output [batch*seq_len, output_dim]
    if (!program_ || !matmul_kernel_) {
        buildKernels();
    }
    if (!input || !weights_buffer_ || !output_buffer_) {
        std::stringstream ss; ss << "Invalid buffers in LinearLayer::forward: "
        << " input=" << (input?"ok":"null")
        << " weights=" << (weights_buffer_?"ok":"null")
        << " output=" << (output_buffer_?"ok":"null");
        throw std::runtime_error(ss.str());
    }
    clFinish(queue);
    cl_int err;
    err = clSetKernelArg(matmul_kernel_, 0, sizeof(cl_mem), &input);
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 0 (input) err=") + std::to_string(err)); }
    err = clSetKernelArg(matmul_kernel_, 1, sizeof(cl_mem), &weights_buffer_);
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 1 (weights) err=") + std::to_string(err)); }
    err = clSetKernelArg(matmul_kernel_, 2, sizeof(cl_mem), &output_buffer_);
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 2 (output) err=") + std::to_string(err)); }
    err = clSetKernelArg(matmul_kernel_, 3, sizeof(int), &total_rows);  // m
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 3 (m) err=") + std::to_string(err)); }
    err = clSetKernelArg(matmul_kernel_, 4, sizeof(int), &input_dim_);  // k
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 4 (k) err=") + std::to_string(err)); }
    err = clSetKernelArg(matmul_kernel_, 5, sizeof(int), &output_dim_); // n
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 5 (n) err=") + std::to_string(err)); }
    err = clSetKernelArg(matmul_kernel_, 6, sizeof(int), &batch_size);   // batch_size (1 for flattened view)
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 6 (batch_size) err=") + std::to_string(err)); }
    
    // Execute kernel
    size_t global_size[2] = {static_cast<size_t>(total_rows), static_cast<size_t>(output_dim_)};
    
    // Get device max work group size and adjust local_size accordingly
    size_t max_work_group_size = 0;
    cl_device_id device = ctx_->getDevice();
    clGetKernelWorkGroupInfo(matmul_kernel_, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);
    
    // Use NULL local_size to let OpenCL choose, or use smaller local_size if device doesn't support 16x16
    size_t* local_size_ptr = nullptr;
    size_t local_size[2] = {16, 16};
    
    // Check if device supports the desired local size
    if (max_work_group_size > 0 && max_work_group_size < 256) {
        // Device has small max work group size, use smaller local size or NULL
        local_size_ptr = nullptr; // Let OpenCL choose
    }
    
    err = clEnqueueNDRangeKernel(queue, matmul_kernel_, 2, nullptr, global_size, local_size_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::string error_msg = "Failed to enqueue matmul kernel - OpenCL error: " + std::to_string(err);
        error_msg += "\n    Global size: [" + std::to_string(global_size[0]) + ", " + std::to_string(global_size[1]) + "]";
        error_msg += "\n    Layer: " + std::to_string(input_dim_) + "x" + std::to_string(output_dim_);
        error_msg += "\n    Batch size: " + std::to_string(batch_size) + ", Seq len: " + std::to_string(seq_len);
        if (err == CL_INVALID_WORK_GROUP_SIZE) {
            error_msg += "\n    Device max work group size: " + std::to_string(max_work_group_size);
            error_msg += "\n    This indicates the local work group size is incompatible with the device.";
        }
        throw std::runtime_error(error_msg);
    }
    
    // Add bias if present
    if (has_bias_ && bias_buffer_) {
        // TODO: Implement bias addition kernel
        // For now, bias is added in CPU fallback if needed
    }
    
    // Retain buffer before returning so it survives when LinearLayer releases it on next call
    clRetainMemObject(output_buffer_);
    return output_buffer_;
}

cl_mem LinearLayer::step(cl_mem input, int batch_size, cl_command_queue queue) {
    if (!weights_initialized_) {
        throw std::runtime_error("Linear layer weights not initialized");
    }
    
    cl_context context = ctx_->getContext();
    size_t output_size = batch_size * output_dim_ * sizeof(float);
    
    // Allocate output buffer (always recreate to avoid stale handles)
    if (output_buffer_) { clReleaseMemObject(output_buffer_); output_buffer_ = nullptr; }
    {
        cl_int buf_err = CL_SUCCESS;
        output_buffer_ = clCreateBuffer(context, CL_MEM_READ_WRITE, output_size, nullptr, &buf_err);
        if (buf_err != CL_SUCCESS || !output_buffer_) {
            std::stringstream ss; ss << "Failed to create linear layer output buffer (step), err=" << buf_err
            << ", size=" << output_size;
            throw std::runtime_error(ss.str());
        }
        output_buffer_size_ = output_size;
    }
    
    // Use matvec kernel for single token
    // For each batch element, compute: output = weights @ input
    // Since we have multiple batch elements, we'll use matmul with m=batch_size
    
    if (!program_ || !matmul_kernel_) {
        buildKernels();
    }
    if (!input || !weights_buffer_ || !output_buffer_) {
        std::stringstream ss; ss << "Invalid buffers in LinearLayer::step: "
        << " input=" << (input?"ok":"null")
        << " weights=" << (weights_buffer_?"ok":"null")
        << " output=" << (output_buffer_?"ok":"null");
        throw std::runtime_error(ss.str());
    }
    clFinish(queue);
    cl_int err;
    err = clSetKernelArg(matmul_kernel_, 0, sizeof(cl_mem), &input);
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 0 (input) err=") + std::to_string(err)); }
    err = clSetKernelArg(matmul_kernel_, 1, sizeof(cl_mem), &weights_buffer_);
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 1 (weights) err=") + std::to_string(err)); }
    err = clSetKernelArg(matmul_kernel_, 2, sizeof(cl_mem), &output_buffer_);
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 2 (output) err=") + std::to_string(err)); }
    err = clSetKernelArg(matmul_kernel_, 3, sizeof(int), &batch_size);   // m
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 3 (m) err=") + std::to_string(err)); }
    err = clSetKernelArg(matmul_kernel_, 4, sizeof(int), &input_dim_);   // k
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 4 (k) err=") + std::to_string(err)); }
    err = clSetKernelArg(matmul_kernel_, 5, sizeof(int), &output_dim_);  // n
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 5 (n) err=") + std::to_string(err)); }
    err = clSetKernelArg(matmul_kernel_, 6, sizeof(int), &batch_size);   // batch_size (already accounted)
    if (err != CL_SUCCESS) { throw std::runtime_error(std::string("Failed to set arg 6 (batch_size) err=") + std::to_string(err)); }
    
    size_t global_size[2] = {static_cast<size_t>(batch_size), static_cast<size_t>(output_dim_)};
    
    // Get device max work group size
    size_t max_work_group_size = 0;
    cl_device_id device = ctx_->getDevice();
    clGetKernelWorkGroupInfo(matmul_kernel_, device, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, nullptr);
    
    // Use NULL local_size to let OpenCL choose appropriate size
    size_t* local_size_ptr = nullptr;
    
    err = clEnqueueNDRangeKernel(queue, matmul_kernel_, 2, nullptr, global_size, local_size_ptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::string error_msg = "Failed to enqueue matmul kernel (step) - OpenCL error: " + std::to_string(err);
        error_msg += "\n    Global size: [" + std::to_string(global_size[0]) + ", " + std::to_string(global_size[1]) + "]";
        error_msg += "\n    Layer: " + std::to_string(input_dim_) + "x" + std::to_string(output_dim_);
        if (err == CL_INVALID_WORK_GROUP_SIZE) {
            error_msg += "\n    Device max work group size: " + std::to_string(max_work_group_size);
        }
        throw std::runtime_error(error_msg);
    }
    
    // Retain buffer before returning so it survives when LinearLayer releases it on next call
    clRetainMemObject(output_buffer_);
    return output_buffer_;
}

} // namespace cartesia_opencl

