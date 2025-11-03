#pragma once

#include <CL/cl.h>
#include <vector>

namespace cartesia_opencl {

class OpenCLContextManager;

/**
 * Generic linear layer for matrix multiplication.
 * Used by SwiGLU, Attention, and other layers.
 */
class LinearLayer {
public:
    LinearLayer(OpenCLContextManager* ctx, int input_dim, int output_dim, bool has_bias = false);
    ~LinearLayer();
    
    // Initialize weights
    // Weights: [output_dim, input_dim] (row-major)
    // Bias: [output_dim] (optional)
    void initializeWeights(const std::vector<float>& weights, const std::vector<float>& bias = {});
    
    // Forward pass (batched)
    // Input: [batch_size, seq_len, input_dim]
    // Output: [batch_size, seq_len, output_dim]
    cl_mem forward(cl_mem input, int batch_size, int seq_len, cl_command_queue queue);
    
    // Step function (single token)
    // Input: [batch_size, input_dim]
    // Output: [batch_size, output_dim]
    cl_mem step(cl_mem input, int batch_size, cl_command_queue queue);
    
    int getInputDim() const { return input_dim_; }
    int getOutputDim() const { return output_dim_; }

private:
    OpenCLContextManager* ctx_;
    int input_dim_;
    int output_dim_;
    bool has_bias_;
    
    cl_program program_;
    cl_kernel matmul_kernel_;
    cl_kernel matvec_kernel_;
    
    cl_mem weights_buffer_;
    cl_mem bias_buffer_;
    bool weights_initialized_;
    
    cl_mem output_buffer_;
    size_t output_buffer_size_;
    
    void buildKernels();
};

} // namespace cartesia_opencl

