#pragma once

#include <CL/cl.h>
#include <vector>

namespace cartesia_opencl {

class OpenCLContextManager;

/**
 * RMS Norm layer wrapper.
 * Simple layer that applies RMS normalization.
 */
class RMSNormLayer {
public:
    RMSNormLayer(OpenCLContextManager* ctx, int d_model);
    ~RMSNormLayer();
    
    // Initialize weights [d_model]
    void initializeWeights(const std::vector<float>& weights);
    
    // Forward pass
    // Input/Output: [batch_size, seq_len, d_model]
    cl_mem forward(cl_mem input, int batch_size, int seq_len, cl_command_queue queue);
    
    // Step function
    // Input/Output: [batch_size, d_model]
    cl_mem step(cl_mem input, int batch_size, cl_command_queue queue);
    
    int getDModel() const { return d_model_; }

private:
    OpenCLContextManager* ctx_;
    int d_model_;
    
    cl_program program_;
    cl_kernel kernel_;
    
    cl_mem weights_buffer_;
    bool weights_initialized_;
    
    cl_mem output_buffer_;
    size_t output_buffer_size_;
    
    void buildKernels();
};

} // namespace cartesia_opencl

