#pragma once

#include "../residual_block.h"
#include "linear_layer.h"
#include "rms_norm_layer.h"
#include <CL/cl.h>
#include <memory>

namespace cartesia_opencl {

class OpenCLContextManager;
class LinearLayer;
class RMSNormLayer;

/**
 * SwiGLU (Swish-Gated Linear Unit) feedforward layer.
 * Stateless layer - no state needed.
 */
class SwiGLULayer : public Layer {
public:
    SwiGLULayer(OpenCLContextManager* ctx, int d_model, int expand = 2);
    ~SwiGLULayer();
    
    // Initialize weights
    // Weights: gate_proj [d_inner, d_model], up_proj [d_inner, d_model], down_proj [d_model, d_inner]
    void initializeWeights(
        const std::vector<float>& gate_weights,
        const std::vector<float>& up_weights,
        const std::vector<float>& down_weights
    );
    
    // Layer interface
    cl_mem forward(
        cl_mem input,
        int batch_size,
        int seq_len,
        LayerState* state,
        cl_command_queue queue
    ) override;
    
    cl_mem step(
        cl_mem input,
        int batch_size,
        LayerState* state,
        cl_command_queue queue
    ) override;
    
    bool isStateful() const override { return false; }
    
    int getDModel() const { return d_model_; }
    int getDInner() const { return d_inner_; }

private:
    OpenCLContextManager* ctx_;
    int d_model_;
    int d_inner_;  // d_model * expand
    
    bool weights_initialized_;
    
    // Linear layers
    std::unique_ptr<LinearLayer> gate_layer_;
    std::unique_ptr<LinearLayer> up_layer_;
    std::unique_ptr<LinearLayer> down_layer_;
    
    // RMS norm layer (applied after GLU combine, before out_proj)
    std::unique_ptr<RMSNormLayer> norm_layer_;
    
    // Swish kernel
    cl_program program_;
    cl_kernel swish_kernel_;
    cl_kernel combine_kernel_;
    
    // Temporary buffers
    cl_mem swish_output_;   // [batch_size, seq_len, d_inner]
    cl_mem intermediate_;   // [batch_size, seq_len, d_inner]
    size_t swish_output_size_;
    size_t intermediate_size_;
    
    // Step buffers (for single token)
    cl_mem swish_output_step_;  // [batch_size, d_inner]
    cl_mem intermediate_step_;  // [batch_size, d_inner]
    size_t swish_output_step_size_;
    size_t intermediate_step_size_;
    
    // Helper: apply Swish activation (SiLU)
    cl_mem applySwish(cl_mem input, int batch_size, int seq_len, cl_command_queue queue);
    cl_mem applySwishStep(cl_mem input, int batch_size, cl_command_queue queue);
    
    void buildKernels();
};

} // namespace cartesia_opencl

