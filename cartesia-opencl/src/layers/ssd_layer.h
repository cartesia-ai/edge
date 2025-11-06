#pragma once

#include "../residual_block.h"
#include "linear_layer.h"
#include "rms_norm_layer.h"
#include <CL/cl.h>
#include <vector>
#include <memory>

namespace cartesia_opencl {

class OpenCLContextManager;
class LinearLayer;
class RMSNormLayer;

/**
 * SSD (State Space Duality) layer - Mamba-2 style layer.
 * Stateful layer - maintains conv_state and ssm_state.
 */
class SSDLayer : public Layer {
public:
    SSDLayer(
        OpenCLContextManager* ctx,
        int d_model,
        int expand = 2,
        int kernel_size = 4,
        int d_state = 64,
        int d_head = 64,
        int n_groups = 1
    );
    
    ~SSDLayer();
    
    // Initialize weights
    // Weights structure:
    // - in_proj weights: [in_proj_dim, d_model] where in_proj_dim = 2*d_inner + 2*d_state*n_groups + n_heads
    // - conv_weight: [conv_dim, kernel_size] where conv_dim = d_inner + 2*d_state*n_groups
    // - conv_bias: [conv_dim]
    // - A: [n_heads] (SSM state matrix)
    // - dt_bias: [n_heads] (time step bias)
    // - D: [n_heads] (skip connection)
    // - out_proj weights: [d_model, d_inner]
    void initializeWeights(
        const std::vector<float>& in_proj_weights,
        const std::vector<float>& conv_weight,
        const std::vector<float>& conv_bias,
        const std::vector<float>& A,
        const std::vector<float>& dt_bias,
        const std::vector<float>& D,
        const std::vector<float>& out_proj_weights,
        const std::vector<float>& rms_norm_weights
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
    
    bool isStateful() const override { return true; }
    
    int getDModel() const { return d_model_; }
    int getDInner() const { return d_inner_; }
    int getDState() const { return d_state_; }
    int getNHeads() const { return n_heads_; }
    int getNGroups() const { return n_groups_; }

private:
    OpenCLContextManager* ctx_;
    int d_model_;
    int d_inner_;       // d_model * expand
    int kernel_size_;
    int d_state_;
    int d_head_;
    int n_groups_;
    int n_heads_;       // d_inner / d_head
    int in_proj_dim_;   // 2*d_inner + 2*d_state*n_groups + n_heads
    int conv_dim_;      // d_inner + 2*d_state*n_groups
    
    // Weight buffers
    cl_mem conv_weight_;
    cl_mem conv_bias_;
    cl_mem A_;           // [n_heads]
    cl_mem dt_bias_;     // [n_heads]
    cl_mem D_;           // [n_heads]
    bool weights_initialized_;
    
    // Linear layers
    std::unique_ptr<LinearLayer> in_proj_layer_;
    std::unique_ptr<LinearLayer> out_proj_layer_;
    std::unique_ptr<RMSNormLayer> norm_layer_;
    
    // Kernels
    cl_program program_;
    cl_kernel ssm_kernel_;
    cl_kernel conv_forward_kernel_;
    cl_kernel conv_update_kernel_;
    
    // Helper kernels
    cl_kernel split_in_proj_kernel_;
    cl_kernel gate_kernel_;
    cl_kernel copy_channels_kernel_;
    
    // SSM forward kernels (for prefill, GPU-based)
    cl_kernel process_dt_kernel_;
    cl_kernel compute_dtA_kernel_;
    cl_kernel compute_segsum_decay_kernel_;
    cl_kernel compute_CB_kernel_;
    cl_kernel compute_ssm_output_kernel_;
    
    void buildKernels();
    
    // Helper: split in_proj output into z, xBC, dt
    void splitInProjOutput(
        cl_mem in_proj_output,
        int batch_size,
        int seq_len,
        cl_mem z_out,
        cl_mem xBC_out,
        cl_mem dt_out,
        cl_command_queue queue
    );
    
    // TODO: Allocate temporary buffers as needed
};

} // namespace cartesia_opencl

