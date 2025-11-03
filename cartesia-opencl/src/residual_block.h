#pragma once

#include <CL/cl.h>
#include <string>
#include <memory>
#include <vector>

namespace cartesia_opencl {

// Forward declarations
class OpenCLContextManager;

/**
 * Layer state type - can be null for stateless layers
 */
struct LayerState {
    // For SSD layers: (conv_state, ssm_state)
    // For Attention layers: (keys, values, mask_state)
    cl_mem state1;  // First state buffer
    cl_mem state2;  // Second state buffer (optional)
    cl_mem state3;  // Third state buffer (optional, for attention mask)
    
    bool is_null() const { return state1 == nullptr; }
    
    static LayerState null() { return {nullptr, nullptr, nullptr}; }
};

/**
 * Base interface for layers used in residual blocks
 */
class Layer {
public:
    virtual ~Layer() = default;
    
    // Forward pass for prefill (batch processing)
    // Returns output tensor and optionally state
    virtual cl_mem forward(
        cl_mem input,
        int batch_size,
        int seq_len,
        LayerState* state,
        cl_command_queue queue
    ) = 0;
    
    // Step function for autoregressive generation (single token)
    // Returns output tensor and updated state
    virtual cl_mem step(
        cl_mem input,
        int batch_size,
        LayerState* state,
        cl_command_queue queue
    ) = 0;
    
    virtual bool isStateful() const = 0;
};

// Forward declaration
class RMSNormLayer;

/**
 * Residual block wrapper around a layer.
 * Handles normalization and residual connections.
 */
class ResidualBlock {
public:
    ResidualBlock(
        OpenCLContextManager* ctx,
        Layer* layer,
        int d_model,
        const std::string& norm_point = "pre",  // "pre", "post", "pre_resid", or ""
        bool stateful = true
    );
    
    ~ResidualBlock();
    
    // Forward pass
    // Returns output and optionally state if layer is stateful
    cl_mem forward(
        cl_mem input,
        int batch_size,
        int seq_len,
        LayerState* state,
        cl_command_queue queue
    );
    
    // Step function
    cl_mem step(
        cl_mem input,
        int batch_size,
        LayerState* state,
        cl_command_queue queue
    );
    
    bool isStateful() const { return stateful_; }
    
private:
    OpenCLContextManager* ctx_;
    Layer* layer_;
    int d_model_;
    std::string norm_point_;
    bool stateful_;
    
    // RMS norm layer (if needed)
    std::unique_ptr<RMSNormLayer> norm_layer_;
    std::vector<float> norm_weights_;  // Will be initialized with ones
    
    // Helper: apply RMS normalization
    cl_mem applyNorm(cl_mem input, int batch_size, int seq_len, cl_command_queue queue);
    cl_mem applyNormStep(cl_mem input, int batch_size, cl_command_queue queue);
};

} // namespace cartesia_opencl

