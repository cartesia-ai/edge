#pragma once

#include "model_config.h"
#include "residual_block.h"
#include <vector>
#include <memory>
#include <CL/cl.h>

namespace cartesia_opencl {

// Forward declarations
class RMSNormLayer;
class OpenCLContextManager;

/**
 * Sequence model - manages all layers in sequence.
 * Handles forward (prefill) and step (autoregressive) passes.
 */
class SequenceModel {
public:
    SequenceModel(OpenCLContextManager* ctx, int d_model, int n_layer_repeats, bool post_norm = true);
    ~SequenceModel();
    
    // Add a layer to the sequence
    void addLayer(std::unique_ptr<ResidualBlock> layer);
    
    // Forward pass (prefill) - processes entire sequence
    // Input: [batch_size, seq_len, d_model]
    // Output: [batch_size, seq_len, d_model], state vector
    cl_mem forward(
        cl_mem input,
        int batch_size,
        int seq_len,
        std::vector<LayerState>* state,
        cl_command_queue queue,
        const std::string& output_prefix = ""
    );
    
    // Step function - processes single token
    // Input: [batch_size, d_model]
    // Output: [batch_size, d_model], updated state
    cl_mem step(
        cl_mem input,
        int batch_size,
        std::vector<LayerState>* state,
        cl_command_queue queue,
        const std::string& output_prefix = ""
    );
    
    int getDModel() const { return d_model_; }
    int getNumLayers() const { return layers_.size(); }

private:
    OpenCLContextManager* ctx_;
    int d_model_;
    bool post_norm_;
    
    std::vector<std::unique_ptr<ResidualBlock>> layers_;
    
    // RMS norm for post-norm (if needed)
    std::unique_ptr<RMSNormLayer> norm_layer_;
    bool use_post_norm_;
};

} // namespace cartesia_opencl

