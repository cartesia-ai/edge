#pragma once

#include "model_config.h"
#include <vector>

namespace cartesia_opencl {

/**
 * Model weights structure - holds all weights for the Rene model.
 * Initially will have hardcoded small test weights.
 */
struct ModelWeights {
    // Embedding weights [vocab_size, d_model]
    std::vector<float> embedding_weights;
    
    // Per-layer weights (48 layers total)
    struct LayerWeights {
        // For SSD layers
        std::vector<float> in_proj_weights;
        std::vector<float> conv_weight;
        std::vector<float> conv_bias;
        std::vector<float> A;
        std::vector<float> dt_bias;
        std::vector<float> D;
        std::vector<float> out_proj_weights;
        
        // For SwiGLU layers
        std::vector<float> gate_weights;
        std::vector<float> up_weights;
        std::vector<float> down_weights;
        
        // For Attention layers
        std::vector<float> qkv_weights;
        std::vector<float> out_weights;
        
        // RMS norm weights (if layer has norm)
        std::vector<float> norm_weights;
    };
    
    std::vector<LayerWeights> layer_weights;
    
    // Sequence model post-norm weights [d_model]
    std::vector<float> sequence_norm_weights;
    
    // LM head weights [vocab_size, d_model]
    std::vector<float> lm_head_weights;
};

/**
 * Initialize model weights with small random values for testing.
 * Uses minimal dimensions for initial testing.
 */
ModelWeights initializeTestWeights(
    int d_model = 64,      // Small for testing
    int vocab_size = 1000, // Small for testing
    int n_layers = 1       // Just 1 layer for initial test
);

/**
 * Load weights from file (future extension)
 */
// ModelWeights loadWeightsFromFile(const std::string& path);

} // namespace cartesia_opencl

