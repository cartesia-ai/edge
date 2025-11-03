#include "weights.h"
#include <random>
#include <cmath>

namespace cartesia_opencl {

ModelWeights initializeTestWeights(int d_model, int vocab_size, int n_layers) {
    ModelWeights weights;
    
    // Initialize with small random values using fixed seed for reproducibility
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
    
    // Embedding weights [vocab_size, d_model]
    weights.embedding_weights.resize(vocab_size * d_model);
    for (float& w : weights.embedding_weights) {
        w = dist(gen);
    }
    
    // Layer weights
    weights.layer_weights.resize(n_layers);
    
    for (auto& layer : weights.layer_weights) {
        // Initialize all weight types (some won't be used depending on layer type)
        
        // SSD layer weights (minimal example)
        int d_inner = d_model * 2;  // expand = 2
        int n_heads = d_inner / 64;
        int n_groups = 1;
        int d_state = 64;
        int kernel_size = 4;
        int in_proj_dim = 2 * d_inner + 2 * d_state * n_groups + n_heads;
        int conv_dim = d_inner + 2 * d_state * n_groups;
        
        layer.in_proj_weights.resize(in_proj_dim * d_model);
        layer.conv_weight.resize(conv_dim * kernel_size);
        layer.conv_bias.resize(conv_dim);
        layer.A.resize(n_heads);
        layer.dt_bias.resize(n_heads);
        layer.D.resize(n_heads);
        layer.out_proj_weights.resize(d_model * d_inner);
        
        for (float& w : layer.in_proj_weights) w = dist(gen);
        for (float& w : layer.conv_weight) w = dist(gen);
        for (float& w : layer.conv_bias) w = dist(gen) * 0.01f;
        for (float& w : layer.A) w = dist(gen) * 10.0f + 5.0f;  // Positive values
        for (float& w : layer.dt_bias) w = dist(gen) * 0.01f;
        for (float& w : layer.D) w = dist(gen) * 0.01f + 1.0f;  // Around 1.0
        
        // SwiGLU layer weights
        layer.gate_weights.resize(d_inner * d_model);
        layer.up_weights.resize(d_inner * d_model);
        layer.down_weights.resize(d_model * d_inner);
        for (float& w : layer.gate_weights) w = dist(gen);
        for (float& w : layer.up_weights) w = dist(gen);
        for (float& w : layer.down_weights) w = dist(gen);
        
        // Attention layer weights
        int n_attn_heads = 16;
        int kv_heads = 1;
        int d_head = 128;
        int d_proj = (n_attn_heads + 2 * kv_heads) * d_head;
        
        layer.qkv_weights.resize(d_proj * d_model);
        layer.out_weights.resize(d_model * d_proj);
        for (float& w : layer.qkv_weights) w = dist(gen);
        for (float& w : layer.out_weights) w = dist(gen);
        
        // Norm weights (for layers with normalization)
        layer.norm_weights.resize(d_model);
        for (float& w : layer.norm_weights) {
            w = 1.0f + dist(gen) * 0.1f;  // Around 1.0
        }
    }
    
    // Sequence model post-norm weights
    weights.sequence_norm_weights.resize(d_model);
    for (float& w : weights.sequence_norm_weights) {
        w = 1.0f + dist(gen) * 0.1f;
    }
    
    // LM head weights [vocab_size, d_model]
    weights.lm_head_weights.resize(vocab_size * d_model);
    for (float& w : weights.lm_head_weights) {
        w = dist(gen);
    }
    
    return weights;
}

} // namespace cartesia_opencl

