#pragma once

#include "../residual_block.h"
#include "linear_layer.h"
#include <CL/cl.h>
#include <vector>
#include <memory>

namespace cartesia_opencl {

class OpenCLContextManager;
class LinearLayer;

/**
 * Self-attention layer with multi-head attention.
 * Stateful layer - maintains key/value cache.
 */
class AttentionLayer : public Layer {
public:
    AttentionLayer(
        OpenCLContextManager* ctx,
        int d_model,
        int n_heads = 16,
        int kv_heads = 1,  // For multi-query attention
        int d_head = 128,
        int max_context_len = 4096,
        bool causal = true
    );
    
    ~AttentionLayer();
    
    // Initialize weights
    // Weights: qkv_proj [d_proj, d_model], out_proj [d_model, d_proj]
    // where d_proj = (n_heads + 2 * kv_heads) * d_head
    void initializeWeights(
        const std::vector<float>& qkv_weights,
        const std::vector<float>& out_weights
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
    int getNHeads() const { return n_heads_; }
    int getKVHeads() const { return kv_heads_; }
    int getDHead() const { return d_head_; }

private:
    OpenCLContextManager* ctx_;
    int d_model_;
    int n_heads_;
    int kv_heads_;
    int d_head_;
    int max_context_len_;
    bool causal_;
    float softmax_scale_;  // 1.0 / sqrt(d_head)
    int d_proj_;  // (n_heads + 2 * kv_heads) * d_head
    
    // Weight buffers
    cl_mem qkv_weights_;   // [d_proj, d_model]
    cl_mem out_weights_;  // [d_model, d_proj]
    bool weights_initialized_;
    
    // Linear layers
    std::unique_ptr<LinearLayer> qkv_layer_;
    std::unique_ptr<LinearLayer> out_layer_;
    
    // Kernels - split into separate programs to reduce compilation memory
    cl_program split_qkv_program_;
    cl_program reshape_program_;
    cl_program attention_program_;
    cl_program concat_program_;
    
    cl_kernel split_qkv_kernel_;
    cl_kernel reshape_q_kernel_;
    cl_kernel reshape_kv_kernel_;
    cl_kernel attention_kernel_;
    cl_kernel reshape_out_kernel_;
    cl_kernel concatenate_kv_kernel_;
    
    // Temporary buffers for forward pass
    cl_mem qkv_output_;      // [batch_size, seq_len, d_proj]
    size_t qkv_output_size_;
    cl_mem queries_;         // [batch_size, seq_len, n_heads * d_head]
    size_t queries_size_;
    cl_mem keys_;            // [batch_size, seq_len, kv_heads * d_head]
    size_t keys_size_;
    cl_mem values_;          // [batch_size, seq_len, kv_heads * d_head]
    size_t values_size_;
    cl_mem queries_reshaped_; // [batch_size, n_heads, seq_len, d_head]
    size_t queries_reshaped_size_;
    cl_mem keys_reshaped_;    // [batch_size, kv_heads, seq_len, d_head]
    size_t keys_reshaped_size_;
    cl_mem values_reshaped_;  // [batch_size, kv_heads, seq_len, d_head]
    size_t values_reshaped_size_;
    cl_mem keys_concat_;      // Concatenated keys
    size_t keys_concat_size_;
    cl_mem values_concat_;    // Concatenated values
    size_t values_concat_size_;
    cl_mem attn_output_;     // [batch_size, n_heads, seq_len, d_head]
    size_t attn_output_size_;
    cl_mem attn_output_flat_; // [batch_size, seq_len, n_heads * d_head]
    size_t attn_output_flat_size_;

    // Kernel building
    bool kernels_built_;
    bool kernel_build_failed_;
    bool use_cpu_fallback_;  // Use CPU fallback if kernel build fails
    
    void buildKernels();
    void ensureKernelsBuilt();  // Lazy kernel building with error handling
    
    // CPU fallback methods (works with any dimensions, no compilation needed)
    cl_mem forwardCPU(cl_mem input, int batch_size, int seq_len, LayerState* state, cl_command_queue queue);
    cl_mem stepCPU(cl_mem input, int batch_size, LayerState* state, cl_command_queue queue);
};

} // namespace cartesia_opencl

