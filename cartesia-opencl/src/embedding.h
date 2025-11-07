#pragma once

#include <vector>
#include <memory>
#include <CL/cl.h>

namespace cartesia_opencl {

// Forward declaration
class OpenCLContextManager;

/**
 * Embedding layer that maps token IDs to embedding vectors.
 */
class EmbeddingLayer {
public:
    EmbeddingLayer(OpenCLContextManager* ctx, int vocab_size, int d_model);
    ~EmbeddingLayer();

    // Initialize with weights (vocab_size x d_model)
    void initializeWeights(const std::vector<float>& weights);

    // Encode token IDs to embeddings
    // Input: token_ids [batch_size, seq_len]
    // Output: embeddings [batch_size, seq_len, d_model]
    cl_mem encode(cl_mem token_ids, int batch_size, int seq_len, cl_command_queue queue);

    // Encode single token (for step function)
    // Input: token_id [batch_size] (single int per batch)
    // Output: embeddings [batch_size, d_model]
    cl_mem encodeStep(cl_mem token_ids, int batch_size, cl_command_queue queue);

    int getVocabSize() const { return vocab_size_; }
    int getDModel() const { return d_model_; }

private:
    OpenCLContextManager* ctx_;
    int vocab_size_;
    int d_model_;
    
    // Embedding weights split into chunks to fit device memory limits
    std::vector<cl_mem> weights_buffers_;  // Multiple chunks
    int chunk_size_;                        // Vocab tokens per chunk
    int num_chunks_;                        // Number of chunks
    bool weights_initialized_;
    
    // GPU kernel for embedding lookup
    cl_program program_;
    cl_kernel embedding_kernel_;
    
    // Temporary output buffers
    cl_mem output_buffer_;
    size_t output_buffer_size_;
    
    // Build OpenCL kernels
    void buildKernels();
};

} // namespace cartesia_opencl

