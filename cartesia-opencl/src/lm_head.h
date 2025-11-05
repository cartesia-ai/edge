#pragma once

#include <vector>
#include <memory>
#include <CL/cl.h>

namespace cartesia_opencl {

class OpenCLContextManager;

/**
 * Language Model Head - final linear layer from d_model to vocab_size.
 */
class LMHead {
public:
    LMHead(OpenCLContextManager* ctx, int d_model, int vocab_size);
    ~LMHead();

    // Initialize with weights (vocab_size x d_model)
    // Note: weights are transposed for efficient matrix multiplication
    void initializeWeights(const std::vector<float>& weights);

    // Forward pass
    // Input: hidden_states [batch_size, d_model]
    // Output: logits [batch_size, vocab_size]
    cl_mem forward(cl_mem hidden_states, int batch_size, cl_command_queue queue);

    int getDModel() const { return d_model_; }
    int getVocabSize() const { return vocab_size_; }

private:
    OpenCLContextManager* ctx_;
    int d_model_;
    int vocab_size_;
    
    // Weights buffers split into chunks to fit device memory limits
    std::vector<cl_mem> weights_buffers_;  // Multiple chunks
    int chunk_size_;                        // Vocab tokens per chunk
    int num_chunks_;                        // Number of chunks
    bool weights_initialized_;
    
    // Output buffer
    cl_mem output_buffer_;
    size_t output_buffer_size_;
};

} // namespace cartesia_opencl

