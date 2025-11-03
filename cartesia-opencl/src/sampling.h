#pragma once

#include <vector>
#include <random>
#include <CL/cl.h>

namespace cartesia_opencl {

/**
 * Sampling utilities for token generation.
 * These operate on CPU for simplicity, but could be moved to GPU later.
 */
class Sampler {
public:
    Sampler();
    
    /**
     * Categorical sampling with temperature.
     * @param logits Logits vector [vocab_size]
     * @param temperature Sampling temperature (1.0 = no scaling)
     * @return Sampled token ID
     */
    int categoricalSample(const std::vector<float>& logits, float temperature = 1.0f);
    
    /**
     * Top-p (nucleus) sampling with temperature.
     * @param logits Logits vector [vocab_size]
     * @param top_p Cumulative probability threshold
     * @param temperature Sampling temperature
     * @return Sampled token ID
     */
    int topPSample(const std::vector<float>& logits, float top_p = 0.99f, float temperature = 1.0f);
    
    /**
     * Sample from logits stored in OpenCL buffer.
     * Reads buffer to CPU, then samples.
     * @param logits_buffer OpenCL buffer containing logits [vocab_size]
     * @param vocab_size Vocabulary size
     * @param queue Command queue
     * @param top_p Top-p value (if > 0, use top-p sampling, else categorical)
     * @param temperature Sampling temperature
     * @return Sampled token ID
     */
    int sampleFromBuffer(
        cl_mem logits_buffer,
        int vocab_size,
        cl_command_queue queue,
        float top_p = 0.0f,
        float temperature = 1.0f
    );

private:
    std::mt19937 rng_;
    
    // Helper: apply temperature and softmax
    std::vector<float> softmax(const std::vector<float>& logits, float temperature);
    
    // Helper: sample from probability distribution
    int sampleFromProbs(const std::vector<float>& probs);
};

} // namespace cartesia_opencl

