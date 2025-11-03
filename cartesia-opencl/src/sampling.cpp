#include "sampling.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <CL/cl.h>
#include <vector>
#include <random>

namespace cartesia_opencl {

Sampler::Sampler() : rng_(std::random_device{}()) {}

std::vector<float> Sampler::softmax(const std::vector<float>& logits, float temperature) {
    std::vector<float> probs(logits.size());
    
    // Apply temperature and find max for numerical stability
    float max_logit = *std::max_element(logits.begin(), logits.end());
    float sum_exp = 0.0f;
    
    for (size_t i = 0; i < logits.size(); ++i) {
        float scaled = (logits[i] - max_logit) / temperature;
        probs[i] = std::exp(scaled);
        sum_exp += probs[i];
    }
    
    // Normalize
    for (float& p : probs) {
        p /= sum_exp;
    }
    
    return probs;
}

int Sampler::sampleFromProbs(const std::vector<float>& probs) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    
    float cumsum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return static_cast<int>(i);
        }
    }
    
    // Fallback (shouldn't happen)
    return static_cast<int>(probs.size() - 1);
}

int Sampler::categoricalSample(const std::vector<float>& logits, float temperature) {
    auto probs = softmax(logits, temperature);
    return sampleFromProbs(probs);
}

int Sampler::topPSample(const std::vector<float>& logits, float top_p, float temperature) {
    auto probs = softmax(logits, temperature);
    
    // Create indices and sort by probability (descending)
    std::vector<size_t> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&probs](size_t a, size_t b) {
        return probs[a] > probs[b];
    });
    
    // Find top-p cumulative probability
    float cumsum = 0.0f;
    size_t cutoff = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        cumsum += probs[indices[i]];
        cutoff = i + 1;
        if (cumsum >= top_p) {
            break;
        }
    }
    
    // Renormalize probabilities for top-p tokens
    float sum_top_p = 0.0f;
    std::vector<float> filtered_probs(probs.size(), 0.0f);
    for (size_t i = 0; i < cutoff; ++i) {
        filtered_probs[indices[i]] = probs[indices[i]];
        sum_top_p += probs[indices[i]];
    }
    
    // Renormalize
    for (float& p : filtered_probs) {
        p /= sum_top_p;
    }
    
    return sampleFromProbs(filtered_probs);
}

int Sampler::sampleFromBuffer(
    cl_mem logits_buffer,
    int vocab_size,
    cl_command_queue queue,
    float top_p,
    float temperature
) {
    // Read logits from GPU
    std::vector<float> logits(vocab_size);
    cl_int err = clEnqueueReadBuffer(
        queue, logits_buffer, CL_TRUE, 0,
        vocab_size * sizeof(float), logits.data(),
        0, nullptr, nullptr
    );
    
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read logits from buffer");
    }
    
    // Sample
    if (top_p > 0.0f && top_p < 1.0f) {
        return topPSample(logits, top_p, temperature);
    } else {
        return categoricalSample(logits, temperature);
    }
}

} // namespace cartesia_opencl

