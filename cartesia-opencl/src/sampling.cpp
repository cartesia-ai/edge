#include "sampling.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <CL/cl.h>
#include <vector>
#include <random>
#include <iostream>

namespace cartesia_opencl {

Sampler::Sampler() : rng_(42) {}  // Use fixed seed for deterministic comparison with MLX

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
    
    // Fallback: if cumsum didn't reach r due to floating point errors,
    // find the token with highest probability instead of always returning the last token
    size_t max_idx = 0;
    float max_prob = probs[0];
    for (size_t i = 1; i < probs.size(); ++i) {
        if (probs[i] > max_prob) {
            max_prob = probs[i];
            max_idx = i;
        }
    }
    return static_cast<int>(max_idx);
}

int Sampler::categoricalSample(const std::vector<float>& logits, float temperature) {
    // Match MLX: no clipping, use logits directly
    auto probs = softmax(logits, temperature);
    return sampleFromProbs(probs);
}

int Sampler::topPSample(const std::vector<float>& logits, float top_p, float temperature) {
    // Match MLX: no clipping, use logits directly (MLX handles NaN/Inf internally)
    auto probs = softmax(logits, temperature);
    
    // Create indices and sort by probability (descending)
    std::vector<size_t> indices(probs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&probs](size_t a, size_t b) {
        return probs[a] > probs[b];
    });
    
    // Find top-p cumulative probability
    // Ensure at least one token is selected (cutoff >= 1)
    float cumsum = 0.0f;
    size_t cutoff = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        cumsum += probs[indices[i]];
        cutoff = i + 1;
        if (cumsum >= top_p) {
            break;
        }
    }
    
    // Ensure at least one token is selected
    if (cutoff == 0) {
        cutoff = 1;
    }
    
    // Renormalize probabilities for top-p tokens
    float sum_top_p = 0.0f;
    std::vector<float> filtered_probs(probs.size(), 0.0f);
    for (size_t i = 0; i < cutoff; ++i) {
        filtered_probs[indices[i]] = probs[indices[i]];
        sum_top_p += probs[indices[i]];
    }
    
    // Safety check: if sum is zero or very small, use uniform distribution over selected tokens
    if (sum_top_p < 1e-10f) {
        // This shouldn't happen, but if it does, use uniform over top tokens
        float uniform_prob = 1.0f / cutoff;
        for (size_t i = 0; i < cutoff; ++i) {
            filtered_probs[indices[i]] = uniform_prob;
        }
    } else {
        // Renormalize
        for (float& p : filtered_probs) {
            p /= sum_top_p;
        }
    }
    
    // Match MLX: MLX uses mx.random.categorical(mx.log(top_probs))
    // This is equivalent to sampling from normalized probabilities, but MLX uses log-space for numerical stability
    // We'll use the standard categorical sampling from normalized probabilities (equivalent result)
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
    
    // Debug: Check logits statistics (first time only)
    static bool first_sample = true;
    if (first_sample) {
        float min_logit = logits[0], max_logit = logits[0], sum_logit = 0.0f;
        for (float l : logits) {
            min_logit = std::min(min_logit, l);
            max_logit = std::max(max_logit, l);
            sum_logit += l;
        }
        float mean_logit = sum_logit / logits.size();
        std::cout << "  [Sampling Debug] logits: min=" << min_logit 
                  << ", max=" << max_logit << ", mean=" << mean_logit << std::endl;
        std::cout << "  [Sampling Debug] first 10 logits: ";
        for (int i = 0; i < 10 && i < vocab_size; ++i) {
            std::cout << logits[i] << " ";
        }
        std::cout << std::endl;
        first_sample = false;
    }
    
    // Sample
    if (top_p > 0.0f && top_p < 1.0f) {
        return topPSample(logits, top_p, temperature);
    } else {
        return categoricalSample(logits, temperature);
    }
}

} // namespace cartesia_opencl

