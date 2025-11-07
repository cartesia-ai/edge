#include "sampling.h"
#include "debug.h"
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
    // GREEDY_MODE: set to true for deterministic sampling (always picks highest prob)
    const bool GREEDY_MODE = false;
    
    if (GREEDY_MODE) {
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
    
    // Random sampling
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng_);
    
    float cumsum = 0.0f;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumsum += probs[i];
        if (r <= cumsum) {
            return static_cast<int>(i);
        }
    }
    
    // Fallback: return highest probability token
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
    // Match MLX implementation: ascending sort + cumsum > 1-top_p filtering
    auto probs = softmax(logits, temperature);
    
    // Sort probabilities in ASCENDING order (low to high)
    std::vector<size_t> sorted_indices(probs.size());
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    std::sort(sorted_indices.begin(), sorted_indices.end(), [&probs](size_t a, size_t b) {
        return probs[a] < probs[b];
    });
    
    // Compute cumulative sum from lowest to highest probability
    std::vector<float> cumulative_probs(probs.size());
    float cumsum = 0.0f;
    for (size_t i = 0; i < sorted_indices.size(); ++i) {
        cumsum += probs[sorted_indices[i]];
        cumulative_probs[i] = cumsum;
    }
    
    // Select tokens where cumulative_probs > 1 - top_p
    float threshold = 1.0f - top_p;
    std::vector<float> top_probs(probs.size(), 0.0f);
    float sum_selected = 0.0f;
    
    for (size_t i = 0; i < sorted_indices.size(); ++i) {
        if (cumulative_probs[i] > threshold) {
            size_t token_idx = sorted_indices[i];
            top_probs[token_idx] = probs[token_idx];
            sum_selected += probs[token_idx];
        }
    }
    
    // Safety: if no tokens selected, use highest probability token
    if (sum_selected < 1e-10f) {
        auto max_it = std::max_element(probs.begin(), probs.end());
        return std::distance(probs.begin(), max_it);
    }
    
    // Renormalize selected probabilities
    for (float& p : top_probs) {
        p /= sum_selected;
    }
    
    // Create dense array of selected tokens sorted by descending probability
    std::vector<std::pair<int, float>> token_prob_pairs;
    for (size_t i = 0; i < top_probs.size(); ++i) {
        if (top_probs[i] > 0.0f) {
            token_prob_pairs.push_back({i, top_probs[i]});
        }
    }
    
    std::sort(token_prob_pairs.begin(), token_prob_pairs.end(),
              [](const auto& a, const auto& b) { return a.second > b.second; });
    
    std::vector<float> dense_probs;
    std::vector<int> dense_tokens;
    for (const auto& pair : token_prob_pairs) {
        dense_tokens.push_back(pair.first);
        dense_probs.push_back(pair.second);
    }
    
    // Sample from the dense distribution
    int dense_idx = sampleFromProbs(dense_probs);
    return dense_tokens[dense_idx];
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

