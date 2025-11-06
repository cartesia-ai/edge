#pragma once

#include <iostream>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <CL/cl.h>

namespace cartesia_opencl {
namespace debug {

// Check if debug is enabled at compile time
#ifdef ENABLE_DEBUG
    constexpr bool BUILD_DEBUG_ENABLED = true;
#else
    constexpr bool BUILD_DEBUG_ENABLED = false;
#endif

// Runtime check for environment variables
inline bool isDebugEnabled(const char* env_var) {
    if (!BUILD_DEBUG_ENABLED) {
        return false;
    }
    const char* val = std::getenv(env_var);
    return val != nullptr && (std::strcmp(val, "1") == 0 || std::strcmp(val, "true") == 0 || std::strcmp(val, "TRUE") == 0);
}

// Token debugging
inline bool shouldDebugTokens() {
    return isDebugEnabled("CARTESIA_DEBUG_TOKENS");
}

// Logits debugging
inline bool shouldDebugLogits() {
    return isDebugEnabled("CARTESIA_DEBUG_LOGITS");
}

// Buffer verification debugging
inline bool shouldDebugBuffers() {
    return isDebugEnabled("CARTESIA_DEBUG_BUFFERS");
}

// Compute checksum of a buffer (simple sum for determinism checking)
inline double computeBufferChecksum(const std::vector<float>& data) {
    double sum = 0.0;
    for (size_t i = 0; i < data.size(); ++i) {
        // Use double precision to minimize rounding errors
        sum += static_cast<double>(data[i]);
    }
    return sum;
}

// Compute checksum statistics (sum, min, max, mean) for better diagnostics
struct BufferStats {
    double sum;
    float min;
    float max;
    double mean;
    size_t size;
    size_t nan_count;
    size_t inf_count;
};

inline BufferStats computeBufferStats(const std::vector<float>& data) {
    BufferStats stats;
    stats.size = data.size();
    stats.sum = 0.0;
    stats.min = data.empty() ? 0.0f : data[0];
    stats.max = data.empty() ? 0.0f : data[0];
    stats.nan_count = 0;
    stats.inf_count = 0;
    
    for (size_t i = 0; i < data.size(); ++i) {
        float val = data[i];
        
        if (std::isnan(val)) {
            stats.nan_count++;
            continue;
        }
        if (std::isinf(val)) {
            stats.inf_count++;
            continue;
        }
        
        stats.sum += static_cast<double>(val);
        if (val < stats.min) stats.min = val;
        if (val > stats.max) stats.max = val;
    }
    
    stats.mean = stats.size > 0 ? stats.sum / stats.size : 0.0;
    return stats;
}

// Read OpenCL buffer and compute checksum
inline BufferStats readAndComputeStats(cl_mem buffer, size_t num_elements, cl_command_queue queue) {
    std::vector<float> data(num_elements);
    cl_int err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0,
                                     num_elements * sizeof(float),
                                     data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Warning: Failed to read buffer for checksum (err=" << err << ")" << std::endl;
        return BufferStats{0.0, 0.0f, 0.0f, 0.0, 0, 0, 0};
    }
    return computeBufferStats(data);
}

// Print buffer statistics for debugging
inline void printBufferStats(const std::string& name, const BufferStats& stats) {
    std::cout << "[DEBUG] Buffer: " << name << std::endl;
    std::cout << "  Size: " << stats.size << std::endl;
    std::cout << "  Sum: " << stats.sum << std::endl;
    std::cout << "  Mean: " << stats.mean << std::endl;
    std::cout << "  Min: " << stats.min << std::endl;
    std::cout << "  Max: " << stats.max << std::endl;
    if (stats.nan_count > 0) {
        std::cout << "  NaNs: " << stats.nan_count << std::endl;
    }
    if (stats.inf_count > 0) {
        std::cout << "  Infs: " << stats.inf_count << std::endl;
    }
}

// Compare two buffers for exact equality
inline bool compareBuffers(const std::vector<float>& a, const std::vector<float>& b, 
                          float tolerance = 0.0f, size_t* first_diff_idx = nullptr) {
    if (a.size() != b.size()) {
        return false;
    }
    
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = std::abs(a[i] - b[i]);
        if (diff > tolerance) {
            if (first_diff_idx) {
                *first_diff_idx = i;
            }
            return false;
        }
    }
    return true;
}

} // namespace debug
} // namespace cartesia_opencl

// Debug macros that are completely removed when disabled
#ifdef ENABLE_DEBUG
    // Debug tokens macro
    #define DEBUG_TOKENS(expr) \
        do { \
            if (cartesia_opencl::debug::shouldDebugTokens()) { \
                expr; \
            } \
        } while(0)
    
    // Debug logits macro (first 10 values)
    #define DEBUG_LOGITS(expr) \
        do { \
            if (cartesia_opencl::debug::shouldDebugLogits()) { \
                expr; \
            } \
        } while(0)
    
    // Debug buffers macro (checksums and statistics)
    #define DEBUG_BUFFERS(expr) \
        do { \
            if (cartesia_opencl::debug::shouldDebugBuffers()) { \
                expr; \
            } \
        } while(0)
#else
    // When ENABLE_DEBUG is not defined, these macros are completely removed
    #define DEBUG_TOKENS(expr) ((void)0)
    #define DEBUG_LOGITS(expr) ((void)0)
    #define DEBUG_BUFFERS(expr) ((void)0)
#endif

