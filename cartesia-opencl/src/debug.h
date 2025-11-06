#pragma once

#include <iostream>
#include <cstdlib>
#include <cstring>

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
#else
    // When ENABLE_DEBUG is not defined, these macros are completely removed
    #define DEBUG_TOKENS(expr) ((void)0)
    #define DEBUG_LOGITS(expr) ((void)0)
#endif

