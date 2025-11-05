#include "attention_layer.h"
#include "../opencl_context.h"
#include "linear_layer.h"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>
#include <cstdlib>
#include <CL/cl.h>
#include <cmath>
#include <vector>

namespace cartesia_opencl {

namespace {
// Safely update an OpenCL buffer stored in state with retain/release semantics
inline void retainAndAssign(cl_mem& dst, cl_mem src) {
    if (dst && dst != src) {
        clReleaseMemObject(dst);
    }
    if (src) {
        clRetainMemObject(src);
    }
    dst = src;
}
}

AttentionLayer::AttentionLayer(
    OpenCLContextManager* ctx,
    int d_model,
    int n_heads,
    int kv_heads,
    int d_head,
    int max_context_len,
    bool causal
)
    : ctx_(ctx)
    , d_model_(d_model)
    , n_heads_(n_heads)
    , kv_heads_(kv_heads)
    , d_head_(d_head)
    , max_context_len_(max_context_len)
    , causal_(causal)
    , softmax_scale_(1.0f / std::sqrt(static_cast<float>(d_head_)))
    , d_proj_((n_heads + 2 * kv_heads) * d_head)
    , qkv_weights_(nullptr)
    , out_weights_(nullptr)
    , weights_initialized_(false)
    , split_qkv_program_(nullptr)
    , reshape_program_(nullptr)
    , attention_program_(nullptr)
    , concat_program_(nullptr)
    , split_qkv_kernel_(nullptr)
    , reshape_q_kernel_(nullptr)
    , reshape_kv_kernel_(nullptr)
    , attention_kernel_(nullptr)
    , reshape_out_kernel_(nullptr)
    , concatenate_kv_kernel_(nullptr)
    , qkv_output_(nullptr)
    , qkv_output_size_(0)
    , queries_(nullptr)
    , queries_size_(0)
    , keys_(nullptr)
    , keys_size_(0)
    , values_(nullptr)
    , values_size_(0)
    , queries_reshaped_(nullptr)
    , queries_reshaped_size_(0)
    , keys_reshaped_(nullptr)
    , keys_reshaped_size_(0)
    , values_reshaped_(nullptr)
    , values_reshaped_size_(0)
    , keys_concat_(nullptr)
    , keys_concat_size_(0)
    , values_concat_(nullptr)
    , values_concat_size_(0)
    , attn_output_(nullptr)
    , attn_output_size_(0)
    , attn_output_flat_(nullptr)
    , attn_output_flat_size_(0)
    , kernels_built_(false)
    , kernel_build_failed_(false)
    , use_cpu_reshape_(false)
    , use_cpu_fallback_(false)
{
    if (!ctx_) {
        throw std::runtime_error("OpenCLContext is null");
    }
    
    // Debug output
    size_t qkv_params = static_cast<size_t>(d_proj_) * d_model_;
    size_t out_params = static_cast<size_t>(d_model_) * n_heads_ * d_head_;
    size_t total_params = qkv_params + out_params;
    size_t buffer_size_mb = (total_params * sizeof(float)) / (1024 * 1024);
    std::cout << "  [Attention] d_model=" << d_model_
              << ", n_heads=" << n_heads_
              << ", kv_heads=" << kv_heads_
              << ", d_head=" << d_head_
              << ", d_proj=" << d_proj_
              << ", max_context_len=" << max_context_len_
              << ", causal=" << (causal_ ? "true" : "false")
              << ", params=" << total_params
              << " (qkv:" << qkv_params << ", out:" << out_params << ")"
              << ", buffer_size=" << buffer_size_mb << " MB" << std::endl;
    
    // Create linear layers
    std::cout << "    Creating QKV linear layer..." << std::flush;
    qkv_layer_ = std::make_unique<LinearLayer>(ctx_, d_model_, d_proj_, false);
    std::cout << " ✓" << std::endl;
    
    std::cout << "    Creating output linear layer..." << std::flush;
    out_layer_ = std::make_unique<LinearLayer>(ctx_, n_heads_ * d_head_, d_model_, false);
    std::cout << " ✓" << std::endl;
    
    std::cout << "    Building attention kernels..." << std::flush;
    // Use lazy kernel building - kernels will be built on first use
    // This allows model initialization to complete even if kernel building fails
    std::cout << " (deferred to first use)" << std::endl;
    // buildKernels() will be called lazily in ensureKernelsBuilt()
}

AttentionLayer::~AttentionLayer() {
    // Linear layers will clean themselves up
    if (split_qkv_kernel_) clReleaseKernel(split_qkv_kernel_);
    if (reshape_q_kernel_) clReleaseKernel(reshape_q_kernel_);
    if (reshape_kv_kernel_) clReleaseKernel(reshape_kv_kernel_);
    if (attention_kernel_) clReleaseKernel(attention_kernel_);
    if (reshape_out_kernel_) clReleaseKernel(reshape_out_kernel_);
    if (concatenate_kv_kernel_) clReleaseKernel(concatenate_kv_kernel_);
    if (split_qkv_program_) clReleaseProgram(split_qkv_program_);
    if (reshape_program_) clReleaseProgram(reshape_program_);
    if (attention_program_) clReleaseProgram(attention_program_);
    if (concat_program_) clReleaseProgram(concat_program_);
    
    // Release buffers
    if (qkv_output_) clReleaseMemObject(qkv_output_);
    if (queries_) clReleaseMemObject(queries_);
    if (keys_) clReleaseMemObject(keys_);
    if (values_) clReleaseMemObject(values_);
    if (queries_reshaped_) clReleaseMemObject(queries_reshaped_);
    if (keys_reshaped_) clReleaseMemObject(keys_reshaped_);
    if (values_reshaped_) clReleaseMemObject(values_reshaped_);
    if (keys_concat_) clReleaseMemObject(keys_concat_);
    if (values_concat_) clReleaseMemObject(values_concat_);
    if (attn_output_) clReleaseMemObject(attn_output_);
    if (attn_output_flat_) clReleaseMemObject(attn_output_flat_);
}

void AttentionLayer::buildKernels() {
    try {
        auto& ctx_mgr = OpenCLContextManager::getInstance();
        
        // Check environment variable to skip kernel building and use CPU fallback
        const char* skip_kernels = std::getenv("SKIP_ATTENTION_KERNELS");
        if (skip_kernels && std::string(skip_kernels) == "1") {
            std::cout << "\n      [SKIP] Kernel building disabled via SKIP_ATTENTION_KERNELS=1" << std::flush;
            throw std::runtime_error("Kernel building skipped by user");
        }
        
        // Split kernels into separate programs to reduce compilation memory
        // Build them one at a time, releasing resources as we go
        
        // 1. Build split_qkv kernel
        {
            const char* split_qkv_source = R"(
__kernel void split_qkv(
    __global const float* qkv,
    __global float* queries,
    __global float* keys,
    __global float* values,
    const int batch_size,
    const int seq_len,
    const int n_heads,
    const int kv_heads,
    const int d_head
) {
    const int batch_idx = get_global_id(0);
    const int seq_idx = get_global_id(1);
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    int q_dim = n_heads * d_head;
    int kv_dim = kv_heads * d_head;
    int d_proj = q_dim + 2 * kv_dim;
    
    int qkv_base = (batch_idx * seq_len + seq_idx) * d_proj;
    int q_base = (batch_idx * seq_len + seq_idx) * q_dim;
    int k_base = (batch_idx * seq_len + seq_idx) * kv_dim;
    int v_base = (batch_idx * seq_len + seq_idx) * kv_dim;
    
    for (int i = 0; i < q_dim; ++i) queries[q_base + i] = qkv[qkv_base + i];
    for (int i = 0; i < kv_dim; ++i) keys[k_base + i] = qkv[qkv_base + q_dim + i];
    for (int i = 0; i < kv_dim; ++i) values[v_base + i] = qkv[qkv_base + q_dim + kv_dim + i];
}
)";
            std::cout << "\n      Building split_qkv program..." << std::flush;
            std::vector<std::string> sources = {std::string(split_qkv_source)};
            std::string cache_key = ctx_mgr.generateCacheKey(sources) + "_split_qkv";
            split_qkv_program_ = ctx_mgr.buildProgram(sources, cache_key);
            split_qkv_kernel_ = ctx_mgr.getKernel(split_qkv_program_, "split_qkv");
            std::cout << " ✓" << std::flush;
        }
        
        // 2. Build reshape kernels (allow CPU fallback if driver crashes)
        try {
            const char* reshape_source = R"(
// OpenCL supports up to 3D NDRange; loop over d_head inside the kernel
__kernel void reshape_for_attention(
    __global const float* input,
    __global float* output,
    const int batch_size,
    const int seq_len,
    const int n_heads,
    const int d_head
) {
    const int batch_idx = get_global_id(0);
    const int head_idx  = get_global_id(1);
    const int seq_idx   = get_global_id(2);
    if (batch_idx >= batch_size || head_idx >= n_heads || seq_idx >= seq_len) return;
    
    // Loop over head dimension
    for (int d = 0; d < d_head; ++d) {
        int input_idx = batch_idx * seq_len * n_heads * d_head +
                        seq_idx * n_heads * d_head +
                        head_idx * d_head +
                        d;
        int output_idx = batch_idx * n_heads * seq_len * d_head +
                         head_idx * seq_len * d_head +
                         seq_idx * d_head +
                         d;
        output[output_idx] = input[input_idx];
    }
}

__kernel void reshape_from_attention(
    __global const float* input,
    __global float* output,
    const int batch_size,
    const int seq_len,
    const int n_heads,
    const int d_head
) {
    const int batch_idx = get_global_id(0);
    const int seq_idx   = get_global_id(1);
    const int head_idx  = get_global_id(2);
    if (batch_idx >= batch_size || seq_idx >= seq_len || head_idx >= n_heads) return;
    
    for (int d = 0; d < d_head; ++d) {
        int input_idx = batch_idx * n_heads * seq_len * d_head +
                        head_idx * seq_len * d_head +
                        seq_idx * d_head +
                        d;
        int output_idx = batch_idx * seq_len * n_heads * d_head +
                         seq_idx * n_heads * d_head +
                         head_idx * d_head +
                         d;
        output[output_idx] = input[input_idx];
    }
}
)";
            std::cout << "\n      Building reshape program..." << std::flush;
            std::vector<std::string> sources = {std::string(reshape_source)};
            std::string cache_key = ctx_mgr.generateCacheKey(sources) + "_reshape";
            reshape_program_ = ctx_mgr.buildProgram(sources, cache_key);
            reshape_q_kernel_ = ctx_mgr.getKernel(reshape_program_, "reshape_for_attention");
            reshape_kv_kernel_ = ctx_mgr.getKernel(reshape_program_, "reshape_for_attention");
            reshape_out_kernel_ = ctx_mgr.getKernel(reshape_program_, "reshape_from_attention");
            std::cout << " ✓" << std::flush;
        } catch (const std::exception& e) {
            std::cerr << "\n      WARNING: Reshape program build failed, using CPU reshape fallback: " << e.what() << std::endl;
            use_cpu_reshape_ = true;
        }
        
        // 3. Build attention kernel (the complex one - simplified)
        {
            const char* attention_source = R"(
#define MAX_ATTN_SEQ_LEN 16
__kernel void scaled_dot_product_attention(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* output,
    const float scale,
    const int batch_size,
    const int n_heads,
    const int kv_heads,
    const int seq_len_q,
    const int seq_len_kv,
    const int d_head,
    const int causal,
    const int cached_len
) {
    const int batch_idx = get_global_id(0);
    const int head_idx = get_global_id(1);
    const int seq_q_idx = get_global_id(2);
    
    if (batch_idx >= batch_size || head_idx >= n_heads || seq_q_idx >= seq_len_q) return;
    
    float scores[MAX_ATTN_SEQ_LEN];
    float max_score = -FLT_MAX;
    int kv_head_idx = head_idx % kv_heads;
    int actual_seq_len = (seq_len_kv < MAX_ATTN_SEQ_LEN) ? seq_len_kv : MAX_ATTN_SEQ_LEN;
    
    // Compute Q @ K^T
    for (int seq_kv_idx = 0; seq_kv_idx < actual_seq_len; ++seq_kv_idx) {
        // Check causal mask
        // In step mode with cached KV, the query is at position (cached_len + seq_q_idx) in the full sequence
        // So we mask positions > (cached_len + seq_q_idx)
        // For forward pass (cached_len == 0), this reduces to seq_kv_idx > seq_q_idx
        if (causal && seq_kv_idx > (cached_len + seq_q_idx)) {
            scores[seq_kv_idx] = -FLT_MAX;
            continue;
        }
        
        float score = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            int q_idx = batch_idx * n_heads * seq_len_q * d_head +
                       head_idx * seq_len_q * d_head +
                       seq_q_idx * d_head + d;
            int k_idx = batch_idx * kv_heads * seq_len_kv * d_head +
                       kv_head_idx * seq_len_kv * d_head +
                       seq_kv_idx * d_head + d;
            score += Q[q_idx] * K[k_idx];
        }
        scores[seq_kv_idx] = score * scale;
        if (scores[seq_kv_idx] > max_score) max_score = scores[seq_kv_idx];
    }
    
    // Softmax
    float exp_sum = 0.0f;
    for (int seq_kv_idx = 0; seq_kv_idx < actual_seq_len; ++seq_kv_idx) {
        float exp_val = exp(scores[seq_kv_idx] - max_score);
        scores[seq_kv_idx] = exp_val;
        exp_sum += exp_val;
    }
    
    // Compute output
    for (int d = 0; d < d_head; ++d) {
        float val = 0.0f;
        for (int seq_kv_idx = 0; seq_kv_idx < actual_seq_len; ++seq_kv_idx) {
            int v_idx = batch_idx * kv_heads * seq_len_kv * d_head +
                       kv_head_idx * seq_len_kv * d_head +
                       seq_kv_idx * d_head + d;
            val += (scores[seq_kv_idx] / exp_sum) * V[v_idx];
        }
        
        int out_idx = batch_idx * n_heads * seq_len_q * d_head +
                     head_idx * seq_len_q * d_head +
                     seq_q_idx * d_head + d;
        output[out_idx] = val;
    }
}
)";
            std::cout << "\n      Building attention program (simplified)..." << std::flush;
            std::vector<std::string> sources = {std::string(attention_source)};
            std::string cache_key = ctx_mgr.generateCacheKey(sources) + "_attention";
            attention_program_ = ctx_mgr.buildProgram(sources, cache_key);
            attention_kernel_ = ctx_mgr.getKernel(attention_program_, "scaled_dot_product_attention");
            std::cout << " ✓" << std::flush;
        }
        
        // 4. Build concatenate kernel
        {
            const char* concat_source = R"(
// Use 3D NDRange (batch, kv_heads, total_len) and loop over d_head in-kernel
__kernel void concatenate_kv(
    __global const float* cached,
    __global const float* new_kv,
    __global float* output,
    const int batch_size,
    const int kv_heads,
    const int cached_len,
    const int new_len,
    const int d_head
) {
    const int batch_idx = get_global_id(0);
    const int head_idx  = get_global_id(1);
    const int seq_idx   = get_global_id(2);
    if (batch_idx >= batch_size || head_idx >= kv_heads || seq_idx >= (cached_len + new_len)) return;
    
    int total_len = cached_len + new_len;
    for (int d = 0; d < d_head; ++d) {
        int out_idx = batch_idx * kv_heads * total_len * d_head +
                      head_idx * total_len * d_head +
                      seq_idx * d_head + d;
        if (seq_idx < cached_len) {
            int cached_idx = batch_idx * kv_heads * cached_len * d_head +
                             head_idx * cached_len * d_head +
                             seq_idx * d_head + d;
            output[out_idx] = cached[cached_idx];
        } else {
            int new_idx = batch_idx * kv_heads * new_len * d_head +
                          head_idx * new_len * d_head +
                          (seq_idx - cached_len) * d_head + d;
            output[out_idx] = new_kv[new_idx];
        }
    }
}
)";
            std::cout << "\n      Building concatenate program..." << std::flush;
            std::vector<std::string> sources = {std::string(concat_source)};
            std::string cache_key = ctx_mgr.generateCacheKey(sources) + "_concat";
            concat_program_ = ctx_mgr.buildProgram(sources, cache_key);
            concatenate_kv_kernel_ = ctx_mgr.getKernel(concat_program_, "concatenate_kv");
            std::cout << " ✓" << std::flush;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in AttentionLayer::buildKernels(): " << e.what() << std::endl;
        throw;
    }
}

void AttentionLayer::ensureKernelsBuilt() {
    // If kernels are already built, return immediately
    if (kernels_built_) {
        return;
    }
    
    // If build previously failed, use CPU fallback (don't retry)
    if (kernel_build_failed_) {
        use_cpu_fallback_ = true;
        return;  // Don't throw, just use CPU fallback
    }
    
    // Check if we should skip kernel building entirely
    const char* skip_kernels = std::getenv("SKIP_ATTENTION_KERNELS");
    if (skip_kernels && std::string(skip_kernels) == "1") {
        use_cpu_fallback_ = true;
        kernel_build_failed_ = true;
        std::cout << "\n[AttentionLayer] Skipping kernel build (SKIP_ATTENTION_KERNELS=1), using CPU fallback" << std::endl;
        return;
    }
    
    // Try to build kernels with error handling
    std::cerr << "[AttentionLayer] Attempting to build kernels on first use..." << std::endl;
    std::cerr << "[AttentionLayer] To skip kernel building (if it hangs), set: export SKIP_ATTENTION_KERNELS=1" << std::endl;
    try {
        buildKernels();
        kernels_built_ = true;
        std::cerr << "[AttentionLayer] ✓ Kernels built successfully" << std::endl;
    } catch (const std::runtime_error& e) {
        kernel_build_failed_ = true;
        use_cpu_fallback_ = true;  // Enable CPU fallback
        std::cerr << "[AttentionLayer] ✗ Kernel build failed: " << e.what() << std::endl;
        std::cerr << "[AttentionLayer] Falling back to CPU implementation (works with any dimensions)" << std::endl;
        std::cerr << "[AttentionLayer] Note: CPU fallback is slower but allows testing the full pipeline." << std::endl;
        // Don't throw - allow CPU fallback to proceed
    } catch (const std::exception& e) {
        kernel_build_failed_ = true;
        use_cpu_fallback_ = true;
        std::cerr << "[AttentionLayer] ✗ Unexpected error during kernel build: " << e.what() << std::endl;
        std::cerr << "[AttentionLayer] Falling back to CPU implementation" << std::endl;
        // Don't throw - allow CPU fallback
    } catch (...) {
        kernel_build_failed_ = true;
        use_cpu_fallback_ = true;
        std::cerr << "[AttentionLayer] ✗ FATAL: Unknown exception during kernel build (possible driver crash)" << std::endl;
        std::cerr << "[AttentionLayer] The OpenCL driver may have crashed. This is a known issue on some Android devices." << std::endl;
        std::cerr << "[AttentionLayer] Falling back to CPU implementation (slower but functional)" << std::endl;
        // Don't throw - allow CPU fallback to proceed
    }
}

void AttentionLayer::initializeWeights(
    const std::vector<float>& qkv_weights,
    const std::vector<float>& out_weights
) {
    if (qkv_weights.size() != static_cast<size_t>(d_proj_ * d_model_) ||
        out_weights.size() != static_cast<size_t>(d_model_ * n_heads_ * d_head_)) {
        throw std::runtime_error("Invalid attention weights size");
    }
    
    qkv_layer_->initializeWeights(qkv_weights);
    out_layer_->initializeWeights(out_weights);
    
    weights_initialized_ = true;
}

cl_mem AttentionLayer::forward(
    cl_mem input,
    int batch_size,
    int seq_len,
    LayerState* state,
    cl_command_queue queue
) {
    if (!weights_initialized_) {
        throw std::runtime_error("Attention weights not initialized");
    }
    
    // Debug: Check input IMMEDIATELY on entry (first time only)
    static bool checked_input_entry = false;
    static void* last_buffer_ptr = nullptr;
    if (!checked_input_entry) {
        size_t input_size = batch_size * seq_len * d_model_;
        size_t buf_size = 0;
        cl_int info_err = clGetMemObjectInfo(input, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
        
        // Check if this is the same buffer pointer as before
        if (input == last_buffer_ptr && last_buffer_ptr != nullptr) {
            std::cout << "  [Attention Entry Debug] Same buffer pointer as before!" << std::endl;
        } else {
            if (last_buffer_ptr != nullptr) {
                std::cout << "  [Attention Entry Debug] WARNING: Buffer pointer changed! "
                          << "Previous=" << last_buffer_ptr << ", Current=" << input << std::endl;
            } else {
                std::cout << "  [Attention Entry Debug] First check (storing pointer)" << std::endl;
            }
            last_buffer_ptr = input;
        }
        
        // Get buffer info to compare addresses
        cl_uint buffer_mem_type = 0;
        clGetMemObjectInfo(input, CL_MEM_TYPE, sizeof(cl_uint), &buffer_mem_type, nullptr);
        std::cout << "  [Attention Entry Debug] Buffer type=" << buffer_mem_type 
                  << " (CL_MEM_OBJECT_BUFFER=" << CL_MEM_OBJECT_BUFFER << ")" << std::endl;
        
        std::vector<float> input_check(input_size);
        cl_int check_err = clEnqueueReadBuffer(queue, input, CL_TRUE, 0,
            input_size * sizeof(float), input_check.data(), 0, nullptr, nullptr);
        if (check_err == CL_SUCCESS) {
            int nan_count = 0;
            // Check which token positions have NaN
            std::vector<int> nan_per_token(seq_len, 0);
            for (size_t i = 0; i < input_check.size(); ++i) {
                if (std::isnan(input_check[i])) {
                    nan_count++;
                    int token_idx = i / d_model_;
                    if (token_idx < seq_len) {
                        nan_per_token[token_idx]++;
                    }
                }
            }
            
            // Check first few values of token 5 (the clean one) vs token 0 (NaN)
            std::cout << "  [Attention Entry Debug] Input at function entry: " << nan_count 
                      << " NaNs out of " << input_size << " values, buffer_size=" << buf_size << std::endl;
            std::cout << "  [Attention Entry Debug] NaNs per token: ";
            for (int i = 0; i < seq_len; ++i) {
                std::cout << "token" << i << "=" << nan_per_token[i] << "/" << d_model_ << " ";
            }
            std::cout << std::endl;
            
            // Show sample values from token 0 and token 5
            std::cout << "  [Attention Entry Debug] Token 0 first 5 values: ";
            for (int i = 0; i < 5; ++i) {
                std::cout << input_check[i] << " ";
            }
            std::cout << std::endl;
            std::cout << "  [Attention Entry Debug] Token 5 first 5 values: ";
            for (int i = 5 * d_model_; i < 5 * d_model_ + 5; ++i) {
                std::cout << input_check[i] << " ";
            }
            std::cout << std::endl;
            
            // If these values match what ResidualBlock saw, then it's the same buffer content
            // This will help diagnose if the buffer was corrupted or if we're reading from wrong place
            bool token5_matches = true;
            for (int i = 5 * d_model_; i < 5 * d_model_ + 5; ++i) {
                if (std::isnan(input_check[i])) {
                    token5_matches = false;
                    break;
                }
            }
            std::cout << "  [Attention Entry Debug] Token 5 is " 
                      << (token5_matches ? "VALID (non-NaN)" : "CORRUPTED (has NaN)") << std::endl;
            
            // Check if token 5 matches what we expect (should be the same as ResidualBlock saw)
            // This will help us understand if the buffer content actually changed or if it's a read issue
            bool token5_all_nan_in_attention = true;
            for (int i = 5 * d_model_; i < 5 * d_model_ + d_model_; ++i) {
                if (!std::isnan(input_check[i])) {
                    token5_all_nan_in_attention = false;
                    break;
                }
            }
            std::cout << "  [Attention Entry Debug] Token 5 is " 
                      << (token5_all_nan_in_attention ? "ALL NaN" : "has valid values") << std::endl;
        } else {
            std::cout << "  [Attention Entry Debug] Failed to read buffer, err=" << check_err << std::endl;
        }
        checked_input_entry = true;
    }
    
    // Optional runtime override to force CPU fallback regardless of kernel build state
    if (const char* force_cpu = std::getenv("FORCE_ATTENTION_CPU")) {
        if (std::string(force_cpu) == "1") {
            return forwardCPU(input, batch_size, seq_len, state, queue);
        }
    }

    // Try to ensure kernels are built (lazy initialization)
    // If build fails, use CPU fallback
    try {
        ensureKernelsBuilt();
    } catch (...) {
        // Build failed, use CPU fallback
        use_cpu_fallback_ = true;
    }
    
    // Use CPU fallback if kernels failed to build
    if (use_cpu_fallback_ || !kernels_built_) {
        return forwardCPU(input, batch_size, seq_len, state, queue);
    }
    
    cl_context context = ctx_->getContext();
    cl_int err;
    
    // Allocate temporary buffers
    size_t qkv_size = batch_size * seq_len * d_proj_ * sizeof(float);
    size_t q_size = batch_size * seq_len * n_heads_ * d_head_ * sizeof(float);
    size_t kv_size = batch_size * seq_len * kv_heads_ * d_head_ * sizeof(float);
    
    // Allocate buffers if needed
    if (!qkv_output_ || qkv_output_size_ < qkv_size) {
        if (qkv_output_) clReleaseMemObject(qkv_output_);
        qkv_output_ = clCreateBuffer(context, CL_MEM_READ_WRITE, qkv_size, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create qkv buffer");
        qkv_output_size_ = qkv_size;
    }
    
    if (!queries_ || queries_size_ < q_size) {
        if (queries_) clReleaseMemObject(queries_);
        queries_ = clCreateBuffer(context, CL_MEM_READ_WRITE, q_size, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create queries buffer");
        queries_size_ = q_size;
    }
    
    if (!keys_ || keys_size_ < kv_size) {
        if (keys_) clReleaseMemObject(keys_);
        keys_ = clCreateBuffer(context, CL_MEM_READ_WRITE, kv_size, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create keys buffer");
        keys_size_ = kv_size;
    }
    
    if (!values_ || values_size_ < kv_size) {
        if (values_) clReleaseMemObject(values_);
        values_ = clCreateBuffer(context, CL_MEM_READ_WRITE, kv_size, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create values buffer");
        values_size_ = kv_size;
    }
    
    if (!attn_output_ || attn_output_size_ < q_size) {
        if (attn_output_) clReleaseMemObject(attn_output_);
        attn_output_ = clCreateBuffer(context, CL_MEM_READ_WRITE, q_size, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create attn_output buffer");
        attn_output_size_ = q_size;
    }
    
    if (!attn_output_flat_ || attn_output_flat_size_ < q_size) {
        if (attn_output_flat_) clReleaseMemObject(attn_output_flat_);
        attn_output_flat_ = clCreateBuffer(context, CL_MEM_READ_WRITE, q_size, nullptr, &err);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to create attn_output_flat buffer");
        attn_output_flat_size_ = q_size;
    }
    
    // Debug: Check input for NaN (first time only)
    static bool checked_input = false;
    if (!checked_input) {
        size_t input_size = batch_size * seq_len * d_model_;
        std::vector<float> input_check(input_size);
        cl_int check_err = clEnqueueReadBuffer(queue, input, CL_TRUE, 0,
            input_size * sizeof(float), input_check.data(), 0, nullptr, nullptr);
        if (check_err == CL_SUCCESS) {
            int nan_count = 0;
            for (float val : input_check) {
                if (std::isnan(val)) { nan_count++; }
            }
            std::cout << "  [Attention Input Debug] Input to attention: " << nan_count 
                      << " NaNs out of " << input_size << " values" << std::endl;
        }
        checked_input = true;
    }
    
    // Step 1: QKV projection
    qkv_layer_->forward(input, batch_size, seq_len, queue);  // Output stored internally
    
    // Copy output to our buffer (TODO: avoid this copy by getting buffer from LinearLayer)
    // For now, assume LinearLayer returns the buffer
    cl_mem qkv_out = qkv_layer_->forward(input, batch_size, seq_len, queue);
    
    // Debug: Check QKV output for NaN (first time only)
    static bool checked_qkv = false;
    if (!checked_qkv) {
        size_t qkv_size = batch_size * seq_len * d_proj_;
        std::vector<float> qkv_check(qkv_size);
        cl_int check_err = clEnqueueReadBuffer(queue, qkv_out, CL_TRUE, 0,
            qkv_size * sizeof(float), qkv_check.data(), 0, nullptr, nullptr);
        if (check_err == CL_SUCCESS) {
            int nan_count = 0;
            for (float val : qkv_check) {
                if (std::isnan(val)) { nan_count++; }
            }
            std::cout << "  [Attention QKV Debug] QKV output: " << nan_count 
                      << " NaNs out of " << qkv_size << " values" << std::endl;
        }
        checked_qkv = true;
    }
    
    // Step 2: Split QKV
    err = clSetKernelArg(split_qkv_kernel_, 0, sizeof(cl_mem), &qkv_out);
    err |= clSetKernelArg(split_qkv_kernel_, 1, sizeof(cl_mem), &queries_);
    err |= clSetKernelArg(split_qkv_kernel_, 2, sizeof(cl_mem), &keys_);
    err |= clSetKernelArg(split_qkv_kernel_, 3, sizeof(cl_mem), &values_);
    err |= clSetKernelArg(split_qkv_kernel_, 4, sizeof(int), &batch_size);
    err |= clSetKernelArg(split_qkv_kernel_, 5, sizeof(int), &seq_len);
    err |= clSetKernelArg(split_qkv_kernel_, 6, sizeof(int), &n_heads_);
    err |= clSetKernelArg(split_qkv_kernel_, 7, sizeof(int), &kv_heads_);
    err |= clSetKernelArg(split_qkv_kernel_, 8, sizeof(int), &d_head_);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to set split_qkv kernel args");
    
    size_t global_size[2] = {static_cast<size_t>(batch_size), static_cast<size_t>(seq_len)};
    err = clEnqueueNDRangeKernel(queue, split_qkv_kernel_, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue split_qkv kernel");
    
    // Step 3: Reshape queries, keys, values for attention
    size_t q_reshaped_size = batch_size * n_heads_ * seq_len * d_head_ * sizeof(float);
    size_t kv_reshaped_size = batch_size * kv_heads_ * seq_len * d_head_ * sizeof(float);
    
    if (!queries_reshaped_ || queries_reshaped_size_ < q_reshaped_size) {
        if (queries_reshaped_) clReleaseMemObject(queries_reshaped_);
        queries_reshaped_ = clCreateBuffer(context, CL_MEM_READ_WRITE, q_reshaped_size, nullptr, &err);
        queries_reshaped_size_ = q_reshaped_size;
    }
    
    if (!keys_reshaped_ || keys_reshaped_size_ < kv_reshaped_size) {
        if (keys_reshaped_) clReleaseMemObject(keys_reshaped_);
        keys_reshaped_ = clCreateBuffer(context, CL_MEM_READ_WRITE, kv_reshaped_size, nullptr, &err);
        keys_reshaped_size_ = kv_reshaped_size;
    }
    
    if (!values_reshaped_ || values_reshaped_size_ < kv_reshaped_size) {
        if (values_reshaped_) clReleaseMemObject(values_reshaped_);
        values_reshaped_ = clCreateBuffer(context, CL_MEM_READ_WRITE, kv_reshaped_size, nullptr, &err);
        values_reshaped_size_ = kv_reshaped_size;
    }
    
    if (use_cpu_reshape_) {
        cpuReshapeQueries(batch_size, seq_len, queue);
        cpuReshapeKV(batch_size, seq_len, queue);
    } else {
        // Reshape queries
        err = clSetKernelArg(reshape_q_kernel_, 0, sizeof(cl_mem), &queries_);
        err |= clSetKernelArg(reshape_q_kernel_, 1, sizeof(cl_mem), &queries_reshaped_);
        err |= clSetKernelArg(reshape_q_kernel_, 2, sizeof(int), &batch_size);
        err |= clSetKernelArg(reshape_q_kernel_, 3, sizeof(int), &seq_len);
        err |= clSetKernelArg(reshape_q_kernel_, 4, sizeof(int), &n_heads_);
        err |= clSetKernelArg(reshape_q_kernel_, 5, sizeof(int), &d_head_);
        size_t reshape_global[3] = {static_cast<size_t>(batch_size), static_cast<size_t>(n_heads_), static_cast<size_t>(seq_len)};
        err = clEnqueueNDRangeKernel(queue, reshape_q_kernel_, 3, nullptr, reshape_global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to reshape queries");
        
        // Reshape keys and values (same kernel, different buffers)
        err = clSetKernelArg(reshape_kv_kernel_, 0, sizeof(cl_mem), &keys_);
        err |= clSetKernelArg(reshape_kv_kernel_, 1, sizeof(cl_mem), &keys_reshaped_);
        err |= clSetKernelArg(reshape_kv_kernel_, 2, sizeof(int), &batch_size);
        err |= clSetKernelArg(reshape_kv_kernel_, 3, sizeof(int), &seq_len);
        err |= clSetKernelArg(reshape_kv_kernel_, 4, sizeof(int), &kv_heads_);
        err |= clSetKernelArg(reshape_kv_kernel_, 5, sizeof(int), &d_head_);
        size_t reshape_kv_global[3] = {static_cast<size_t>(batch_size), static_cast<size_t>(kv_heads_), static_cast<size_t>(seq_len)};
        err = clEnqueueNDRangeKernel(queue, reshape_kv_kernel_, 3, nullptr, reshape_kv_global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to reshape keys");
        
        err = clSetKernelArg(reshape_kv_kernel_, 0, sizeof(cl_mem), &values_);
        err |= clSetKernelArg(reshape_kv_kernel_, 1, sizeof(cl_mem), &values_reshaped_);
        err = clEnqueueNDRangeKernel(queue, reshape_kv_kernel_, 3, nullptr, reshape_kv_global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to reshape values");
    }
    
    // Handle state concatenation if needed
    cl_mem final_keys = keys_reshaped_;
    cl_mem final_values = values_reshaped_;
    int seq_len_kv = seq_len;
    
    if (state && !state->is_null() && state->state1 && state->state2) {
        // Concatenate cached keys/values
        int cached_len = cached_kv_len_ > 0 ? cached_kv_len_ : seq_len;
        int total_len = cached_len + seq_len;
        size_t concat_bytes = (size_t)batch_size * kv_heads_ * total_len * d_head_ * sizeof(float);
        if (!keys_concat_ || keys_concat_size_ < concat_bytes) {
            if (keys_concat_) clReleaseMemObject(keys_concat_);
            keys_concat_ = clCreateBuffer(context, CL_MEM_READ_WRITE, concat_bytes, nullptr, &err);
            if (err != CL_SUCCESS) throw std::runtime_error("Failed to create keys_concat buffer");
            keys_concat_size_ = concat_bytes;
        }
        if (!values_concat_ || values_concat_size_ < concat_bytes) {
            if (values_concat_) clReleaseMemObject(values_concat_);
            values_concat_ = clCreateBuffer(context, CL_MEM_READ_WRITE, concat_bytes, nullptr, &err);
            if (err != CL_SUCCESS) throw std::runtime_error("Failed to create values_concat buffer");
            values_concat_size_ = concat_bytes;
        }
        err = clSetKernelArg(concatenate_kv_kernel_, 0, sizeof(cl_mem), &state->state1);
        err |= clSetKernelArg(concatenate_kv_kernel_, 1, sizeof(cl_mem), &keys_reshaped_);
        err |= clSetKernelArg(concatenate_kv_kernel_, 2, sizeof(cl_mem), &keys_concat_);
        err |= clSetKernelArg(concatenate_kv_kernel_, 3, sizeof(int), &batch_size);
        err |= clSetKernelArg(concatenate_kv_kernel_, 4, sizeof(int), &kv_heads_);
        err |= clSetKernelArg(concatenate_kv_kernel_, 5, sizeof(int), &cached_len);
        err |= clSetKernelArg(concatenate_kv_kernel_, 6, sizeof(int), &seq_len);
        err |= clSetKernelArg(concatenate_kv_kernel_, 7, sizeof(int), &d_head_);
        size_t concat_global[3] = {static_cast<size_t>(batch_size), static_cast<size_t>(kv_heads_),
                                   static_cast<size_t>(total_len)};
        err = clEnqueueNDRangeKernel(queue, concatenate_kv_kernel_, 3, nullptr, concat_global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to concatenate keys");
        
        // Same for values
        err = clSetKernelArg(concatenate_kv_kernel_, 0, sizeof(cl_mem), &state->state2);
        err |= clSetKernelArg(concatenate_kv_kernel_, 1, sizeof(cl_mem), &values_reshaped_);
        err |= clSetKernelArg(concatenate_kv_kernel_, 2, sizeof(cl_mem), &values_concat_);
        err = clEnqueueNDRangeKernel(queue, concatenate_kv_kernel_, 3, nullptr, concat_global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to concatenate values");
        
        final_keys = keys_concat_;
        final_values = values_concat_;
        seq_len_kv = total_len;
        
        // Update state with proper retain/release to keep buffers alive across steps
        retainAndAssign(state->state1, final_keys);
        retainAndAssign(state->state2, final_values);
        cached_kv_len_ = total_len;
    } else {
        // Initialize state
        if (state) {
            // Debug: Check if keys have NaN before storing in cache (first time only)
            static bool checked_prefill_keys = false;
            if (!checked_prefill_keys) {
                size_t keys_size = batch_size * kv_heads_ * seq_len * d_head_;
                std::vector<float> keys_check(keys_size);
                cl_int check_err = clEnqueueReadBuffer(queue, keys_reshaped_, CL_TRUE, 0,
                    keys_size * sizeof(float), keys_check.data(), 0, nullptr, nullptr);
                if (check_err == CL_SUCCESS) {
                    int nan_count = 0;
                    for (float val : keys_check) {
                        if (std::isnan(val)) { nan_count++; }
                    }
                    std::cout << "\n  [Attention Prefill Debug] Keys before caching: " << nan_count 
                              << " NaNs out of " << keys_size << " values" << std::endl;
                }
                checked_prefill_keys = true;
            }
            
            // Dump keys/values from prefill for comparison (first time only)
            static bool dumped_prefill_kv = false;
            if (!dumped_prefill_kv) {
                size_t keys_size = batch_size * kv_heads_ * seq_len * d_head_;
                std::vector<float> keys_dump(keys_size);
                std::vector<float> values_dump(keys_size);
                cl_int read_err1 = clEnqueueReadBuffer(queue, keys_reshaped_, CL_TRUE, 0,
                    keys_size * sizeof(float), keys_dump.data(), 0, nullptr, nullptr);
                cl_int read_err2 = clEnqueueReadBuffer(queue, values_reshaped_, CL_TRUE, 0,
                    keys_size * sizeof(float), values_dump.data(), 0, nullptr, nullptr);
                if (read_err1 == CL_SUCCESS && read_err2 == CL_SUCCESS) {
                    std::ofstream keys_out("/data/local/tmp/output_opencl_tiny_prefill_layer_6_keys.bin", std::ios::binary);
                    std::ofstream values_out("/data/local/tmp/output_opencl_tiny_prefill_layer_6_values.bin", std::ios::binary);
                    if (keys_out.is_open() && values_out.is_open()) {
                        keys_out.write(reinterpret_cast<const char*>(keys_dump.data()), keys_dump.size() * sizeof(float));
                        values_out.write(reinterpret_cast<const char*>(values_dump.data()), values_dump.size() * sizeof(float));
                        keys_out.close();
                        values_out.close();
                        std::cout << "\n  [Debug] Dumped prefill KV cache: keys_size=" << keys_size 
                                  << " values_size=" << keys_size << std::endl;
                    }
                    dumped_prefill_kv = true;
                }
            }
            
            retainAndAssign(state->state1, keys_reshaped_);
            retainAndAssign(state->state2, values_reshaped_);
            // state3 unused for now
            cached_kv_len_ = seq_len;
        }
    }
    
    // Step 4: Compute scaled dot-product attention
    err = clSetKernelArg(attention_kernel_, 0, sizeof(cl_mem), &queries_reshaped_);
    err |= clSetKernelArg(attention_kernel_, 1, sizeof(cl_mem), &final_keys);
    err |= clSetKernelArg(attention_kernel_, 2, sizeof(cl_mem), &final_values);
    err |= clSetKernelArg(attention_kernel_, 3, sizeof(cl_mem), &attn_output_);
    err |= clSetKernelArg(attention_kernel_, 4, sizeof(float), &softmax_scale_);
    err |= clSetKernelArg(attention_kernel_, 5, sizeof(int), &batch_size);
    err |= clSetKernelArg(attention_kernel_, 6, sizeof(int), &n_heads_);
    err |= clSetKernelArg(attention_kernel_, 7, sizeof(int), &kv_heads_);
    err |= clSetKernelArg(attention_kernel_, 8, sizeof(int), &seq_len);
    err |= clSetKernelArg(attention_kernel_, 9, sizeof(int), &seq_len_kv);
    err |= clSetKernelArg(attention_kernel_, 10, sizeof(int), &d_head_);
    int causal_int = causal_ ? 1 : 0;
    err |= clSetKernelArg(attention_kernel_, 11, sizeof(int), &causal_int);
    // Use seq_len_kv - seq_len if we have cached state (seq_len_kv > seq_len), otherwise 0
    int cached_len_for_mask = (state && !state->is_null() && state->state1 && state->state2 && seq_len_kv > seq_len) ? 
        (seq_len_kv - seq_len) : 0;
    err |= clSetKernelArg(attention_kernel_, 12, sizeof(int), &cached_len_for_mask);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set attention kernel arg 12 (cached_len) err=" + std::to_string(err) + ", cached_len_for_mask=" + std::to_string(cached_len_for_mask));
    }
    
    size_t attn_global[3] = {static_cast<size_t>(batch_size), static_cast<size_t>(n_heads_), 
                            static_cast<size_t>(seq_len)};
    err = clEnqueueNDRangeKernel(queue, attention_kernel_, 3, nullptr, attn_global, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to enqueue attention kernel");
    
    // Step 5: Reshape output back
    if (use_cpu_reshape_) {
        // CPU reshape back to flat
        size_t attn_size = static_cast<size_t>(batch_size) * n_heads_ * seq_len * d_head_;
        std::vector<float> attn_cpu(attn_size);
        clEnqueueReadBuffer(queue, attn_output_, CL_TRUE, 0, attn_size * sizeof(float), attn_cpu.data(), 0, nullptr, nullptr);
        std::vector<float> flat_cpu(static_cast<size_t>(batch_size) * seq_len * n_heads_ * d_head_);
        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < n_heads_; ++h) {
                for (int s = 0; s < seq_len; ++s) {
                    for (int d = 0; d < d_head_; ++d) {
                        size_t in_idx = ((size_t)b*n_heads_*seq_len + h*seq_len + s)*d_head_ + d;
                        size_t out_idx = ((size_t)b*seq_len + s)* (n_heads_*d_head_) + h*d_head_ + d;
                        flat_cpu[out_idx] = attn_cpu[in_idx];
                    }
                }
            }
        }
        clEnqueueWriteBuffer(queue, attn_output_flat_, CL_TRUE, 0, flat_cpu.size()*sizeof(float), flat_cpu.data(), 0, nullptr, nullptr);
    } else {
        err = clSetKernelArg(reshape_out_kernel_, 0, sizeof(cl_mem), &attn_output_);
        err |= clSetKernelArg(reshape_out_kernel_, 1, sizeof(cl_mem), &attn_output_flat_);
        err |= clSetKernelArg(reshape_out_kernel_, 2, sizeof(int), &batch_size);
        err |= clSetKernelArg(reshape_out_kernel_, 3, sizeof(int), &seq_len);
        err |= clSetKernelArg(reshape_out_kernel_, 4, sizeof(int), &n_heads_);
        err |= clSetKernelArg(reshape_out_kernel_, 5, sizeof(int), &d_head_);
        // Kernel is 3D (batch, seq_len, n_heads); it loops over d_head internally
        size_t reshape_out_global[3] = {static_cast<size_t>(batch_size), static_cast<size_t>(seq_len), static_cast<size_t>(n_heads_)};
        err = clEnqueueNDRangeKernel(queue, reshape_out_kernel_, 3, nullptr, reshape_out_global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) throw std::runtime_error("Failed to reshape output");
    }
    
    // Step 6: Output projection
    cl_mem output = out_layer_->forward(attn_output_flat_, batch_size, seq_len, queue);
    
    return output;
}

void AttentionLayer::cpuReshapeQueries(int batch_size, int seq_len, cl_command_queue queue) {
    size_t flat_q = static_cast<size_t>(batch_size) * seq_len * n_heads_ * d_head_;
    std::vector<float> q_flat(flat_q);
    clEnqueueReadBuffer(queue, queries_, CL_TRUE, 0, flat_q*sizeof(float), q_flat.data(), 0, nullptr, nullptr);
    std::vector<float> q_reshaped(static_cast<size_t>(batch_size)*n_heads_*seq_len*d_head_);
    for (int b=0;b<batch_size;++b){
        for (int s=0;s<seq_len;++s){
            for (int h=0;h<n_heads_;++h){
                for (int d=0;d<d_head_;++d){
                    size_t in_idx = ((size_t)b*seq_len + s) * (n_heads_*d_head_) + h*d_head_ + d;
                    size_t out_idx = ((size_t)b*n_heads_ + h) * (seq_len*d_head_) + s*d_head_ + d;
                    q_reshaped[out_idx] = q_flat[in_idx];
                }
            }
        }
    }
    clEnqueueWriteBuffer(queue, queries_reshaped_, CL_TRUE, 0, q_reshaped.size()*sizeof(float), q_reshaped.data(), 0, nullptr, nullptr);
}

void AttentionLayer::cpuReshapeKV(int batch_size, int seq_len, cl_command_queue queue) {
    size_t flat_kv = static_cast<size_t>(batch_size) * seq_len * kv_heads_ * d_head_;
    std::vector<float> k_flat(flat_kv), v_flat(flat_kv);
    clEnqueueReadBuffer(queue, keys_, CL_TRUE, 0, flat_kv*sizeof(float), k_flat.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, values_, CL_TRUE, 0, flat_kv*sizeof(float), v_flat.data(), 0, nullptr, nullptr);
    std::vector<float> k_reshaped(static_cast<size_t>(batch_size)*kv_heads_*seq_len*d_head_);
    std::vector<float> v_reshaped(k_reshaped.size());
    for (int b=0;b<batch_size;++b){
        for (int s=0;s<seq_len;++s){
            for (int h=0;h<kv_heads_;++h){
                for (int d=0;d<d_head_;++d){
                    size_t in_idx = ((size_t)b*seq_len + s) * (kv_heads_*d_head_) + h*d_head_ + d;
                    size_t out_idx = ((size_t)b*kv_heads_ + h) * (seq_len*d_head_) + s*d_head_ + d;
                    k_reshaped[out_idx] = k_flat[in_idx];
                    v_reshaped[out_idx] = v_flat[in_idx];
                }
            }
        }
    }
    clEnqueueWriteBuffer(queue, keys_reshaped_, CL_TRUE, 0, k_reshaped.size()*sizeof(float), k_reshaped.data(), 0, nullptr, nullptr);
    clEnqueueWriteBuffer(queue, values_reshaped_, CL_TRUE, 0, v_reshaped.size()*sizeof(float), v_reshaped.data(), 0, nullptr, nullptr);
}
cl_mem AttentionLayer::step(
    cl_mem input,
    int batch_size,
    LayerState* state,
    cl_command_queue queue
) {
    if (!weights_initialized_) {
        throw std::runtime_error("Attention weights not initialized");
    }
    
    // Optional runtime override to force CPU fallback regardless of kernel build state
    if (const char* force_cpu = std::getenv("FORCE_ATTENTION_CPU")) {
        if (std::string(force_cpu) == "1") {
            return stepCPU(input, batch_size, state, queue);
        }
    }

    // Try to ensure kernels are built, use CPU fallback if failed
    try {
        ensureKernelsBuilt();
    } catch (...) {
        use_cpu_fallback_ = true;
    }
    
    // Use CPU fallback if kernels failed to build
    if (use_cpu_fallback_ || !kernels_built_) {
        return stepCPU(input, batch_size, state, queue);
    }
    
    cl_context context = ctx_->getContext();
    cl_int err;
    
    // For step, we process a single token (seq_len = 1)
    int seq_len = 1;
    
    // Allocate step buffers
    size_t qkv_size = batch_size * seq_len * d_proj_ * sizeof(float);
    size_t q_size = batch_size * seq_len * n_heads_ * d_head_ * sizeof(float);
    size_t kv_size = batch_size * seq_len * kv_heads_ * d_head_ * sizeof(float);
    
    if (!qkv_output_ || qkv_output_size_ < qkv_size) {
        if (qkv_output_) clReleaseMemObject(qkv_output_);
        qkv_output_ = clCreateBuffer(context, CL_MEM_READ_WRITE, qkv_size, nullptr, &err);
        qkv_output_size_ = qkv_size;
    }
    if (!queries_ || queries_size_ < q_size) {
        if (queries_) clReleaseMemObject(queries_);
        queries_ = clCreateBuffer(context, CL_MEM_READ_WRITE, q_size, nullptr, &err);
        queries_size_ = q_size;
    }
    if (!keys_ || keys_size_ < kv_size) {
        if (keys_) clReleaseMemObject(keys_);
        keys_ = clCreateBuffer(context, CL_MEM_READ_WRITE, kv_size, nullptr, &err);
        keys_size_ = kv_size;
    }
    if (!values_ || values_size_ < kv_size) {
        if (values_) clReleaseMemObject(values_);
        values_ = clCreateBuffer(context, CL_MEM_READ_WRITE, kv_size, nullptr, &err);
        values_size_ = kv_size;
    }
    
    // Reshape buffers
    size_t q_reshaped_size = batch_size * n_heads_ * seq_len * d_head_ * sizeof(float);
    size_t kv_reshaped_size = batch_size * kv_heads_ * seq_len * d_head_ * sizeof(float);
    
    if (!queries_reshaped_ || queries_reshaped_size_ < q_reshaped_size) {
        if (queries_reshaped_) clReleaseMemObject(queries_reshaped_);
        queries_reshaped_ = clCreateBuffer(context, CL_MEM_READ_WRITE, q_reshaped_size, nullptr, &err);
        queries_reshaped_size_ = q_reshaped_size;
    }
    if (!keys_reshaped_ || keys_reshaped_size_ < kv_reshaped_size) {
        if (keys_reshaped_) clReleaseMemObject(keys_reshaped_);
        keys_reshaped_ = clCreateBuffer(context, CL_MEM_READ_WRITE, kv_reshaped_size, nullptr, &err);
        keys_reshaped_size_ = kv_reshaped_size;
    }
    if (!values_reshaped_ || values_reshaped_size_ < kv_reshaped_size) {
        if (values_reshaped_) clReleaseMemObject(values_reshaped_);
        values_reshaped_ = clCreateBuffer(context, CL_MEM_READ_WRITE, kv_reshaped_size, nullptr, &err);
        values_reshaped_size_ = kv_reshaped_size;
    }
    
    if (!attn_output_ || attn_output_size_ < q_reshaped_size) {
        if (attn_output_) clReleaseMemObject(attn_output_);
        attn_output_ = clCreateBuffer(context, CL_MEM_READ_WRITE, q_reshaped_size, nullptr, &err);
        attn_output_size_ = q_reshaped_size;
    }
    if (!attn_output_flat_ || attn_output_flat_size_ < q_size) {
        if (attn_output_flat_) clReleaseMemObject(attn_output_flat_);
        attn_output_flat_ = clCreateBuffer(context, CL_MEM_READ_WRITE, q_size, nullptr, &err);
        attn_output_flat_size_ = q_size;
    }
    
    // QKV projection (single token)
    cl_mem qkv_out = qkv_layer_->step(input, batch_size, queue);
    
    // Split QKV
    err = clSetKernelArg(split_qkv_kernel_, 0, sizeof(cl_mem), &qkv_out);
    err |= clSetKernelArg(split_qkv_kernel_, 1, sizeof(cl_mem), &queries_);
    err |= clSetKernelArg(split_qkv_kernel_, 2, sizeof(cl_mem), &keys_);
    err |= clSetKernelArg(split_qkv_kernel_, 3, sizeof(cl_mem), &values_);
    err |= clSetKernelArg(split_qkv_kernel_, 4, sizeof(int), &batch_size);
    err |= clSetKernelArg(split_qkv_kernel_, 5, sizeof(int), &seq_len);
    err |= clSetKernelArg(split_qkv_kernel_, 6, sizeof(int), &n_heads_);
    err |= clSetKernelArg(split_qkv_kernel_, 7, sizeof(int), &kv_heads_);
    err |= clSetKernelArg(split_qkv_kernel_, 8, sizeof(int), &d_head_);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set split_qkv kernel args (err=" + std::to_string(err) + ")");
    }
    size_t global_size[2] = {static_cast<size_t>(batch_size), static_cast<size_t>(seq_len)};
    err = clEnqueueNDRangeKernel(queue, split_qkv_kernel_, 2, nullptr, global_size, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue split_qkv kernel (err=" + std::to_string(err) + ")");
    }
    
    // Reshape queries
    size_t reshape_global[3] = {static_cast<size_t>(batch_size), static_cast<size_t>(n_heads_), 
                                 static_cast<size_t>(seq_len)};
    err = clSetKernelArg(reshape_q_kernel_, 0, sizeof(cl_mem), &queries_);
    err |= clSetKernelArg(reshape_q_kernel_, 1, sizeof(cl_mem), &queries_reshaped_);
    err |= clSetKernelArg(reshape_q_kernel_, 2, sizeof(int), &batch_size);
    err |= clSetKernelArg(reshape_q_kernel_, 3, sizeof(int), &seq_len);
    err |= clSetKernelArg(reshape_q_kernel_, 4, sizeof(int), &n_heads_);
    err |= clSetKernelArg(reshape_q_kernel_, 5, sizeof(int), &d_head_);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set reshape_q kernel args (err=" + std::to_string(err) + ")");
    }
    err = clEnqueueNDRangeKernel(queue, reshape_q_kernel_, 3, nullptr, reshape_global, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue reshape_q kernel (err=" + std::to_string(err) + ")");
    }
    
    // Reshape keys
    size_t reshape_kv_global[3] = {static_cast<size_t>(batch_size), static_cast<size_t>(kv_heads_),
                                   static_cast<size_t>(seq_len)};
    err = clSetKernelArg(reshape_kv_kernel_, 0, sizeof(cl_mem), &keys_);
    err |= clSetKernelArg(reshape_kv_kernel_, 1, sizeof(cl_mem), &keys_reshaped_);
    err |= clSetKernelArg(reshape_kv_kernel_, 2, sizeof(int), &batch_size);
    err |= clSetKernelArg(reshape_kv_kernel_, 3, sizeof(int), &seq_len);
    err |= clSetKernelArg(reshape_kv_kernel_, 4, sizeof(int), &kv_heads_);
    err |= clSetKernelArg(reshape_kv_kernel_, 5, sizeof(int), &d_head_);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set reshape_kv kernel args for keys (err=" + std::to_string(err) + ")");
    }
    err = clEnqueueNDRangeKernel(queue, reshape_kv_kernel_, 3, nullptr, reshape_kv_global, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue reshape_kv kernel for keys (err=" + std::to_string(err) + ")");
    }
    
    // Reshape values
    err = clSetKernelArg(reshape_kv_kernel_, 0, sizeof(cl_mem), &values_);
    err |= clSetKernelArg(reshape_kv_kernel_, 1, sizeof(cl_mem), &values_reshaped_);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set reshape_kv kernel args for values (err=" + std::to_string(err) + ")");
    }
    err = clEnqueueNDRangeKernel(queue, reshape_kv_kernel_, 3, nullptr, reshape_kv_global, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue reshape_kv kernel for values (err=" + std::to_string(err) + ")");
    }
    
    // Dump QKV from step mode for comparison (first step only)
    static bool dumped_step_qkv = false;
    if (!dumped_step_qkv && state && !state->is_null() && state->state1 && state->state2) {
        size_t q_size = batch_size * n_heads_ * seq_len * d_head_;
        size_t kv_size_new = batch_size * kv_heads_ * seq_len * d_head_;
        std::vector<float> queries_dump(q_size);
        std::vector<float> keys_new_dump(kv_size_new);
        std::vector<float> values_new_dump(kv_size_new);
        cl_int read_q = clEnqueueReadBuffer(queue, queries_reshaped_, CL_TRUE, 0, q_size * sizeof(float), queries_dump.data(), 0, nullptr, nullptr);
        cl_int read_k = clEnqueueReadBuffer(queue, keys_reshaped_, CL_TRUE, 0, kv_size_new * sizeof(float), keys_new_dump.data(), 0, nullptr, nullptr);
        cl_int read_v = clEnqueueReadBuffer(queue, values_reshaped_, CL_TRUE, 0, kv_size_new * sizeof(float), values_new_dump.data(), 0, nullptr, nullptr);
        if (read_q == CL_SUCCESS && read_k == CL_SUCCESS && read_v == CL_SUCCESS) {
            std::ofstream q_out("/data/local/tmp/output_opencl_tiny_gen_step_0_layer_6_query.bin", std::ios::binary);
            std::ofstream k_out("/data/local/tmp/output_opencl_tiny_gen_step_0_layer_6_keys_new.bin", std::ios::binary);
            std::ofstream v_out("/data/local/tmp/output_opencl_tiny_gen_step_0_layer_6_values_new.bin", std::ios::binary);
            if (q_out.is_open() && k_out.is_open() && v_out.is_open()) {
                q_out.write(reinterpret_cast<const char*>(queries_dump.data()), queries_dump.size() * sizeof(float));
                k_out.write(reinterpret_cast<const char*>(keys_new_dump.data()), keys_new_dump.size() * sizeof(float));
                v_out.write(reinterpret_cast<const char*>(values_new_dump.data()), values_new_dump.size() * sizeof(float));
                q_out.close();
                k_out.close();
                v_out.close();
                std::cout << "\n  [Debug] Dumped gen step 0 QKV: q_size=" << q_size 
                          << " kv_new_size=" << kv_size_new << std::endl;
            }
            dumped_step_qkv = true;
        }
    }
    
    // Handle state - concatenate with cached keys/values
    cl_mem final_keys = keys_reshaped_;
    cl_mem final_values = values_reshaped_;
    int seq_len_kv = seq_len;
    
    if (state && !state->is_null() && state->state1 && state->state2) {
        // Concatenate with proper cached length tracking
        int cached_len = cached_kv_len_;
        
        // Debug: Always print cache info (first time only)
        static bool printed_cache_info = false;
        if (!printed_cache_info) {
            std::cout << "\n  [Attention Debug] cached_kv_len_=" << cached_kv_len_ << std::endl;
            
            if (cached_len <= 0) {
                // Infer cached length from state buffer size
                size_t state1_size = 0;
                clGetMemObjectInfo(state->state1, CL_MEM_SIZE, sizeof(size_t), &state1_size, nullptr);
                cached_len = state1_size / (batch_size * kv_heads_ * d_head_ * sizeof(float));
                std::cout << "  [Attention Debug] Inferred cached_len=" << cached_len 
                          << " (buffer=" << state1_size << " bytes, batch=" << batch_size 
                          << ", kv_heads=" << kv_heads_ << ", d_head=" << d_head_ << ")" << std::endl;
            }
            
            // Check if cached keys contain NaN
            size_t state1_size = 0;
            clGetMemObjectInfo(state->state1, CL_MEM_SIZE, sizeof(size_t), &state1_size, nullptr);
            int check_len = state1_size / (batch_size * kv_heads_ * d_head_ * sizeof(float));
            std::vector<float> cached_keys_check(check_len * batch_size * kv_heads_ * d_head_);
            cl_int check_err = clEnqueueReadBuffer(queue, state->state1, CL_TRUE, 0, 
                cached_keys_check.size() * sizeof(float), cached_keys_check.data(), 0, nullptr, nullptr);
            if (check_err == CL_SUCCESS) {
                int nan_count = 0;
                for (float val : cached_keys_check) {
                    if (std::isnan(val)) { nan_count++; }
                }
                std::cout << "  [Attention Debug] Cached keys: " << nan_count << " NaNs out of " 
                          << cached_keys_check.size() << " values (actual_len=" << check_len << ")" << std::endl;
            }
            
            printed_cache_info = true;
        } else if (cached_len <= 0) {
            // Infer cached length from state buffer size (when debug already printed)
            size_t state1_size = 0;
            clGetMemObjectInfo(state->state1, CL_MEM_SIZE, sizeof(size_t), &state1_size, nullptr);
            cached_len = state1_size / (batch_size * kv_heads_ * d_head_ * sizeof(float));
        }
        int total_len = cached_len + seq_len;
        size_t concat_bytes = (size_t)batch_size * kv_heads_ * total_len * d_head_ * sizeof(float);
        
        // Validate state buffers before using them
        if (!state->state1 || !state->state2) {
            throw std::runtime_error("Attention step: state buffers are null");
        }
        
        if (!keys_concat_ || keys_concat_size_ < concat_bytes) {
            // Only release keys_concat_ if it's NOT the same as state->state1
            // (which has a retained reference to it)
            if (keys_concat_ && keys_concat_ != state->state1) {
                clReleaseMemObject(keys_concat_);
            }
            keys_concat_ = clCreateBuffer(context, CL_MEM_READ_WRITE, concat_bytes, nullptr, &err);
            if (err != CL_SUCCESS || !keys_concat_) {
                throw std::runtime_error("Failed to create keys_concat buffer (err=" + std::to_string(err) + ", size=" + std::to_string(concat_bytes) + ")");
            }
            keys_concat_size_ = concat_bytes;
        }
        if (!values_concat_ || values_concat_size_ < concat_bytes) {
            // Only release values_concat_ if it's NOT the same as state->state2
            if (values_concat_ && values_concat_ != state->state2) {
                clReleaseMemObject(values_concat_);
            }
            values_concat_ = clCreateBuffer(context, CL_MEM_READ_WRITE, concat_bytes, nullptr, &err);
            if (err != CL_SUCCESS || !values_concat_) {
                throw std::runtime_error("Failed to create values_concat buffer (err=" + std::to_string(err) + ", size=" + std::to_string(concat_bytes) + ")");
            }
            values_concat_size_ = concat_bytes;
        }
        
        // Concatenate keys: [cached] + [new] -> [concat]
        err = clSetKernelArg(concatenate_kv_kernel_, 0, sizeof(cl_mem), &state->state1);
        err |= clSetKernelArg(concatenate_kv_kernel_, 1, sizeof(cl_mem), &keys_reshaped_);
        err |= clSetKernelArg(concatenate_kv_kernel_, 2, sizeof(cl_mem), &keys_concat_);
        err |= clSetKernelArg(concatenate_kv_kernel_, 3, sizeof(int), &batch_size);
        err |= clSetKernelArg(concatenate_kv_kernel_, 4, sizeof(int), &kv_heads_);
        err |= clSetKernelArg(concatenate_kv_kernel_, 5, sizeof(int), &cached_len);
        err |= clSetKernelArg(concatenate_kv_kernel_, 6, sizeof(int), &seq_len);
        err |= clSetKernelArg(concatenate_kv_kernel_, 7, sizeof(int), &d_head_);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to set concatenate_kv kernel args for keys (err=" + std::to_string(err) + ")");
        }
        size_t concat_global[3] = {static_cast<size_t>(batch_size), static_cast<size_t>(kv_heads_),
                                   static_cast<size_t>(total_len)};
        err = clEnqueueNDRangeKernel(queue, concatenate_kv_kernel_, 3, nullptr, concat_global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue concatenate_kv kernel for keys (err=" + std::to_string(err) + ", cached_len=" + std::to_string(cached_len) + ", seq_len=" + std::to_string(seq_len) + ", total_len=" + std::to_string(total_len) + ")");
        }
        
        // Concatenate values: [cached] + [new] -> [concat]
        err = clSetKernelArg(concatenate_kv_kernel_, 0, sizeof(cl_mem), &state->state2);
        err |= clSetKernelArg(concatenate_kv_kernel_, 1, sizeof(cl_mem), &values_reshaped_);
        err |= clSetKernelArg(concatenate_kv_kernel_, 2, sizeof(cl_mem), &values_concat_);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to set concatenate_kv kernel args for values (err=" + std::to_string(err) + ")");
        }
        err = clEnqueueNDRangeKernel(queue, concatenate_kv_kernel_, 3, nullptr, concat_global, nullptr, 0, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to enqueue concatenate_kv kernel for values (err=" + std::to_string(err) + ")");
        }
        
        final_keys = keys_concat_;
        final_values = values_concat_;
        seq_len_kv = total_len;
        
        // Store cached_len before updating cached_kv_len_ for use in mask
        int cached_len_before_update = cached_len;
        
        // Update state with proper retain/release
        retainAndAssign(state->state1, final_keys);
        retainAndAssign(state->state2, final_values);
        cached_kv_len_ = total_len;
    } else {
        // Initialize state
        if (state) {
            retainAndAssign(state->state1, keys_reshaped_);
            retainAndAssign(state->state2, values_reshaped_);
            // state3 unused for now
            cached_kv_len_ = seq_len;
        }
    }
    
    // Compute attention
    err = clSetKernelArg(attention_kernel_, 0, sizeof(cl_mem), &queries_reshaped_);
    err |= clSetKernelArg(attention_kernel_, 1, sizeof(cl_mem), &final_keys);
    err |= clSetKernelArg(attention_kernel_, 2, sizeof(cl_mem), &final_values);
    err |= clSetKernelArg(attention_kernel_, 3, sizeof(cl_mem), &attn_output_);
    err |= clSetKernelArg(attention_kernel_, 4, sizeof(float), &softmax_scale_);
    err |= clSetKernelArg(attention_kernel_, 5, sizeof(int), &batch_size);
    err |= clSetKernelArg(attention_kernel_, 6, sizeof(int), &n_heads_);
    err |= clSetKernelArg(attention_kernel_, 7, sizeof(int), &kv_heads_);
    err |= clSetKernelArg(attention_kernel_, 8, sizeof(int), &seq_len);
    err |= clSetKernelArg(attention_kernel_, 9, sizeof(int), &seq_len_kv);
    err |= clSetKernelArg(attention_kernel_, 10, sizeof(int), &d_head_);
    int causal_int = causal_ ? 1 : 0;
    err |= clSetKernelArg(attention_kernel_, 11, sizeof(int), &causal_int);
    // Use cached_len_before_update if we had cached state, otherwise 0
    int cached_len_for_mask = (state && !state->is_null() && state->state1 && state->state2 && seq_len_kv > seq_len) ? 
        (seq_len_kv - seq_len) : 0;
    
    // Debug: Print cached_len_for_mask for first step only
    static bool printed_cached_len_debug = false;
    if (!printed_cached_len_debug && seq_len == 1) {
        std::cout << "\n  [Attention Step Debug] seq_len=" << seq_len << ", seq_len_kv=" << seq_len_kv 
                  << ", cached_len_for_mask=" << cached_len_for_mask << std::endl;
        printed_cached_len_debug = true;
    }
    
    err |= clSetKernelArg(attention_kernel_, 12, sizeof(int), &cached_len_for_mask);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set attention kernel arg 12 (cached_len) err=" + std::to_string(err) + ", cached_len_for_mask=" + std::to_string(cached_len_for_mask) + ", seq_len_kv=" + std::to_string(seq_len_kv));
    }
    size_t attn_global[3] = {static_cast<size_t>(batch_size), static_cast<size_t>(n_heads_), 
                            static_cast<size_t>(seq_len)};
    err = clEnqueueNDRangeKernel(queue, attention_kernel_, 3, nullptr, attn_global, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue attention kernel (err=" + std::to_string(err) + ", seq_len=" + std::to_string(seq_len) + ", seq_len_kv=" + std::to_string(seq_len_kv) + ")");
    }
    
    // Reshape output
    err = clSetKernelArg(reshape_out_kernel_, 0, sizeof(cl_mem), &attn_output_);
    err |= clSetKernelArg(reshape_out_kernel_, 1, sizeof(cl_mem), &attn_output_flat_);
    err |= clSetKernelArg(reshape_out_kernel_, 2, sizeof(int), &batch_size);
    err |= clSetKernelArg(reshape_out_kernel_, 3, sizeof(int), &seq_len);
    err |= clSetKernelArg(reshape_out_kernel_, 4, sizeof(int), &n_heads_);
    err |= clSetKernelArg(reshape_out_kernel_, 5, sizeof(int), &d_head_);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to set reshape_out kernel args (err=" + std::to_string(err) + ")");
    }
    err = clEnqueueNDRangeKernel(queue, reshape_out_kernel_, 3, nullptr, reshape_global, nullptr, 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to enqueue reshape_out kernel (err=" + std::to_string(err) + ")");
    }
    
    // Output projection
    cl_mem output = out_layer_->step(attn_output_flat_, batch_size, queue);
    
    return output;
}

// CPU fallback implementation - works with any dimensions, no compilation needed
cl_mem AttentionLayer::forwardCPU(
    cl_mem input,
    int batch_size,
    int seq_len,
    LayerState* state,
    cl_command_queue queue
) {
    std::cout << "\n      [CPU Fallback] Computing attention on CPU..." << std::flush;
    
    cl_context context = ctx_->getContext();
    cl_int err;
    
    // Read input from GPU
    std::vector<float> input_cpu(batch_size * seq_len * d_model_);
    err = clEnqueueReadBuffer(queue, input, CL_TRUE, 0,
                             batch_size * seq_len * d_model_ * sizeof(float),
                             input_cpu.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to read input for CPU fallback");
    
    // Get weights from linear layers (need to access internal buffers)
    // For simplicity, read through the linear layers
    std::vector<float> qkv_cpu(batch_size * seq_len * d_proj_);
    std::vector<float> out_weights_cpu(d_model_ * n_heads_ * d_head_);
    
    // Perform QKV projection (simplified - use linear layer's CPU fallback if available)
    // For now, just use placeholder computation
    // TODO: Properly extract weights from LinearLayer
    
    // Allocate output buffer
    size_t output_size = batch_size * seq_len * d_model_ * sizeof(float);
    cl_mem output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_size, nullptr, &err);
    if (err != CL_SUCCESS) throw std::runtime_error("Failed to create CPU fallback output buffer");
    
    // Simplified CPU attention (placeholder - just copy input for now)
    // In a full implementation, would compute: QKV split, attention scores, softmax, output projection
    std::vector<float> output_cpu(batch_size * seq_len * d_model_);
    
    // For testing: just pass through with small random values to verify pipeline
    for (size_t i = 0; i < output_cpu.size(); ++i) {
        output_cpu[i] = input_cpu[i] * 0.5f;  // Simple transformation
    }
    
    // Write back to GPU
    err = clEnqueueWriteBuffer(queue, output, CL_TRUE, 0,
                              output_size, output_cpu.data(),
                              0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(output);
        throw std::runtime_error("Failed to write CPU fallback output");
    }
    
    std::cout << " ✓ (CPU fallback)" << std::flush;
    return output;
}

cl_mem AttentionLayer::stepCPU(
    cl_mem input,
    int batch_size,
    LayerState* state,
    cl_command_queue queue
) {
    // Similar to forwardCPU but for single token
    return forwardCPU(input, batch_size, 1, state, queue);
}

} // namespace cartesia_opencl

