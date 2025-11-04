#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
#include <cerrno>
#include <cstring>
#include <unistd.h>
#include <CL/cl.h>

#include "src/opencl_context.h"
#include "src/model_config.h"
#include "src/embedding.h"
#include "src/sequence_model.h"
#include "src/residual_block.h"
#include "src/layers/ssd_layer.h"
#include "src/layers/swiglu_layer.h"
#include "src/layers/attention_layer.h"
#include "src/lm_head.h"
#include "src/sampling.h"
#include "src/weights.h"
#include <cstdlib>
#include <ctime>
#include <cmath>

using namespace cartesia_opencl;

// Helper: Print device memory info
void printMemoryInfo(cl_device_id device, const std::string& label) {
    cl_ulong free_mem = 0;
    cl_ulong total_mem = 0;
    cl_int err;
    
    // Try to get global memory size (total available)
    err = clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &total_mem, nullptr);
    if (err == CL_SUCCESS) {
        std::cout << "  [Memory " << label << "] Total device memory: " 
                  << (total_mem / 1024 / 1024) << " MB" << std::endl;
    }
    
    // Try to get max allocation size
    size_t max_alloc = 0;
    err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &max_alloc, nullptr);
    if (err == CL_SUCCESS) {
        std::cout << "  [Memory " << label << "] Max allocation: " 
                  << (max_alloc / 1024 / 1024) << " MB" << std::endl;
    }
    
    // Note: OpenCL doesn't have a standard way to query free memory
    // Some vendors provide extensions, but they're not universal
    std::cout.flush();
}

// Helper: Read token IDs from binary file
std::vector<int32_t> readTokenFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open token file: " + filename);
    }
    
    // Read file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read token IDs (assuming int32_t)
    size_t num_tokens = file_size / sizeof(int32_t);
    std::vector<int32_t> tokens(num_tokens);
    file.read(reinterpret_cast<char*>(tokens.data()), file_size);
    
    if (!file) {
        throw std::runtime_error("Failed to read token file completely");
    }
    
    return tokens;
}

// Helper: Check for NaN in buffer (debug)
bool checkForNaN(cl_mem buffer, size_t size, cl_command_queue queue, const std::string& name) {
    std::vector<float> data(size);
    cl_int err = clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, size * sizeof(float), data.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "  [NaN Check] Failed to read " << name << std::endl;
        return false;
    }
    
    int nan_count = 0, inf_count = 0;
    float min_val = data[0], max_val = data[0];
    for (float val : data) {
        if (std::isnan(val)) nan_count++;
        if (std::isinf(val)) inf_count++;
        if (std::isfinite(val)) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
    }
    
    if (nan_count > 0 || inf_count > 0) {
        std::cout << "  [NaN Check] " << name << ": " << nan_count << " NaNs, " << inf_count << " Infs, "
                  << "min=" << min_val << ", max=" << max_val << std::endl;
        return true;
    }
    return false;
}

// Helper: Load weights from binary file
std::vector<float> loadWeights(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open weight file: " + filename);
    }
    
    // Read file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Read weights (float32)
    size_t num_weights = file_size / sizeof(float);
    std::vector<float> weights(num_weights);
    file.read(reinterpret_cast<char*>(weights.data()), file_size);
    
    if (!file) {
        throw std::runtime_error("Failed to read weight file completely");
    }
    
    std::cout << "  Loaded " << num_weights << " weights from " << filename << std::endl;
    return weights;
}

// Helper: Write token IDs to binary file
void writeTokenFile(const std::string& filename, const std::vector<int32_t>& tokens) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        char cwd[1024];
        if (getcwd(cwd, sizeof(cwd)) != nullptr) {
            std::cerr << "Current working directory: " << cwd << std::endl;
        }
        std::cerr << "Attempting to write to: " << filename << std::endl;
        std::cerr << "errno: " << errno << " (" << strerror(errno) << ")" << std::endl;
        throw std::runtime_error("Failed to open output file: " + filename);
    }
    file.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(int32_t));
    if (!file.good()) {
        throw std::runtime_error("Failed to write to output file: " + filename);
    }
    file.close();
}

// Helper: Create token IDs buffer in OpenCL
cl_mem createTokenBuffer(cl_context context, const std::vector<int32_t>& tokens) {
    cl_int err;
    cl_mem buffer = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        tokens.size() * sizeof(int32_t),
        (void*)tokens.data(),
        &err
    );
    if (err != CL_SUCCESS || !buffer) {
        throw std::runtime_error("Failed to create token buffer");
    }
    return buffer;
}

int main(int argc, char* argv[]) {
    try {
        std::cout << "========================================" << std::endl;
        std::cout << "Cartesia OpenCL Rene Model Driver" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << std::endl;
        
        // Parse arguments
        if (argc < 2) {
            std::cerr << "Usage: " << argv[0] << " <token_file.bin> [weights_dir] [output_file.bin] [max_tokens]" << std::endl;
            std::cerr << "  token_file.bin: Input file with token IDs (int32 binary)" << std::endl;
            std::cerr << "  weights_dir: Directory containing model weights (optional, generates random if not provided)" << std::endl;
            std::cerr << "  output_file.bin: Output file for generated tokens (default: /data/local/tmp/output_tokens.bin)" << std::endl;
            std::cerr << "  max_tokens: Maximum tokens to generate (default: 3)" << std::endl;
            return 1;
        }
        
        std::string token_file = argv[1];
        std::string weights_dir = (argc >= 3) ? argv[2] : "";
        std::string output_file = (argc >= 4) ? argv[3] : "/data/local/tmp/output_tokens.bin";
        int max_tokens = (argc >= 5) ? std::stoi(argv[4]) : 3;
        
        bool use_pretrained_weights = !weights_dir.empty();
        
        // Use config from model_config.h (now set to Rene dimensions)
        const int ACTUAL_VOCAB_SIZE = VOCAB_SIZE;  // 50288
        const int ACTUAL_D_MODEL = D_MODEL;         // 2048
        int n_layer_repeats = N_LAYER_REPEATS;      // 4
        
        if (use_pretrained_weights) {
            std::cout << "Loading pretrained Rene weights from: " << weights_dir << std::endl;
            std::cout << "Model config: vocab=" << ACTUAL_VOCAB_SIZE << ", d_model=" << ACTUAL_D_MODEL 
                      << ", layers=" << (12 * n_layer_repeats) << std::endl;
        } else {
            std::cout << "Generating random weights for testing" << std::endl;
            std::cout << "Model config: vocab=" << ACTUAL_VOCAB_SIZE << ", d_model=" << ACTUAL_D_MODEL 
                      << ", layers=" << (12 * n_layer_repeats) << std::endl;
        }
        
        std::cout << "Using " << n_layer_repeats << " layer repeats (will create " 
                  << (12 * n_layer_repeats) << " layers)" << std::endl;
        
        // Initialize random seed for weight generation
        // Use fixed seed for reproducible weights (for MLX comparison)
        std::srand(42);  // Fixed seed instead of time-based
        
        // Read input tokens
        std::cout << "Reading token file: " << token_file << std::endl;
        std::vector<int32_t> prompt_tokens = readTokenFile(token_file);
        std::cout << "Loaded " << prompt_tokens.size() << " prompt tokens: [";
        for (size_t i = 0; i < prompt_tokens.size(); ++i) {
            std::cout << prompt_tokens[i];
            if (i < prompt_tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Initialize OpenCL first (needed for vocab size check)
        std::cout << "Initializing OpenCL..." << std::endl;
        auto& ctx_mgr = OpenCLContextManager::getInstance();
        ctx_mgr.initialize();
        std::cout << "✓ OpenCL initialized" << std::endl;
        
        // Clamp token IDs to valid vocab range
        for (int32_t& token_id : prompt_tokens) {
            if (token_id < 0) token_id = 0;
            if (token_id >= ACTUAL_VOCAB_SIZE) token_id = token_id % ACTUAL_VOCAB_SIZE;
        }
        std::cout << "Clamped token IDs to range [0, " << ACTUAL_VOCAB_SIZE << ")" << std::endl;
        
        cl_context context = ctx_mgr.getContext();
        cl_command_queue queue = ctx_mgr.getQueue();
        
        // Initialize model components
        std::cout << "Initializing model components..." << std::endl;
        
        // Embedding layer
        EmbeddingLayer embedding(&ctx_mgr, ACTUAL_VOCAB_SIZE, ACTUAL_D_MODEL);
        std::vector<float> embedding_weights;
        
        if (use_pretrained_weights) {
            std::cout << "Loading embedding weights..." << std::endl;
            embedding_weights = loadWeights(weights_dir + "/embedding_weight.bin");
            if (embedding_weights.size() != (size_t)(ACTUAL_VOCAB_SIZE * ACTUAL_D_MODEL)) {
                throw std::runtime_error("Embedding weight size mismatch");
            }
        } else {
            // Generate random weights for testing
            embedding_weights.resize(ACTUAL_VOCAB_SIZE * ACTUAL_D_MODEL);
            for (float& w : embedding_weights) {
                w = 0.01f * (std::rand() % 200 - 100) / 100.0f;
            }
        }
        
        embedding.initializeWeights(embedding_weights);
        std::cout << "✓ Embedding layer initialized (" << (ACTUAL_VOCAB_SIZE * ACTUAL_D_MODEL * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;
        
        // Sequence model
        SequenceModel seq_model(&ctx_mgr, D_MODEL, n_layer_repeats, false);
        
        // Build model
        // Pattern: 12 unique layers repeated n_layer_repeats times
        std::cout << "Building model with " << (12 * n_layer_repeats) << " layers..." << std::endl;
        
        // Print device memory info for diagnostics
        {
            cl_device_id device = ctx_mgr.getDevice();
            size_t max_alloc_size = 0;
            cl_ulong global_mem_size = 0;
            cl_int err = clGetDeviceInfo(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE, sizeof(size_t), &max_alloc_size, nullptr);
            clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &global_mem_size, nullptr);
            if (err == CL_SUCCESS) {
                std::cout << "Device memory: max_alloc=" << (max_alloc_size / 1024 / 1024) << " MB, "
                          << "global=" << (global_mem_size / 1024 / 1024) << " MB" << std::endl;
            }
        }
        
        // Helper function to create SSD layer with weights (loaded or generated)
        auto createSSDLayer = [&](int layer_idx, int expand, int kernel_size, int d_state, int d_head, int n_groups) -> SSDLayer* {
            SSDLayer* layer = new SSDLayer(&ctx_mgr, D_MODEL, expand, kernel_size, d_state, d_head, n_groups);
            
            int d_inner = D_MODEL * expand;
            int n_heads = d_inner / d_head;
            int in_proj_dim = 2 * d_inner + 2 * d_state * n_groups + n_heads;
            int conv_dim = d_inner + 2 * d_state * n_groups;
            
            std::vector<float> in_proj_weights, conv_weight, conv_bias, A, dt_bias, D, out_proj_weights;
            
            if (use_pretrained_weights) {
                // Load weights from files
                char layer_dir[256];
                snprintf(layer_dir, sizeof(layer_dir), "%s/layer_%03d_ssd", weights_dir.c_str(), layer_idx);
                in_proj_weights = loadWeights(std::string(layer_dir) + "/in_proj_weight.bin");
                conv_weight = loadWeights(std::string(layer_dir) + "/conv_weight.bin");
                conv_bias = loadWeights(std::string(layer_dir) + "/conv_bias.bin");
                A = loadWeights(std::string(layer_dir) + "/A_log.bin");
                dt_bias = loadWeights(std::string(layer_dir) + "/dt_bias.bin");
                D = loadWeights(std::string(layer_dir) + "/D.bin");
                out_proj_weights = loadWeights(std::string(layer_dir) + "/out_proj_weight.bin");
            } else {
                // Generate random weights
                in_proj_weights.resize(in_proj_dim * D_MODEL);
                conv_weight.resize(conv_dim * kernel_size);
                conv_bias.resize(conv_dim);
                A.resize(n_heads);
                dt_bias.resize(n_heads);
                D.resize(n_heads);
                out_proj_weights.resize(D_MODEL * d_inner);
                
                for (size_t i = 0; i < in_proj_weights.size(); ++i) in_proj_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
                for (size_t i = 0; i < conv_weight.size(); ++i) conv_weight[i] = 0.01f * ((i % 200) - 100) / 100.0f;
                for (size_t i = 0; i < conv_bias.size(); ++i) conv_bias[i] = 0.001f * ((i % 200) - 100) / 100.0f;
                for (size_t i = 0; i < A.size(); ++i) A[i] = 0.1f + 0.01f * (i % 100) / 100.0f;
                for (float& dt : dt_bias) dt = 0.0f;
                for (size_t i = 0; i < D.size(); ++i) D[i] = 0.1f * (i % 100) / 100.0f;
                for (size_t i = 0; i < out_proj_weights.size(); ++i) out_proj_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
            }
            
            layer->initializeWeights(in_proj_weights, conv_weight, conv_bias, A, dt_bias, D, out_proj_weights);
            return layer;
        };
        
        // Helper function to create SwiGLU layer with weights (loaded or generated)
        auto createSwiGLULayer = [&](int layer_idx, int expand) -> SwiGLULayer* {
            SwiGLULayer* layer = new SwiGLULayer(&ctx_mgr, D_MODEL, expand);
            
            int d_inner = D_MODEL * expand;
            std::vector<float> gate_weights, up_weights, down_weights;
            
            if (use_pretrained_weights) {
                // Load weights from files (MLX exports use "ffn" for SwiGLU)
                char layer_dir[256];
                snprintf(layer_dir, sizeof(layer_dir), "%s/layer_%03d", weights_dir.c_str(), layer_idx);
                gate_weights = loadWeights(std::string(layer_dir) + "/gate_weight.bin");
                up_weights = loadWeights(std::string(layer_dir) + "/up_weight.bin");
                down_weights = loadWeights(std::string(layer_dir) + "/down_weight.bin");
            } else {
                // Generate random weights
                gate_weights.resize(d_inner * D_MODEL);
                up_weights.resize(d_inner * D_MODEL);
                down_weights.resize(D_MODEL * d_inner);
                
                for (size_t i = 0; i < gate_weights.size(); ++i) gate_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
                for (size_t i = 0; i < up_weights.size(); ++i) up_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
                for (size_t i = 0; i < down_weights.size(); ++i) down_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
            }
            
            layer->initializeWeights(gate_weights, up_weights, down_weights);
            return layer;
        };
        
        // Helper function to create Attention layer with weights (loaded or generated)
        auto createAttentionLayer = [&](int layer_idx) -> AttentionLayer* {
            std::cout << "\n    [createAttentionLayer] Creating AttentionLayer object..." << std::flush;
            AttentionLayer* layer = new AttentionLayer(
                &ctx_mgr, D_MODEL, ATTENTION_N_HEADS, ATTENTION_N_HEADS, ATTENTION_HEAD_DIM, 4096, true
            );
            std::cout << " ✓" << std::endl;
            
            int d_proj = (ATTENTION_N_HEADS + 2 * ATTENTION_N_HEADS) * ATTENTION_HEAD_DIM;
            size_t qkv_size = d_proj * D_MODEL;
            size_t out_size = D_MODEL * ATTENTION_N_HEADS * ATTENTION_HEAD_DIM;
            
            std::vector<float> qkv_weights;
            std::vector<float> out_weights;
            
            try {
                if (use_pretrained_weights) {
                    // Load weights from files
                    char layer_dir[256];
                    snprintf(layer_dir, sizeof(layer_dir), "%s/layer_%03d", weights_dir.c_str(), layer_idx);
                    std::cout << "    [createAttentionLayer] Loading weights from " << layer_dir << "..." << std::flush;
                    qkv_weights = loadWeights(std::string(layer_dir) + "/qkv_weight.bin");
                    out_weights = loadWeights(std::string(layer_dir) + "/out_weight.bin");
                    std::cout << " ✓" << std::endl;
                } else {
                    // Generate random weights
                    std::cout << "    [createAttentionLayer] Generating weights (qkv: " << qkv_size << ", out: " << out_size << ")..." << std::flush;
                    
                    qkv_weights.resize(qkv_size);
                    out_weights.resize(out_size);
                    
                    for (size_t i = 0; i < qkv_weights.size(); ++i) {
                        qkv_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
                    }
                    for (size_t i = 0; i < out_weights.size(); ++i) {
                        out_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
                    }
                    std::cout << " ✓" << std::endl;
                }
            } catch (const std::bad_alloc& e) {
                std::cerr << "\n    FATAL: Out of memory allocating attention weights!" << std::endl;
                std::cerr << "    qkv_weights size: " << qkv_size << " floats (" << (qkv_size * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;
                std::cerr << "    out_weights size: " << out_size << " floats (" << (out_size * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;
                delete layer;
                throw std::runtime_error("Out of memory during attention weight allocation");
            }
            
            std::cout << "    [createAttentionLayer] Initializing weights..." << std::flush;
            try {
                layer->initializeWeights(qkv_weights, out_weights);
                std::cout << " ✓" << std::endl;
            } catch (const std::exception& e) {
                std::cerr << "\n    ERROR initializing attention weights: " << e.what() << std::endl;
                delete layer;
                throw;
            }
            return layer;
        };
        
        // Create all layers (12 unique layers × n_layer_repeats)
        int layer_count = 0;
        std::cout << "Starting layer creation loop (will create " << (12 * n_layer_repeats) << " layers)..." << std::endl;
        for (int repeat = 0; repeat < n_layer_repeats; ++repeat) {
            std::cout << "  Repeat " << (repeat + 1) << " of " << n_layer_repeats << "..." << std::endl;
            // Layer 0: SSD
            {
                SSDLayer* layer = createSSDLayer(layer_count, SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 1: SSD
            {
                SSDLayer* layer = createSSDLayer(layer_count, SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 2: SwiGLU
            {
                SwiGLULayer* layer = createSwiGLULayer(layer_count, SSD_EXPAND);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", false);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 3: SSD
            {
                SSDLayer* layer = createSSDLayer(layer_count, SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 4: SSD
            {
                SSDLayer* layer = createSSDLayer(layer_count, SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 5: SwiGLU
            {
                SwiGLULayer* layer = createSwiGLULayer(layer_count, SSD_EXPAND);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", false);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 6: Attention
            {
                std::cout << "  Creating Attention layer " << layer_count << "..." << std::flush;
                try {
                    AttentionLayer* layer = createAttentionLayer(layer_count);
                    std::cout << " ✓" << std::endl;
                    std::cout << "  Creating ResidualBlock for Attention..." << std::flush;
                    ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                    std::cout << " ✓" << std::endl;
                    std::cout << "  Adding Attention layer to sequence model..." << std::flush;
                    seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                    std::cout << " ✓" << std::endl;
                    layer_count++;
                } catch (const std::exception& e) {
                    std::cerr << "\n  ERROR creating Attention layer: " << e.what() << std::endl;
                    throw;
                }
            }
            
            // Layer 7: SSD
            {
                SSDLayer* layer = createSSDLayer(layer_count, SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 8: SwiGLU
            {
                SwiGLULayer* layer = createSwiGLULayer(layer_count, SSD_EXPAND);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", false);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 9: SSD
            {
                SSDLayer* layer = createSSDLayer(layer_count, SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 10: SSD
            {
                SSDLayer* layer = createSSDLayer(layer_count, SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 11: SwiGLU
            {
                SwiGLULayer* layer = createSwiGLULayer(layer_count, SSD_EXPAND);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", false);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            if ((repeat + 1) % 1 == 0) {
                std::cout << "  Added " << (repeat + 1) << " repeats (" << layer_count << " layers)..." << std::endl;
                std::cout.flush();  // Force output
            }
        }
        
        std::cout << "✓ Built full model with " << layer_count << " layers" << std::endl;
        std::cout.flush();
        
        // LM Head
        std::cout << "Initializing LM Head..." << std::flush;
        LMHead lm_head(&ctx_mgr, D_MODEL, ACTUAL_VOCAB_SIZE);
        std::cout << " ✓ (created)" << std::endl;
        
        std::vector<float> lm_weights;
        if (use_pretrained_weights) {
            std::cout << "  Loading LM Head weights..." << std::flush;
            lm_weights = loadWeights(weights_dir + "/lm_head_weight.bin");
            if (lm_weights.size() != (size_t)(ACTUAL_VOCAB_SIZE * D_MODEL)) {
                throw std::runtime_error("LM head weight size mismatch");
            }
        } else {
            std::cout << "  Generating LM Head weights..." << std::flush;
            lm_weights.resize(ACTUAL_VOCAB_SIZE * D_MODEL);
            for (float& w : lm_weights) {
                w = 0.01f * (std::rand() % 200 - 100) / 100.0f;
            }
        }
        std::cout << " ✓ (" << lm_weights.size() << " weights)" << std::endl;
        
        std::cout << "  Initializing LM Head..." << std::flush;
        lm_head.initializeWeights(lm_weights);
        std::cout << " ✓" << std::endl;
        std::cout << "✓ LM Head initialized (" << (ACTUAL_VOCAB_SIZE * D_MODEL * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;
        
        // Sampler
        Sampler sampler;
        std::cout << "✓ Sampler initialized" << std::endl;
        
        std::cout << "✓ Model initialization complete" << std::endl;
        std::cout << std::endl;
        
        // Create token buffer
        cl_mem token_buffer = createTokenBuffer(context, prompt_tokens);
        int batch_size = 1;
        int seq_len = prompt_tokens.size();
        
        // Prefill: Process prompt tokens
        std::cout << "Running prefill on " << seq_len << " tokens..." << std::endl;
        std::cout << "  Step 1: Encoding tokens..." << std::flush;
        cl_mem embeddings = embedding.encode(token_buffer, batch_size, seq_len, queue);
        std::cout << " ✓" << std::endl;
        
        // Check embedding output for NaN
        static bool checked_embedding = false;
        if (!checked_embedding) {
            size_t emb_size = batch_size * seq_len * ACTUAL_D_MODEL;
            std::vector<float> emb_check(emb_size);
            cl_int check_err = clEnqueueReadBuffer(queue, embeddings, CL_TRUE, 0,
                emb_size * sizeof(float), emb_check.data(), 0, nullptr, nullptr);
            if (check_err == CL_SUCCESS) {
                int nan_count = 0;
                for (float val : emb_check) {
                    if (std::isnan(val)) { nan_count++; }
                }
                std::cout << "  [Embedding Debug] Embedding output: " << nan_count 
                          << " NaNs out of " << emb_size << " values" << std::endl;
            }
            checked_embedding = true;
        }
        
    // Forward through sequence model
    std::cout << "  Step 2: Forward pass through " << seq_model.getNumLayers() << " layers..." << std::flush;
    std::vector<LayerState> states;  // Will be populated by stateful layers
    cl_mem hidden = seq_model.forward(embeddings, batch_size, seq_len, &states, queue);
    std::cout << " ✓" << std::endl;
    
    // Step 3: Get logits from last token and sample first generation token
    std::cout << "  Step 3: Computing logits from last token..." << std::flush;
    // LMHead expects [batch_size, d_model] as input and outputs [batch_size, vocab_size]
    // Since seq_model outputs [batch_size, seq_len, d_model], we treat all tokens as a batch
    int effective_batch_size = batch_size * seq_len;
    cl_mem prefill_logits = lm_head.forward(hidden, effective_batch_size, queue);
    std::cout << " ✓" << std::endl;
    
    // Extract logits for the last token in the sequence
    // The LM head outputs [batch_size * seq_len, vocab_size]
    std::vector<float> all_logits(effective_batch_size * ACTUAL_VOCAB_SIZE);
    cl_int err = clEnqueueReadBuffer(queue, prefill_logits, CL_TRUE, 0, 
                                     all_logits.size() * sizeof(float), 
                                     all_logits.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read prefill logits from device");
    }
    
    // Get last token's logits (last position in the effective batch)
    size_t last_token_offset = (seq_len - 1) * ACTUAL_VOCAB_SIZE;
    std::vector<float> last_token_logits(all_logits.begin() + last_token_offset, 
                                         all_logits.begin() + last_token_offset + ACTUAL_VOCAB_SIZE);
    
    // Sample first token from prefill logits
    std::cout << "  Step 4: Sampling first token from prefill logits..." << std::flush;
    int current_token_id = sampler.topPSample(last_token_logits, DEFAULT_TOP_P, DEFAULT_TEMPERATURE);
    std::cout << " ✓ (token=" << current_token_id << ")" << std::endl;
    
    // Release prefill logits buffer
    clReleaseMemObject(prefill_logits);
    
    // Generate tokens
    std::cout << "Generating " << max_tokens << " tokens..." << std::endl;
    cl_device_id device = ctx_mgr.getDevice();
    printMemoryInfo(device, "Before Generation");
    std::vector<int32_t> generated_tokens;
        
        for (int i = 0; i < max_tokens; ++i) {
            std::cout << "\n[Gen] Step " << (i+1) << " / " << max_tokens << std::endl;
            printMemoryInfo(device, "Step " + std::to_string(i+1));
            std::cout.flush();
            cl_mem current_token_buf = nullptr;
            cl_mem current_embedding = nullptr;
            cl_mem next_hidden = nullptr;
            cl_mem logits = nullptr;
            try {
                std::cout << "  [Gen] EncodeStep: token_id=" << current_token_id << std::endl;
                std::cout.flush();
                // Encode current token
                std::vector<int32_t> current_token_vec = {current_token_id};
                current_token_buf = createTokenBuffer(context, current_token_vec);
                current_embedding = embedding.encodeStep(current_token_buf, batch_size, queue);
                if (!current_embedding) throw std::runtime_error("encodeStep returned null buffer");
                std::cout << "  [Gen] EncodeStep ✓" << std::endl;
                
                // Check embedding for NaN (first iteration only)
                if (i == 0) {
                    checkForNaN(current_embedding, batch_size * ACTUAL_D_MODEL, queue, "embedding_output");
                }
                std::cout.flush();
                
                // Step through sequence model
                next_hidden = seq_model.step(current_embedding, batch_size, &states, queue);
                if (!next_hidden) throw std::runtime_error("seq_model.step returned null buffer");
                std::cout << "  [Gen] seq_model.step ✓" << std::endl;
                
                // Check hidden state for NaN (first iteration only)
                if (i == 0) {
                    checkForNaN(next_hidden, batch_size * ACTUAL_D_MODEL, queue, "hidden_state");
                }
                std::cout.flush();
                
                // Get logits from LM head
                logits = lm_head.forward(next_hidden, batch_size, queue);
                if (!logits) throw std::runtime_error("LMHead.forward returned null buffer");
                std::cout << "  [Gen] LMHead.forward ✓" << std::endl;
                std::cout.flush();
                
                // Ensure all writes are visible before CPU read in sampler
                clFinish(queue);
                std::cout << "  [Gen] clFinish ✓" << std::endl;
                std::cout.flush();
                
                // Sample next token
                int next_token = sampler.sampleFromBuffer(
                    logits, ACTUAL_VOCAB_SIZE, queue,
                    DEFAULT_TOP_P, DEFAULT_TEMPERATURE
                );
                std::cout << "  [Gen] sample ✓ -> token=" << next_token << std::endl;
                // printMemoryInfo(device, "After Step " + std::to_string(i+1));
                std::cout.flush();
                
                // Clamp token ID to valid range
                if (next_token >= ACTUAL_VOCAB_SIZE) {
                    next_token = next_token % ACTUAL_VOCAB_SIZE;
                }
                
                generated_tokens.push_back(next_token);
                current_token_id = next_token;
                
                // Check for EOS
                if (next_token == EOS_TOKEN_ID) {
                    std::cout << "Generated EOS token, stopping generation" << std::endl;
                    // Cleanup temporary buffers
                    if (current_token_buf) clReleaseMemObject(current_token_buf);
                    if (current_embedding) clReleaseMemObject(current_embedding);
                    if (next_hidden) clReleaseMemObject(next_hidden);
                    if (logits) clReleaseMemObject(logits);
                    break;
                }
                
                // Cleanup temporary buffers
                if (current_token_buf) { clReleaseMemObject(current_token_buf); current_token_buf = nullptr; }
                if (current_embedding) { clReleaseMemObject(current_embedding); current_embedding = nullptr; }
                if (next_hidden) { clReleaseMemObject(next_hidden); next_hidden = nullptr; }
                if (logits) { clReleaseMemObject(logits); logits = nullptr; }
                
                if ((i + 1) % 10 == 0) {
                    std::cout << "Generated " << (i + 1) << " tokens..." << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error during generation step " << (i + 1) << ": " << e.what() << std::endl;
                // Ensure we free any allocated buffers on error to avoid driver crashes
                if (current_token_buf) { clReleaseMemObject(current_token_buf); current_token_buf = nullptr; }
                if (current_embedding) { clReleaseMemObject(current_embedding); current_embedding = nullptr; }
                if (next_hidden) { clReleaseMemObject(next_hidden); next_hidden = nullptr; }
                if (logits) { clReleaseMemObject(logits); logits = nullptr; }
                break;
            }
        }
        
        std::cout << std::endl;
        std::cout << "Generation loop completed!" << std::endl;
        // printMemoryInfo(device, "After All Steps");
        std::cout.flush();
        
        // Cleanup - protect against double-release
        std::cout << "Cleaning up prefill buffers..." << std::flush;
        if (token_buffer) clReleaseMemObject(token_buffer);
        if (embeddings) clReleaseMemObject(embeddings);
        if (hidden) clReleaseMemObject(hidden);
        std::cout << " ✓" << std::endl;
        
        std::cout << std::endl;
        std::cout << "Generation complete!" << std::endl;
        std::cout << "Generated " << generated_tokens.size() << " tokens: [";
        for (size_t i = 0; i < generated_tokens.size(); ++i) {
            std::cout << generated_tokens[i];
            if (i < generated_tokens.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        
        // Write output (even if partially generated)
        if (!generated_tokens.empty()) {
            std::cout << "Writing output to: " << output_file << std::endl;
            writeTokenFile(output_file, generated_tokens);
            std::cout << "✓ Output written (" << generated_tokens.size() << " tokens)" << std::endl;
        } else {
            std::cout << "Warning: No tokens generated, skipping output file write" << std::endl;
        }
        
        // Cleanup OpenCL
        ctx_mgr.cleanup();
        
        std::cout << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Success!" << std::endl;
        std::cout << "========================================" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
