#include <iostream>
#include <fstream>
#include <vector>
#include <stdexcept>
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

using namespace cartesia_opencl;

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

// Helper: Write token IDs to binary file
void writeTokenFile(const std::string& filename, const std::vector<int32_t>& tokens) {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open output file: " + std::string(filename));
    }
    file.write(reinterpret_cast<const char*>(tokens.data()), tokens.size() * sizeof(int32_t));
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
            std::cerr << "Usage: " << argv[0] << " <token_file.bin> [output_file.bin] [max_tokens] [n_layer_repeats]" << std::endl;
            std::cerr << "  token_file.bin: Input file with token IDs (int32 binary)" << std::endl;
            std::cerr << "  output_file.bin: Output file for generated tokens (default: output_tokens.bin)" << std::endl;
            std::cerr << "  max_tokens: Maximum tokens to generate (default: 100)" << std::endl;
            std::cerr << "  n_layer_repeats: Number of layer repeats (1-" << N_LAYER_REPEATS 
                      << ", default: " << N_LAYER_REPEATS << "). "
                      << "Use 1 for testing to reduce memory usage." << std::endl;
            return 1;
        }
        
        std::string token_file = argv[1];
        std::string output_file = (argc >= 3) ? argv[2] : "output_tokens.bin";
        int max_tokens = (argc >= 4) ? std::stoi(argv[3]) : 100;
        int n_layer_repeats = (argc >= 5) ? std::stoi(argv[4]) : N_LAYER_REPEATS;
        
        if (n_layer_repeats < 1 || n_layer_repeats > N_LAYER_REPEATS) {
            std::cerr << "Warning: n_layer_repeats must be between 1 and " << N_LAYER_REPEATS 
                      << ". Using " << N_LAYER_REPEATS << std::endl;
            n_layer_repeats = N_LAYER_REPEATS;
        }
        
        std::cout << "Using " << n_layer_repeats << " layer repeats (will create " 
                  << (12 * n_layer_repeats) << " layers)" << std::endl;
        
        // Initialize random seed for weight generation
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        
        // Read input tokens
        std::cout << "Reading token file: " << token_file << std::endl;
        std::vector<int32_t> prompt_tokens = readTokenFile(token_file);
        std::cout << "Loaded " << prompt_tokens.size() << " prompt tokens" << std::endl;
        
        // Initialize OpenCL first (needed for vocab size check)
        std::cout << "Initializing OpenCL..." << std::endl;
        auto& ctx_mgr = OpenCLContextManager::getInstance();
        ctx_mgr.initialize();
        std::cout << "✓ OpenCL initialized" << std::endl;
        
        // Clamp token IDs to valid test vocab range
        constexpr int TEST_VOCAB_SIZE = 1000;  // Reduced for testing
        for (int32_t& token_id : prompt_tokens) {
            if (token_id < 0) token_id = 0;
            if (token_id >= TEST_VOCAB_SIZE) token_id = token_id % TEST_VOCAB_SIZE;
        }
        std::cout << "Clamped token IDs to range [0, " << TEST_VOCAB_SIZE << ")" << std::endl;
        
        cl_context context = ctx_mgr.getContext();
        cl_command_queue queue = ctx_mgr.getQueue();
        
        // Initialize model components
        std::cout << "Initializing model components..." << std::endl;
        std::cout << "Using test vocab size: " << TEST_VOCAB_SIZE << " (full model: " << VOCAB_SIZE << ")" << std::endl;
        
        // Embedding layer
        
        EmbeddingLayer embedding(&ctx_mgr, TEST_VOCAB_SIZE, D_MODEL);
        // Initialize with small random weights for testing
        std::vector<float> embedding_weights(TEST_VOCAB_SIZE * D_MODEL);
        for (float& w : embedding_weights) {
            w = 0.01f * (std::rand() % 200 - 100) / 100.0f;
        }
        embedding.initializeWeights(embedding_weights);
        std::cout << "✓ Embedding layer initialized (" << (TEST_VOCAB_SIZE * D_MODEL * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;
        
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
        
        // Helper function to create SSD layer with default weights
        auto createSSDLayer = [&](int expand, int kernel_size, int d_state, int d_head, int n_groups) -> SSDLayer* {
            SSDLayer* layer = new SSDLayer(&ctx_mgr, D_MODEL, expand, kernel_size, d_state, d_head, n_groups);
            
            int d_inner = D_MODEL * expand;
            int n_heads = d_inner / d_head;
            int in_proj_dim = 2 * d_inner + 2 * d_state * n_groups + n_heads;
            int conv_dim = d_inner + 2 * d_state * n_groups;
            
            // Initialize with small random-like weights for testing
            std::vector<float> in_proj_weights(in_proj_dim * D_MODEL);
            std::vector<float> conv_weight(conv_dim * kernel_size);
            std::vector<float> conv_bias(conv_dim);
            std::vector<float> A(n_heads);
            std::vector<float> dt_bias(n_heads);
            std::vector<float> D(n_heads);
            std::vector<float> out_proj_weights(D_MODEL * d_inner);
            
            // Fill with small values using fast deterministic pattern (instead of slow random)
            for (size_t i = 0; i < in_proj_weights.size(); ++i) in_proj_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
            for (size_t i = 0; i < conv_weight.size(); ++i) conv_weight[i] = 0.01f * ((i % 200) - 100) / 100.0f;
            for (size_t i = 0; i < conv_bias.size(); ++i) conv_bias[i] = 0.001f * ((i % 200) - 100) / 100.0f;
            for (size_t i = 0; i < A.size(); ++i) A[i] = 0.1f + 0.01f * (i % 100) / 100.0f;  // Positive values for A
            for (float& dt : dt_bias) dt = 0.0f;
            for (size_t i = 0; i < D.size(); ++i) D[i] = 0.1f * (i % 100) / 100.0f;
            for (size_t i = 0; i < out_proj_weights.size(); ++i) out_proj_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
            
            layer->initializeWeights(in_proj_weights, conv_weight, conv_bias, A, dt_bias, D, out_proj_weights);
            return layer;
        };
        
        // Helper function to create SwiGLU layer
        auto createSwiGLULayer = [&](int expand) -> SwiGLULayer* {
            SwiGLULayer* layer = new SwiGLULayer(&ctx_mgr, D_MODEL, expand);
            
            int d_inner = D_MODEL * expand;
            std::vector<float> gate_weights(d_inner * D_MODEL);
            std::vector<float> up_weights(d_inner * D_MODEL);
            std::vector<float> down_weights(D_MODEL * d_inner);
            
            // Fast fill using deterministic pattern
            for (size_t i = 0; i < gate_weights.size(); ++i) gate_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
            for (size_t i = 0; i < up_weights.size(); ++i) up_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
            for (size_t i = 0; i < down_weights.size(); ++i) down_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
            
            layer->initializeWeights(gate_weights, up_weights, down_weights);
            return layer;
        };
        
        // Helper function to create Attention layer
        auto createAttentionLayer = [&]() -> AttentionLayer* {
            std::cout << "\n    [createAttentionLayer] Creating AttentionLayer object..." << std::flush;
            AttentionLayer* layer = new AttentionLayer(
                &ctx_mgr, D_MODEL, ATTENTION_N_HEADS, ATTENTION_N_HEADS, ATTENTION_HEAD_DIM, 4096, true
            );
            std::cout << " ✓" << std::endl;
            
            int d_proj = (ATTENTION_N_HEADS + 2 * ATTENTION_N_HEADS) * ATTENTION_HEAD_DIM;
            std::cout << "    [createAttentionLayer] Generating weights (d_proj=" << d_proj << ")..." << std::flush;
            size_t qkv_size = d_proj * D_MODEL;
            size_t out_size = D_MODEL * ATTENTION_N_HEADS * ATTENTION_HEAD_DIM;
            std::cout << " (qkv: " << qkv_size << ", out: " << out_size << ")..." << std::flush;
            std::cout.flush();
            
            std::vector<float> qkv_weights;
            std::vector<float> out_weights;
            
            try {
                std::cout << "\n      Allocating qkv_weights (" << (qkv_size * sizeof(float) / 1024 / 1024) << " MB)..." << std::flush;
                qkv_weights.resize(qkv_size);
                std::cout << " ✓" << std::flush;
                
                std::cout << "\n      Filling qkv_weights (fast fill)..." << std::flush;
                // Fast fill: use simple pattern instead of random for each element
                for (size_t i = 0; i < qkv_weights.size(); ++i) {
                    // Use a deterministic pattern based on index for speed
                    qkv_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
                }
                std::cout << " ✓" << std::flush;
                
                std::cout << "\n      Allocating out_weights (" << (out_size * sizeof(float) / 1024 / 1024) << " MB)..." << std::flush;
                out_weights.resize(out_size);
                std::cout << " ✓" << std::flush;
                
                std::cout << "\n      Filling out_weights (fast fill)..." << std::flush;
                // Fast fill: use simple pattern instead of random
                for (size_t i = 0; i < out_weights.size(); ++i) {
                    out_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
                }
                std::cout << " ✓" << std::endl;
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
                SSDLayer* layer = createSSDLayer(SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 1: SSD
            {
                SSDLayer* layer = createSSDLayer(SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 2: SwiGLU
            {
                SwiGLULayer* layer = createSwiGLULayer(SSD_EXPAND);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", false);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 3: SSD
            {
                SSDLayer* layer = createSSDLayer(SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 4: SSD
            {
                SSDLayer* layer = createSSDLayer(SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 5: SwiGLU
            {
                SwiGLULayer* layer = createSwiGLULayer(SSD_EXPAND);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", false);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 6: Attention
            {
                std::cout << "  Creating Attention layer " << layer_count << "..." << std::flush;
                try {
                    AttentionLayer* layer = createAttentionLayer();
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
                SSDLayer* layer = createSSDLayer(SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 8: SwiGLU
            {
                SwiGLULayer* layer = createSwiGLULayer(SSD_EXPAND);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", false);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 9: SSD
            {
                SSDLayer* layer = createSSDLayer(SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 10: SSD
            {
                SSDLayer* layer = createSSDLayer(SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
                ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, D_MODEL, "pre", true);
                seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
                layer_count++;
            }
            
            // Layer 11: SwiGLU
            {
                SwiGLULayer* layer = createSwiGLULayer(SSD_EXPAND);
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
        
        // LM Head - use same reduced vocab size for testing
        std::cout << "Initializing LM Head..." << std::flush;
        LMHead lm_head(&ctx_mgr, D_MODEL, TEST_VOCAB_SIZE);
        std::cout << " ✓ (created)" << std::endl;
        std::cout << "  Creating LM Head weights..." << std::flush;
        std::vector<float> lm_weights(TEST_VOCAB_SIZE * D_MODEL);
        for (float& w : lm_weights) {
            w = 0.01f * (std::rand() % 200 - 100) / 100.0f;
        }
        std::cout << " ✓ (generated " << lm_weights.size() << " weights)" << std::endl;
        std::cout << "  Initializing LM Head weights..." << std::flush;
        lm_head.initializeWeights(lm_weights);
        std::cout << " ✓" << std::endl;
        std::cout << "✓ LM Head initialized (" << (TEST_VOCAB_SIZE * D_MODEL * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;
        
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
        
        // Forward through sequence model
        std::cout << "  Step 2: Forward pass through " << seq_model.getNumLayers() << " layers..." << std::flush;
        std::vector<LayerState> states;  // Will be populated by stateful layers
        cl_mem hidden = seq_model.forward(embeddings, batch_size, seq_len, &states, queue);
        std::cout << " ✓" << std::endl;
        
        // Get last token's hidden state for generation
        // For simplicity, we'll use the last token's embedding directly
        // In a full implementation, we'd extract the last token's hidden state
        
        // Generate tokens
        std::cout << "Generating " << max_tokens << " tokens..." << std::endl;
        std::vector<int32_t> generated_tokens;
        
        // For the first step, we use the last token from prefill
        // TODO: Extract last token properly from hidden states
        int current_token_id = prompt_tokens.back();  // Placeholder - should be sampled from last logits
        
        for (int i = 0; i < max_tokens; ++i) {
            try {
                // Encode current token
                std::vector<int32_t> current_token_vec = {current_token_id};
                cl_mem current_token_buf = createTokenBuffer(context, current_token_vec);
                cl_mem current_embedding = embedding.encodeStep(current_token_buf, batch_size, queue);
                
                // Step through sequence model
                cl_mem next_hidden = seq_model.step(current_embedding, batch_size, &states, queue);
                
                // Get logits from LM head
                cl_mem logits = lm_head.forward(next_hidden, batch_size, queue);
                
                // Sample next token (use TEST_VOCAB_SIZE)
                int next_token = sampler.sampleFromBuffer(
                    logits, TEST_VOCAB_SIZE, queue,
                    DEFAULT_TOP_P, DEFAULT_TEMPERATURE
                );
                
                // Clamp token ID to valid range
                if (next_token >= TEST_VOCAB_SIZE) {
                    next_token = next_token % TEST_VOCAB_SIZE;
                }
                
                generated_tokens.push_back(next_token);
                current_token_id = next_token;
                
                // Check for EOS
                if (next_token == EOS_TOKEN_ID) {
                    std::cout << "Generated EOS token, stopping generation" << std::endl;
                    break;
                }
                
                // Cleanup temporary buffers
                clReleaseMemObject(current_token_buf);
                clReleaseMemObject(current_embedding);
                clReleaseMemObject(next_hidden);
                clReleaseMemObject(logits);
                
                if ((i + 1) % 10 == 0) {
                    std::cout << "Generated " << (i + 1) << " tokens..." << std::endl;
                }
            } catch (const std::exception& e) {
                std::cerr << "Error during generation step " << (i + 1) << ": " << e.what() << std::endl;
                break;
            }
        }
        
        // Cleanup
        clReleaseMemObject(token_buffer);
        clReleaseMemObject(embeddings);
        clReleaseMemObject(hidden);
        
        std::cout << std::endl;
        std::cout << "Generation complete!" << std::endl;
        std::cout << "Generated " << generated_tokens.size() << " tokens" << std::endl;
        
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
