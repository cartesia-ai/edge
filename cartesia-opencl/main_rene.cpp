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
#include "src/tokenizer.h"
#include "src/debug.h"
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>

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
        throw std::runtime_error("Failed to read weight file completely: " + filename);
    }
    
    return weights;
}

// Helper: Validate weight dimensions with helpful error message
void validateWeightSize(const std::vector<float>& weights, size_t expected_size, 
                       const std::string& weight_name, const std::string& layer_info = "") {
    if (weights.size() != expected_size) {
        std::string error_msg = "Weight size mismatch for " + weight_name;
        if (!layer_info.empty()) {
            error_msg += " (" + layer_info + ")";
        }
        error_msg += ": expected " + std::to_string(expected_size) + 
                     " weights, got " + std::to_string(weights.size());
        throw std::runtime_error(error_msg);
    }
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
            std::cerr << "Usage: " << argv[0] << " <token_file.bin|text> [weights_dir] [output_file.bin] [max_tokens]" << std::endl;
            std::cerr << "  token_file.bin: Input file with token IDs (int32 binary)" << std::endl;
            std::cerr << "  text: English text to tokenize (if not a file path)" << std::endl;
            std::cerr << "  weights_dir: Directory containing model weights (optional, generates random if not provided)" << std::endl;
            std::cerr << "  output_file.bin: Output file for generated tokens (default: /data/local/tmp/output_tokens.bin)" << std::endl;
            std::cerr << "  max_tokens: Maximum tokens to generate (default: 3)" << std::endl;
            return 1;
        }
        
        std::string input_arg = argv[1];
        std::string weights_dir = (argc >= 3) ? argv[2] : "";
        std::string output_file = (argc >= 4) ? argv[3] : "/data/local/tmp/output_tokens.bin";
        int max_tokens = (argc >= 5) ? std::stoi(argv[4]) : 3;
        
        bool use_pretrained_weights = !weights_dir.empty();
        bool is_text_input = false;
        
        // Try to read model config from metadata.json if available
        int ACTUAL_VOCAB_SIZE = VOCAB_SIZE;  // Default: 50288
        int ACTUAL_D_MODEL = D_MODEL;         // Default: 1024
        int n_layer_repeats = N_LAYER_REPEATS; // Default: 1
        
        if (use_pretrained_weights) {
            std::string metadata_file = weights_dir + "/metadata.json";
            std::ifstream metadata_stream(metadata_file);
            if (metadata_stream.is_open()) {
                std::string json_content((std::istreambuf_iterator<char>(metadata_stream)),
                                       std::istreambuf_iterator<char>());
                metadata_stream.close();
                
                // Simple JSON parsing for metadata.json
                // Look for "n_tokens" and "d_model" fields
                auto parse_json_int = [](const std::string& json, const std::string& key) -> int {
                    std::string search_key = "\"" + key + "\"";
                    size_t pos = json.find(search_key);
                    if (pos == std::string::npos) return -1;
                    size_t colon_pos = json.find(':', pos);
                    if (colon_pos == std::string::npos) return -1;
                    // Skip whitespace after colon
                    size_t start = colon_pos + 1;
                    while (start < json.size() && (json[start] == ' ' || json[start] == '\t')) start++;
                    // Find end of number (comma, newline, or closing brace)
                    size_t end = start;
                    while (end < json.size() && json[end] >= '0' && json[end] <= '9') end++;
                    if (end > start) {
                        return std::stoi(json.substr(start, end - start));
                    }
                    return -1;
                };
                
                int parsed_n_tokens = parse_json_int(json_content, "n_tokens");
                if (parsed_n_tokens > 0) {
                    ACTUAL_VOCAB_SIZE = parsed_n_tokens;
                }
                
                int parsed_d_model = parse_json_int(json_content, "d_model");
                if (parsed_d_model > 0) {
                    ACTUAL_D_MODEL = parsed_d_model;
                }
                
                std::cout << "Read model config from metadata.json: vocab=" << ACTUAL_VOCAB_SIZE 
                          << ", d_model=" << ACTUAL_D_MODEL << std::endl;
            } else {
                std::cout << "No metadata.json found, using defaults: vocab=" << ACTUAL_VOCAB_SIZE 
                          << ", d_model=" << ACTUAL_D_MODEL << std::endl;
            }
        }
        
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
        
        // Try to locate and load tokenizer (for both text input and decoding output)
        BPETokenizer tokenizer;
        bool tokenizer_loaded = false;
        
        if (use_pretrained_weights) {
            // Try to locate tokenizer files
            std::string tokenizer_dir = weights_dir + "/tokenizer";
            std::string vocab_path = tokenizer_dir + "/vocab.json";
            std::string merges_path = tokenizer_dir + "/merges.txt";
            
            // Check if tokenizer files exist in tokenizer subdirectory
            std::ifstream vocab_test(vocab_path);
            std::ifstream merges_test(merges_path);
            
            if (!vocab_test.good() || !merges_test.good()) {
                // Try weights_dir directly
                vocab_path = weights_dir + "/vocab.json";
                merges_path = weights_dir + "/merges.txt";
                vocab_test.close();
                merges_test.close();
                
                vocab_test.open(vocab_path);
                merges_test.open(merges_path);
            }
            
            if (vocab_test.good() && merges_test.good()) {
                vocab_test.close();
                merges_test.close();
                
                // Load tokenizer
                std::cout << "Loading tokenizer from: " << vocab_path << " and " << merges_path << std::endl;
                if (tokenizer.loadFromFiles(vocab_path, merges_path)) {
                    std::cout << "✓ Tokenizer loaded (vocab size: " << tokenizer.getVocabSize() << ")" << std::endl;
                    tokenizer_loaded = true;
                } else {
                    std::cout << "Warning: Failed to load tokenizer files" << std::endl;
                }
            } else {
                vocab_test.close();
                merges_test.close();
                std::cout << "Note: Tokenizer files not found (vocab.json/merges.txt)" << std::endl;
            }
        }
        
        // Detect if input is a file or text
        std::vector<int32_t> prompt_tokens;
        std::ifstream test_file(input_arg);
        if (test_file.good()) {
            // File exists - read binary tokens (backward compatible)
            test_file.close();
            std::cout << "Reading token file: " << input_arg << std::endl;
            prompt_tokens = readTokenFile(input_arg);
            DEBUG_TOKENS({
                std::cout << "Loaded " << prompt_tokens.size() << " prompt tokens: [";
                for (size_t i = 0; i < prompt_tokens.size(); ++i) {
                    std::cout << prompt_tokens[i];
                    if (i < prompt_tokens.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            });
        } else {
            // Treat as text input - tokenize it
            is_text_input = true;
            std::cout << "Detected text input: \"" << input_arg << "\"" << std::endl;
            
            if (!tokenizer_loaded) {
                std::cerr << "Error: Text input requires tokenizer files (vocab.json/merges.txt)" << std::endl;
                std::cerr << "Expected locations:" << std::endl;
                if (use_pretrained_weights) {
                    std::cerr << "  " << weights_dir << "/tokenizer/vocab.json and merges.txt" << std::endl;
                    std::cerr << "  Or: " << weights_dir << "/vocab.json and merges.txt" << std::endl;
                } else {
                    std::cerr << "  Provide weights_dir argument with tokenizer files" << std::endl;
                }
                return 1;
            }
            
            // Tokenize text
            std::cout << "Tokenizing text..." << std::endl;
            prompt_tokens = tokenizer.tokenize(input_arg);
            
            if (prompt_tokens.empty()) {
                std::cerr << "Error: Tokenization failed or produced no tokens" << std::endl;
                return 1;
            }
            
            DEBUG_TOKENS({
                std::cout << "Tokenized to " << prompt_tokens.size() << " tokens: [";
                for (size_t i = 0; i < prompt_tokens.size(); ++i) {
                    std::cout << prompt_tokens[i];
                    if (i < prompt_tokens.size() - 1) std::cout << ", ";
                }
                std::cout << "]" << std::endl;
            });
        }
        
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
        auto weight_load_start = std::chrono::high_resolution_clock::now();
        
        // Embedding layer
        EmbeddingLayer embedding(&ctx_mgr, ACTUAL_VOCAB_SIZE, ACTUAL_D_MODEL);
        std::vector<float> embedding_weights;
        
        if (use_pretrained_weights) {
            std::cout << "Loading embedding weights..." << std::endl;
            embedding_weights = loadWeights(weights_dir + "/embedding_weight.bin");
            validateWeightSize(embedding_weights, ACTUAL_VOCAB_SIZE * ACTUAL_D_MODEL, 
                             "embedding", "vocab=" + std::to_string(ACTUAL_VOCAB_SIZE) + 
                             ", d_model=" + std::to_string(ACTUAL_D_MODEL));
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
        // Mamba2-130m uses post_norm = true (based on MLX model configuration)
        SequenceModel seq_model(&ctx_mgr, ACTUAL_D_MODEL, n_layer_repeats, true);
        
        // Load post-norm weights
        if (use_pretrained_weights) {
            std::string post_norm_path = weights_dir + "/post_norm_weight.bin";
            std::vector<float> post_norm_weights = loadWeights(post_norm_path);
            validateWeightSize(post_norm_weights, ACTUAL_D_MODEL, "post-norm", "");
            seq_model.setPostNormWeights(post_norm_weights);
            std::cout << "✓ Loaded post-norm weights (" << post_norm_weights.size() << " values)" << std::endl;
        }
        
        // Build model
        // Pattern: 12 unique layers repeated n_layer_repeats times
        std::cout << "Building model with " << (12 * n_layer_repeats) << " layers..." << std::endl;
        
        
        // Helper function to create SSD layer with weights (loaded or generated)
        auto createSSDLayer = [&](int layer_idx, int expand, int kernel_size, int d_state, int d_head, int n_groups) -> SSDLayer* {
            std::vector<float> in_proj_weights, conv_weight, conv_bias, A, dt_bias, D, out_proj_weights, rms_norm_weights;
            int d_inner, n_heads, in_proj_dim, conv_dim;
            std::string layer_dir_str;
            
            if (use_pretrained_weights) {
                // Load weights from files first to determine actual dimensions
                char layer_dir[256];
                snprintf(layer_dir, sizeof(layer_dir), "%s/layer_%03d_ssd", weights_dir.c_str(), layer_idx);
                layer_dir_str = std::string(layer_dir);
                
                // Load weights to determine actual dimensions
                in_proj_weights = loadWeights(layer_dir_str + "/in_proj_weight.bin");
                out_proj_weights = loadWeights(layer_dir_str + "/out_proj_weight.bin");
                conv_weight = loadWeights(layer_dir_str + "/conv_weight.bin");
                
                // Calculate actual dimensions from file sizes
                in_proj_dim = in_proj_weights.size() / ACTUAL_D_MODEL;
                d_inner = out_proj_weights.size() / ACTUAL_D_MODEL;
                conv_dim = conv_weight.size() / kernel_size;
                
                // Calculate d_state and d_head from actual dimensions
                // conv_dim = d_inner + 2 * d_state * n_groups
                // So: d_state = (conv_dim - d_inner) / (2 * n_groups)
                int calculated_d_state = (conv_dim - d_inner) / (2 * n_groups);
                // in_proj_dim = 2 * d_inner + 2 * d_state * n_groups + n_heads
                // So: n_heads = in_proj_dim - 2 * d_inner - 2 * d_state * n_groups
                int calculated_n_heads = in_proj_dim - 2 * d_inner - 2 * calculated_d_state * n_groups;
                // d_head = d_inner / n_heads
                int calculated_d_head = d_inner / calculated_n_heads;
                
                // Use calculated values if they're valid, otherwise use defaults
                if (calculated_d_state > 0 && calculated_d_head > 0 && calculated_n_heads > 0) {
                    d_state = calculated_d_state;
                    d_head = calculated_d_head;
                    n_heads = calculated_n_heads;
                } else {
                    // Fall back to defaults
                    d_inner = ACTUAL_D_MODEL * expand;
                    n_heads = d_inner / d_head;
                    in_proj_dim = 2 * d_inner + 2 * d_state * n_groups + n_heads;
                    conv_dim = d_inner + 2 * d_state * n_groups;
                }
            } else {
                // Use provided defaults for generated weights
                d_inner = ACTUAL_D_MODEL * expand;
                n_heads = d_inner / d_head;
                in_proj_dim = 2 * d_inner + 2 * d_state * n_groups + n_heads;
                conv_dim = d_inner + 2 * d_state * n_groups;
            }
            
            SSDLayer* layer = new SSDLayer(&ctx_mgr, ACTUAL_D_MODEL, expand, kernel_size, d_state, d_head, n_groups);
            
            if (use_pretrained_weights) {
                // Validate sizes now that we know the actual dimensions
                validateWeightSize(in_proj_weights, in_proj_dim * ACTUAL_D_MODEL, 
                                 "SSD in_proj", "layer " + std::to_string(layer_idx));
                
                // conv_weight was already loaded above
                validateWeightSize(conv_weight, conv_dim * kernel_size, 
                                 "SSD conv_weight", "layer " + std::to_string(layer_idx));
                
                conv_bias = loadWeights(layer_dir_str + "/conv_bias.bin");
                validateWeightSize(conv_bias, conv_dim, 
                                 "SSD conv_bias", "layer " + std::to_string(layer_idx));
                
                A = loadWeights(layer_dir_str + "/A_log.bin");
                validateWeightSize(A, n_heads, 
                                 "SSD A_log", "layer " + std::to_string(layer_idx));
                
                dt_bias = loadWeights(layer_dir_str + "/dt_bias.bin");
                validateWeightSize(dt_bias, n_heads, 
                                 "SSD dt_bias", "layer " + std::to_string(layer_idx));
                
                D = loadWeights(layer_dir_str + "/D.bin");
                validateWeightSize(D, n_heads, 
                                 "SSD D", "layer " + std::to_string(layer_idx));
                
                out_proj_weights = loadWeights(layer_dir_str + "/out_proj_weight.bin");
                validateWeightSize(out_proj_weights, ACTUAL_D_MODEL * d_inner, 
                                 "SSD out_proj", "layer " + std::to_string(layer_idx));
                
                // Load SSD's internal rms_norm weights (different from ResidualBlock's norm!)
                rms_norm_weights = loadWeights(layer_dir_str + "/rms_norm_weight.bin");
                validateWeightSize(rms_norm_weights, d_inner, 
                                 "SSD rms_norm", "layer " + std::to_string(layer_idx));
            } else {
                // Generate random weights
                in_proj_weights.resize(in_proj_dim * ACTUAL_D_MODEL);
                conv_weight.resize(conv_dim * kernel_size);
                conv_bias.resize(conv_dim);
                A.resize(n_heads);
                dt_bias.resize(n_heads);
                D.resize(n_heads);
                out_proj_weights.resize(ACTUAL_D_MODEL * d_inner);
                rms_norm_weights.resize(d_inner);
                
                for (size_t i = 0; i < in_proj_weights.size(); ++i) in_proj_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
                for (size_t i = 0; i < conv_weight.size(); ++i) conv_weight[i] = 0.01f * ((i % 200) - 100) / 100.0f;
                for (size_t i = 0; i < conv_bias.size(); ++i) conv_bias[i] = 0.001f * ((i % 200) - 100) / 100.0f;
                for (size_t i = 0; i < A.size(); ++i) A[i] = 0.1f + 0.01f * (i % 100) / 100.0f;
                for (float& dt : dt_bias) dt = 0.0f;
                for (size_t i = 0; i < D.size(); ++i) D[i] = 0.1f * (i % 100) / 100.0f;
                for (size_t i = 0; i < out_proj_weights.size(); ++i) out_proj_weights[i] = 0.01f * ((i % 200) - 100) / 100.0f;
                for (size_t i = 0; i < rms_norm_weights.size(); ++i) rms_norm_weights[i] = 1.0f;  // RMS norm typically initialized to ones
            }
            
            layer->initializeWeights(in_proj_weights, conv_weight, conv_bias, A, dt_bias, D, out_proj_weights, rms_norm_weights);
            return layer;
        };
        
        // Helper function to create SwiGLU layer with weights (loaded or generated)
        auto createSwiGLULayer = [&](int layer_idx, int expand) -> SwiGLULayer* {
            SwiGLULayer* layer = new SwiGLULayer(&ctx_mgr, ACTUAL_D_MODEL, expand);
            
            int d_inner = ACTUAL_D_MODEL * expand;
            std::vector<float> gate_weights, up_weights, down_weights;
            
            if (use_pretrained_weights) {
                // Load weights from files
                // MLX SwiGLU exports: in_proj_weight.bin, gate_weight.bin, out_proj_weight.bin
                // OpenCL expects: gate_proj (gate), up_proj (maps to in_proj), down_proj (maps to out_proj)
                char layer_dir[256];
                snprintf(layer_dir, sizeof(layer_dir), "%s/layer_%03d", weights_dir.c_str(), layer_idx);
                std::string layer_dir_str(layer_dir);
                
                // Load gate_weight (gate_proj)
                gate_weights = loadWeights(layer_dir_str + "/gate_weight.bin");
                validateWeightSize(gate_weights, d_inner * ACTUAL_D_MODEL, 
                                 "SwiGLU gate", "layer " + std::to_string(layer_idx));
                
                // Load in_proj_weight as up_weights (MLX in_proj maps to OpenCL up_proj)
                up_weights = loadWeights(layer_dir_str + "/in_proj_weight.bin");
                validateWeightSize(up_weights, d_inner * ACTUAL_D_MODEL, 
                                 "SwiGLU in_proj (up)", "layer " + std::to_string(layer_idx));
                
                // Load out_proj_weight as down_weights (MLX out_proj maps to OpenCL down_proj)
                down_weights = loadWeights(layer_dir_str + "/out_proj_weight.bin");
                validateWeightSize(down_weights, ACTUAL_D_MODEL * d_inner, 
                                 "SwiGLU out_proj (down)", "layer " + std::to_string(layer_idx));
            } else {
                // Generate random weights
                gate_weights.resize(d_inner * ACTUAL_D_MODEL);
                up_weights.resize(d_inner * ACTUAL_D_MODEL);
                down_weights.resize(ACTUAL_D_MODEL * d_inner);
                
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
            
            // For pretrained weights, infer attention dimensions from loaded weights
            // For generated weights, use config defaults
            int n_heads = ATTENTION_N_HEADS;
            int head_dim = ATTENTION_HEAD_DIM;
            int d_proj;
            size_t qkv_size;
            size_t out_size;
            
            std::vector<float> qkv_weights;
            std::vector<float> out_weights;
            
            try {
                if (use_pretrained_weights) {
                    // Load weights first to infer dimensions
                    char layer_dir[256];
                    snprintf(layer_dir, sizeof(layer_dir), "%s/layer_%03d", weights_dir.c_str(), layer_idx);
                    std::string layer_dir_str(layer_dir);
                    std::cout << "    [createAttentionLayer] Loading weights from " << layer_dir_str << "..." << std::flush;
                    
                    qkv_weights = loadWeights(layer_dir_str + "/qkv_weight.bin");
                    out_weights = loadWeights(layer_dir_str + "/out_weight.bin");
                    
                    // Infer dimensions from loaded weights
                    // qkv_weight shape: [d_proj, d_model]
                    // So: d_proj = qkv_weights.size() / ACTUAL_D_MODEL
                    d_proj = qkv_weights.size() / ACTUAL_D_MODEL;
                    
                    // MLX out_proj shape: [d_model, n_heads * d_head] (not [d_model, d_proj])
                    // So we infer the actual output dimension from out_weights
                    int out_dim = out_weights.size() / ACTUAL_D_MODEL;
                    
                    // For MLX: out_dim = n_heads * d_head
                    // We can use this to verify n_heads and head_dim
                    // out_dim should equal n_heads * head_dim (after we infer them)
                    
                    // Infer n_heads and head_dim from d_proj and out_dim
                    // d_proj = (n_heads + 2 * kv_heads) * head_dim
                    // For standard attention: kv_heads = n_heads, so d_proj = 3 * n_heads * head_dim
                    // out_dim = n_heads * head_dim (from MLX out_proj shape)
                    // So: d_proj = 3 * out_dim, which means out_dim = d_proj / 3
                    // Then we can infer n_heads and head_dim from out_dim
                    
                    // Verify: out_dim should equal d_proj / 3 (for standard attention)
                    int expected_out_dim = d_proj / 3;
                    if (out_dim != expected_out_dim) {
                        std::cerr << "\n      [Warning] out_dim (" << out_dim << ") != d_proj/3 (" << expected_out_dim << ")" << std::flush;
                        std::cerr << "\n      This might indicate non-standard attention configuration" << std::flush;
                    }
                    
                    // Try to find n_heads and head_dim that satisfy: n_heads * head_dim = out_dim
                    bool found = false;
                    for (int try_n_heads = 1; try_n_heads <= 32 && !found; try_n_heads++) {
                        if (out_dim % try_n_heads == 0) {
                            int try_head_dim = out_dim / try_n_heads;
                            // Verify it also works with d_proj
                            if (d_proj == 3 * try_n_heads * try_head_dim) {
                                n_heads = try_n_heads;
                                head_dim = try_head_dim;
                                found = true;
                                std::cout << "\n      [Inferred] n_heads=" << n_heads << ", head_dim=" << head_dim 
                                          << ", d_proj=" << d_proj << ", out_dim=" << out_dim << std::flush;
                            }
                        }
                    }
                    
                    if (!found) {
                        // Fallback: try to infer from out_dim only
                        for (int try_n_heads = 1; try_n_heads <= 32 && !found; try_n_heads++) {
                            if (out_dim % try_n_heads == 0) {
                                int try_head_dim = out_dim / try_n_heads;
                                n_heads = try_n_heads;
                                head_dim = try_head_dim;
                                found = true;
                                std::cout << "\n      [Inferred from out_dim] n_heads=" << n_heads 
                                          << ", head_dim=" << head_dim << ", out_dim=" << out_dim << std::flush;
                            }
                        }
                    }
                    
                    if (!found) {
                        // Final fallback: assume n_heads = 1
                        n_heads = 1;
                        head_dim = out_dim;
                        std::cout << "\n      [Warning] Could not infer exact n_heads/head_dim, using n_heads=1, head_dim=" << head_dim << std::flush;
                    }
                    
                    qkv_size = qkv_weights.size();
                    out_size = out_weights.size();
                    
                    // Validate sizes
                    if (qkv_size != static_cast<size_t>(d_proj * ACTUAL_D_MODEL)) {
                        throw std::runtime_error("Attention qkv size mismatch: expected " + 
                                               std::to_string(d_proj * ACTUAL_D_MODEL) + 
                                               ", got " + std::to_string(qkv_size));
                    }
                    // Note: out_size validation is skipped since MLX uses different shape
                    
                    std::cout << " ✓" << std::endl;
                } else {
                    // Generate random weights using config defaults
                    d_proj = (ATTENTION_N_HEADS + 2 * ATTENTION_N_HEADS) * ATTENTION_HEAD_DIM;
                    qkv_size = d_proj * ACTUAL_D_MODEL;
                    out_size = ACTUAL_D_MODEL * ATTENTION_N_HEADS * ATTENTION_HEAD_DIM;
                    
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
                throw std::runtime_error("Out of memory during attention weight allocation");
            }
            
            // Create AttentionLayer with inferred/config dimensions
            AttentionLayer* layer = new AttentionLayer(
                &ctx_mgr, ACTUAL_D_MODEL, n_heads, n_heads, head_dim, 4096, true
            );
            std::cout << "    [createAttentionLayer] Created AttentionLayer with n_heads=" << n_heads 
                      << ", head_dim=" << head_dim << std::endl;
            
            // Verify out_weights size matches expected: d_model * n_heads * head_dim
            size_t expected_out_size = static_cast<size_t>(ACTUAL_D_MODEL * n_heads * head_dim);
            if (out_weights.size() != expected_out_size) {
                std::cerr << "\n    [Warning] out_weights size (" << out_weights.size() 
                          << ") != expected (" << expected_out_size << ")" << std::endl;
                std::cerr << "    Expected: d_model * n_heads * head_dim = " << ACTUAL_D_MODEL 
                          << " * " << n_heads << " * " << head_dim << " = " << expected_out_size << std::endl;
                // This might still work if the shapes are compatible
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
        
        // Helper function to load norm weights if available
        auto loadNormWeights = [&](ResidualBlock* block, int layer_idx, bool is_ssd) {
            if (use_pretrained_weights && block) {
                char layer_dir[256];
                if (is_ssd) {
                    snprintf(layer_dir, sizeof(layer_dir), "%s/layer_%03d_ssd", weights_dir.c_str(), layer_idx);
                } else {
                    snprintf(layer_dir, sizeof(layer_dir), "%s/layer_%03d", weights_dir.c_str(), layer_idx);
                }
                std::string norm_file = std::string(layer_dir) + "/norm_weight.bin";
                std::ifstream test_file(norm_file);
                if (test_file.good()) {
                    test_file.close();
                    std::vector<float> norm_weights = loadWeights(norm_file);
                    validateWeightSize(norm_weights, ACTUAL_D_MODEL, "norm", "layer " + std::to_string(layer_idx));
                    block->setNormWeights(norm_weights);
                }
            }
        };
        
        // Create all layers (for mamba2-130m: all 24 layers are SSD)
        int layer_count = 0;
        
        // For mamba2-130m: all layers are SSD (no SwiGLU or Attention)
        for (int i = 0; i < N_LAYER; ++i) {
            SSDLayer* layer = createSSDLayer(layer_count, SSD_EXPAND, SSD_KERNEL_SIZE, SSD_D_STATE, SSD_D_HEAD, SSD_N_GROUPS);
            ResidualBlock* block = new ResidualBlock(&ctx_mgr, layer, ACTUAL_D_MODEL, "pre", true);
            loadNormWeights(block, layer_count, true);
            seq_model.addLayer(std::unique_ptr<ResidualBlock>(block));
            layer_count++;
        }
        
        std::cout << "Initialized " << layer_count << "-layer model" << std::endl;
        std::cout.flush();
        
        // LM Head
        std::cout << "Initializing LM Head..." << std::flush;
        LMHead lm_head(&ctx_mgr, ACTUAL_D_MODEL, ACTUAL_VOCAB_SIZE);
        std::cout << " ✓ (created)" << std::endl;
        
        std::vector<float> lm_weights;
        if (use_pretrained_weights) {
            std::cout << "  Loading LM Head weights..." << std::flush;
            lm_weights = loadWeights(weights_dir + "/lm_head_weight.bin");
            validateWeightSize(lm_weights, ACTUAL_VOCAB_SIZE * ACTUAL_D_MODEL, 
                             "LM head", "vocab=" + std::to_string(ACTUAL_VOCAB_SIZE) + 
                             ", d_model=" + std::to_string(ACTUAL_D_MODEL));
        } else {
            std::cout << "  Generating LM Head weights..." << std::flush;
            lm_weights.resize(ACTUAL_VOCAB_SIZE * ACTUAL_D_MODEL);
            for (float& w : lm_weights) {
                w = 0.01f * (std::rand() % 200 - 100) / 100.0f;
            }
        }
        std::cout << " ✓ (" << lm_weights.size() << " weights)" << std::endl;
        
        std::cout << "  Initializing LM Head..." << std::flush;
        lm_head.initializeWeights(lm_weights);
        std::cout << " ✓" << std::endl;
        std::cout << "✓ LM Head initialized (" << (ACTUAL_VOCAB_SIZE * ACTUAL_D_MODEL * sizeof(float) / 1024 / 1024) << " MB)" << std::endl;
        
        auto weight_load_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> weight_load_elapsed = weight_load_end - weight_load_start;
        
        // Sampler
        Sampler sampler;
        std::cout << "✓ Sampler initialized" << std::endl;
        
        std::cout << "✓ Model initialization complete" << std::endl;
        std::cout << "  Weight loading time: " << std::fixed << std::setprecision(3) 
                  << weight_load_elapsed.count() << " seconds" << std::endl;
        std::cout << std::endl;
        
        // Create token buffer
        cl_mem token_buffer = createTokenBuffer(context, prompt_tokens);
        int batch_size = 1;
        int seq_len = prompt_tokens.size();
        
        // Prefill: Process prompt tokens
        std::cout << "Running prefill on " << seq_len << " tokens..." << std::endl;
        auto prefill_start = std::chrono::high_resolution_clock::now();
        
        cl_mem embeddings = embedding.encode(token_buffer, batch_size, seq_len, queue);
        clFinish(queue);  // Ensure embedding completion for determinism
        
        
    // Forward through sequence model
    std::vector<LayerState> states;  // Will be populated by stateful layers
    cl_mem hidden = seq_model.forward(embeddings, batch_size, seq_len, &states, queue);
    clFinish(queue);  // Ensure forward pass completion for determinism
    
    
    // Get logits from last token and sample first generation token
    // Match MLX: extract last token first, then apply LM head
    // hidden is [batch_size, seq_len, d_model] = [1, seq_len, d_model]
    // Extract last token: [batch_size, d_model] = [1, d_model]
    size_t last_token_hidden_offset = (seq_len - 1) * ACTUAL_D_MODEL;
    std::vector<float> last_token_hidden(ACTUAL_D_MODEL);
    cl_int err = clEnqueueReadBuffer(queue, hidden, CL_TRUE, 
                                     last_token_hidden_offset * sizeof(float),
                                     ACTUAL_D_MODEL * sizeof(float),
                                     last_token_hidden.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read last token hidden state");
    }
    
    // Create a buffer for the last token hidden state
    cl_mem last_token_hidden_buf = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                                   ACTUAL_D_MODEL * sizeof(float),
                                                   last_token_hidden.data(), &err);
    if (err != CL_SUCCESS || !last_token_hidden_buf) {
        throw std::runtime_error("Failed to create last token hidden state buffer");
    }
    
    // Apply LM head to last token only (batch_size=1)
    cl_mem prefill_logits = lm_head.forward(last_token_hidden_buf, batch_size, queue);
    clFinish(queue);  // Ensure LM head completion for determinism
    
    // Read logits for the last token
    std::vector<float> last_token_logits(ACTUAL_VOCAB_SIZE);
    err = clEnqueueReadBuffer(queue, prefill_logits, CL_TRUE, 0,
                              ACTUAL_VOCAB_SIZE * sizeof(float),
                              last_token_logits.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to read prefill logits from device");
    }
    
    // Release the temporary buffer
    clReleaseMemObject(last_token_hidden_buf);
    
    // Sample first token from prefill logits
    int current_token_id = sampler.topPSample(last_token_logits, DEFAULT_TOP_P, DEFAULT_TEMPERATURE);
    
    auto prefill_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> prefill_elapsed = prefill_end - prefill_start;
    double prefill_tokens_per_sec = seq_len / prefill_elapsed.count();
    
    // Release prefill logits buffer
    clReleaseMemObject(prefill_logits);
    
    // Generate tokens (matching MLX: first token + (max_tokens - 1) more)
    std::vector<int32_t> generated_tokens;
    generated_tokens.push_back(current_token_id);  // Include prefill token to match MLX
    
    // Print first token (decoded)
    if (tokenizer_loaded) {
        std::string decoded_token = tokenizer.decode({current_token_id});
        std::cout << decoded_token << std::flush;
    }
    
    // Start timing for generation phase
    auto generation_start = std::chrono::high_resolution_clock::now();
    int tokens_generated = 0;
        
        for (int i = 0; i < max_tokens - 1; ++i) {
            cl_mem current_token_buf = nullptr;
            cl_mem current_embedding = nullptr;
            cl_mem next_hidden = nullptr;
            cl_mem logits = nullptr;
            try {
                // Encode current token
                std::vector<int32_t> current_token_vec = {current_token_id};
                current_token_buf = createTokenBuffer(context, current_token_vec);
                current_embedding = embedding.encodeStep(current_token_buf, batch_size, queue);
                if (!current_embedding) throw std::runtime_error("encodeStep returned null buffer");
                
                // Step through sequence model
                next_hidden = seq_model.step(current_embedding, batch_size, &states, queue);
                if (!next_hidden) throw std::runtime_error("seq_model.step returned null buffer");
                
                // Get logits from LM head
                logits = lm_head.forward(next_hidden, batch_size, queue);
                if (!logits) throw std::runtime_error("LMHead.forward returned null buffer");
                
                // Ensure all writes are visible before CPU read in sampler
                clFinish(queue);
                
                // Sample next token
                int next_token = sampler.sampleFromBuffer(
                    logits, ACTUAL_VOCAB_SIZE, queue,
                    DEFAULT_TOP_P, DEFAULT_TEMPERATURE
                );
                
                // Clamp token ID to valid range
                if (next_token >= ACTUAL_VOCAB_SIZE) {
                    next_token = next_token % ACTUAL_VOCAB_SIZE;
                }
                
                generated_tokens.push_back(next_token);
                current_token_id = next_token;
                tokens_generated++;
                
                // Print decoded token
                if (tokenizer_loaded) {
                    std::string decoded_token = tokenizer.decode({next_token});
                    std::cout << decoded_token << std::flush;
                }
                
                // Check for EOS
                if (next_token == EOS_TOKEN_ID) {
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
        
        auto generation_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> generation_elapsed = generation_end - generation_start;
        double generation_tokens_per_sec = (tokens_generated > 0) ? (tokens_generated / generation_elapsed.count()) : 0.0;
        
        std::cout << std::endl;
        std::cout << std::endl;
        
        // Display performance summary
        std::cout << "--------------------------------------------------" << std::endl;
        std::cout << "Weight loading time: " 
                  << std::fixed << std::setprecision(3) << weight_load_elapsed.count() 
                  << " seconds" << std::endl;
        std::cout << "Time to first token: " 
                  << std::fixed << std::setprecision(3) << prefill_elapsed.count() 
                  << " seconds" << std::endl;
        std::cout << "Prompt: " << seq_len << " tokens, " 
                  << std::fixed << std::setprecision(2) << prefill_tokens_per_sec 
                  << " tokens-per-sec" << std::endl;
        std::cout << "Generation: " << tokens_generated << " tokens, " 
                  << std::fixed << std::setprecision(2) << generation_tokens_per_sec 
                  << " tokens-per-sec" << std::endl;
        
        // Cleanup - protect against double-release and invalid buffers
        // NOTE: embeddings and hidden are owned by their respective layers (EmbeddingLayer and SequenceModel)
        // They will be released when the layers are destroyed, so we should NOT release them here.
        // Only release token_buffer which we created directly.
        
        // Helper function to safely release a buffer (avoiding clGetMemObjectInfo which can crash)
        auto safeRelease = [](cl_mem buf, const char* name) -> bool {
            if (!buf) {
                return true;
            }
            try {
                cl_int release_err = clReleaseMemObject(buf);
                if (release_err == CL_SUCCESS || release_err == CL_INVALID_MEM_OBJECT) {
                    return true;
                } else {
                    return false;
                }
            } catch (...) {
                return false;
            }
        };
        
        // Only release token_buffer - embeddings and hidden are owned by their layers
        safeRelease(token_buffer, "token_buffer");
        token_buffer = nullptr;
        // embeddings and hidden will be cleaned up by their layer destructors
        embeddings = nullptr;  // Just set to null, don't release
        hidden = nullptr;       // Just set to null, don't release
        DEBUG_TOKENS({
            std::cout << "Generated " << generated_tokens.size() << " tokens: [";
            for (size_t i = 0; i < generated_tokens.size(); ++i) {
                std::cout << generated_tokens[i];
                if (i < generated_tokens.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        });
        
        // Write output (even if partially generated)
        if (!generated_tokens.empty()) {
            writeTokenFile(output_file, generated_tokens);
        }
        
        // Cleanup OpenCL
        ctx_mgr.cleanup();
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
