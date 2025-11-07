#pragma once

namespace cartesia_opencl {

// Mamba2-130M model configuration (for testing)
// This is the smallest model for easier debugging and comparison
// Architecture: Pure SSD (all 24 layers are SSD, no SwiGLU or Attention)

// Model dimensions - Mamba2-130M
constexpr int D_MODEL = 768;            // Mamba2-130M size
constexpr int VOCAB_SIZE = 50288;       // Vocabulary size
constexpr int N_LAYER = 24;             // Mamba2-130M has 24 layers (all SSD)
constexpr int N_LAYER_REPEATS = 1;      // 1 repeat (24 SSD layers)
constexpr int N_UNIQUE_LAYERS = 1;      // Only 1 unique layer type (SSD)

// Layer type indices (0-based, for the 12 unique layers)
// These correspond to the layer sequence in LM.base_cfg unique_layers
constexpr int LAYER_TYPE_SSD = 0;
constexpr int LAYER_TYPE_SSD_2 = 1;
constexpr int LAYER_TYPE_SWIGLU = 2;
constexpr int LAYER_TYPE_SSD_3 = 3;
constexpr int LAYER_TYPE_SSD_4 = 4;
constexpr int LAYER_TYPE_SWIGLU_2 = 5;
constexpr int LAYER_TYPE_ATTENTION = 6;
constexpr int LAYER_TYPE_SSD_5 = 7;
constexpr int LAYER_TYPE_SWIGLU_3 = 8;
constexpr int LAYER_TYPE_SSD_6 = 9;
constexpr int LAYER_TYPE_SSD_7 = 10;
constexpr int LAYER_TYPE_SWIGLU_4 = 11;

// Special tokens
constexpr int EOS_TOKEN_ID = 50279;     // End-of-sequence token ID
constexpr int PAD_TOKEN_ID = 1;         // Padding token ID

// SSD layer defaults (can be overridden per layer)
constexpr int SSD_EXPAND = 2;           // Expansion factor
constexpr int SSD_KERNEL_SIZE = 4;      // Convolution kernel size
constexpr int SSD_D_STATE = 64;         // State dimension
constexpr int SSD_D_HEAD = 64;          // Head dimension
constexpr int SSD_N_GROUPS = 1;         // Number of groups

// Attention layer defaults - Reduced for device memory
constexpr int ATTENTION_N_HEADS = 8;    // Reduced from 16 (1024/128=8)
constexpr int ATTENTION_HEAD_DIM = 128;  // Head dimension

// RMS Norm epsilon (must match MLX default)
constexpr float RMS_NORM_EPS = 1e-5f;

// Sampling defaults
constexpr float DEFAULT_TEMPERATURE = 0.85f;
constexpr float DEFAULT_TOP_P = 0.99f;

} // namespace cartesia_opencl

