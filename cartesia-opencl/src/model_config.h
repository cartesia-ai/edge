#pragma once

namespace cartesia_opencl {

// Test model configuration (Mamba2-130M size)
// Optimized for Android devices with limited memory
// To use full Rene (d_model=2048), change back and rebuild

// Model dimensions - REDUCED FOR TESTING
constexpr int D_MODEL = 768;            // Was 2048 (Mamba2-130M: fits in device memory)
constexpr int VOCAB_SIZE = 50288;       // Vocabulary size (padded to multiple of 16)
constexpr int N_LAYER = 24;             // Was 48 (Mamba2-130M)
constexpr int N_LAYER_REPEATS = 1;      // Was 4 (for testing)
constexpr int N_UNIQUE_LAYERS = 12;     // Number of unique layer configurations

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

// Attention layer defaults - adjusted for d_model=768
constexpr int ATTENTION_N_HEADS = 12;   // Number of attention heads (768/64=12)
constexpr int ATTENTION_HEAD_DIM = 64;  // Head dimension (was 128)

// RMS Norm epsilon
constexpr float RMS_NORM_EPS = 1e-6f;

// Sampling defaults
constexpr float DEFAULT_TEMPERATURE = 0.85f;
constexpr float DEFAULT_TOP_P = 0.99f;

} // namespace cartesia_opencl

