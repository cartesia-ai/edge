#pragma once

namespace cartesia_opencl {

// Rene model configuration
// Based on cartesia-ai/Rene-v0.1-1.3b model architecture

// Model dimensions
constexpr int D_MODEL = 2048;           // Hidden dimension
constexpr int VOCAB_SIZE = 50288;       // Vocabulary size (padded to multiple of 16)
constexpr int N_LAYER = 48;              // Total number of layers
constexpr int N_LAYER_REPEATS = 4;      // Number of times to repeat unique layers
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

// Attention layer defaults
constexpr int ATTENTION_N_HEADS = 16;   // Number of attention heads
constexpr int ATTENTION_HEAD_DIM = 128; // Head dimension (d_model / n_heads)

// RMS Norm epsilon
constexpr float RMS_NORM_EPS = 1e-6f;

// Sampling defaults
constexpr float DEFAULT_TEMPERATURE = 0.85f;
constexpr float DEFAULT_TOP_P = 0.99f;

} // namespace cartesia_opencl

