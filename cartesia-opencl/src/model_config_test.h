#pragma once

namespace cartesia_opencl {

// Test model configuration - optimized for Android devices
// Based on Mamba2-130M (matches test data from MLX)

// Model dimensions - SMALLER for testing on mobile
constexpr int D_MODEL = 768;            // Reduced from 2048 (Mamba2-130M size)
constexpr int VOCAB_SIZE = 50288;       // Keep same as test data
constexpr int N_LAYER = 24;             // Reduced from 48 (Mamba2-130M)
constexpr int N_LAYER_REPEATS = 1;      // Reduced from 4 for testing
constexpr int N_UNIQUE_LAYERS = 12;     // Same

// Layer type indices (0-based, for the 12 unique layers)
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
constexpr int EOS_TOKEN_ID = 50279;
constexpr int PAD_TOKEN_ID = 1;

// SSD layer defaults (can be overridden per layer)
constexpr int SSD_EXPAND = 2;
constexpr int SSD_KERNEL_SIZE = 4;
constexpr int SSD_D_STATE = 64;
constexpr int SSD_D_HEAD = 64;
constexpr int SSD_N_GROUPS = 1;

// Attention layer defaults - adjusted for d_model=768
constexpr int ATTENTION_N_HEADS = 12;   // Reduced from 16 (768/64=12)
constexpr int ATTENTION_HEAD_DIM = 64;  // Keep same

// RMS Norm epsilon
constexpr float RMS_NORM_EPS = 1e-6f;

// Sampling defaults
constexpr float DEFAULT_TEMPERATURE = 0.85f;
constexpr float DEFAULT_TOP_P = 0.99f;

// Memory calculation helper
// Embedding size: VOCAB_SIZE * D_MODEL * 4 bytes
// 50288 * 768 * 4 = 154,684,416 bytes = ~147 MB ✓ Fits in 358 MB!
// vs Rene: 50288 * 2048 * 4 = 411,959,296 bytes = ~392 MB ✗ Too big!

} // namespace cartesia_opencl

