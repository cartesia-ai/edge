# Cartesia OpenCL Backend

This document outlines the architecture and implementation details for adding OpenCL support to the Edge library for state-space models (SSMs).

## Overview

The Edge library currently supports three backends:
- **PyTorch** (`cartesia-pytorch/`) - General-purpose backend using PyTorch's tensor operations
- **MLX** (`cartesia-mlx/`) - Apple's MLX framework with Python-based operations  
- **Metal** (`cartesia-metal/`) - Apple's Metal GPU framework with custom C++/Metal kernels

Adding OpenCL support would provide cross-platform GPU acceleration for state-space models on a wide range of hardware.

## Architecture

### Directory Structure (current)
```
cartesia-opencl/
├── CMakeLists.txt                 # Build configuration for the OpenCL test binary
├── build_android.sh               # Build script for Android (OpenCL test app)
├── build_android_opencl_info.sh   # Builds a small OpenCL capability dumper for Android
├── tools/
│   └── decode_tokens.py           # Decodes generated token IDs to text
├── opencl-headers/                # Khronos OpenCL headers (downloaded by scripts if missing)
└── src/
    ├── opencl_context.{h,cpp}     # OpenCL context/queue/program helpers
    ├── layers/                    # Model layers implemented with OpenCL (+ CPU fallbacks)
    │   ├── attention_layer.{h,cpp}
    │   ├── linear_layer.{h,cpp}
    │   ├── ssd_layer.{h,cpp}
    │   ├── swiglu_layer.{h,cpp}
    │   └── rms_norm_layer.{h,cpp}
    ├── embedding.cpp              # Token embedding (CPU fallback + OpenCL buffers)
    ├── lm_head.cpp                # Output projection head (CPU fallback + OpenCL buffers)
    └── layers/*.cl (embedded)     # Kernel sources are embedded as raw strings in *.cpp
```

## Key Operations

The core operations that benefit from GPU acceleration are:

### 1. SSM Update
State-space model updates with matrix operations:
- Input processing with SILU activation
- State transition with exponential decay
- Matrix-vector operations (A, B, C, D matrices)
- Output gating with z parameter

### 2. SSD Update  
State-space dynamics updates:
- Similar to SSM but with different parameter structure
- Decay-based state transitions
- Grouped parameter handling

### 3. Conv1D Operations
1D convolutions with and without Swish activation:
- Forward pass for full sequence processing
- Update operations for streaming inference
- State management for causal convolutions

## Implementation Details

### OpenCL Kernels

The OpenCL kernels follow the same mathematical operations as the Metal implementations but use OpenCL syntax:

```opencl
__kernel void ssm_update_kernel(
    __global const float* x,
    __global const float* dt,
    __global const float* A,
    __global const float* B,
    __global const float* C,
    __global const float* D,
    __global const float* z,
    __global const float* state,
    __global float* y,
    __global float* next_state,
    const int state_size,
    const int channel_size
) {
    const int batch_idx = get_global_id(0);
    const int channel_idx = get_global_id(1);
    
    // ... kernel implementation
}
```

### C++ Wrapper

The C++ wrapper provides:
- OpenCL context management
- Buffer allocation and data transfer
- Kernel execution and synchronization
- Error handling and device selection

### Python Utilities

This repo includes helper Python scripts in `tools/` (e.g., token decoding). There are no Python package modules such as `interface.py` or `version.py` in this directory.

## Build System

### Dependencies
- **OpenCL SDK** - For GPU compute capabilities (provided by device at runtime)
- **CMake** - Build system configuration
- **Android NDK** - For Android cross-compilation

### Build Configuration
This project builds a standalone C++ OpenCL test binary via CMake. Android builds are driven by `build_android.sh`, which configures the NDK toolchain and links against OpenCL at runtime (kernels are embedded in the binary as strings).

## Integration with Existing Backends

## Usage

### 1) Generate input tokens (using Hugging Face transformers)
For now, tokens are generated on CPU using a standard tokenizer and saved as a raw int32 binary file that the C++ binary reads.

Example (Python):
```python
from transformers import AutoTokenizer
import numpy as np

text = "Rene Descartes was"
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMo-1B-hf")
ids = tokenizer.encode(text, add_special_tokens=False)

# Save as raw int32 little-endian binary (no header)
np.asarray(ids, dtype=np.int32).tofile("prompt_tokens.bin")
print(f"Wrote {len(ids)} tokens to prompt_tokens.bin: {ids}")
```

This produces `prompt_tokens.bin` containing the prompt token IDs as 32-bit integers.

### 2) Build for Android
```bash
./build_android.sh
```

### 3) Run on the Android device
Push the binary and token file to the device and execute the test driver. Below is a minimal set of commands (replace paths as needed):

```bash
# Paths
BIN=cartesia_opencl_test
LOCAL_BUILD=./build_android_standalone/${BIN}
DEVICE_BIN=/data/local/tmp/${BIN}
LOCAL_PROMPT=./prompt_tokens.bin
DEVICE_PROMPT=/data/local/tmp/prompt_tokens.bin
DEVICE_OUTPUT=/data/local/tmp/output_tokens.bin

# Copy
adb push "${LOCAL_BUILD}" "${DEVICE_BIN}"
adb push "${LOCAL_PROMPT}" "${DEVICE_PROMPT}"
adb shell chmod +x "${DEVICE_BIN}"

# Optional: skip attention kernel build and use CPU fallback to avoid driver hangs
# (recommended on some devices)
adb shell "export SKIP_ATTENTION_KERNELS=1 && ${DEVICE_BIN} ${DEVICE_PROMPT} ${DEVICE_OUTPUT} 10 1"

# Arguments to the binary are:
#   1) prompt_tokens_path (required)
#   2) output_tokens_path  (optional; default: output_tokens.bin)
#   3) max_tokens          (optional; default: 50)
#   4) n_layer_repeats     (optional; default from model config; try 1 for reduced memory)

# Pull the output back
adb pull "${DEVICE_OUTPUT}" ./output_tokens.bin
```

### 4) Decode generated tokens back to text
Use the provided tool to decode the generated token IDs:
```bash
python3 tools/decode_tokens.py ./output_tokens.bin --tokenizer allenai/OLMo-1B-hf --verbose
```

Notes:
- On memory-constrained devices, start with `n_layer_repeats=1` to build a 12-layer model (instead of 48).
- If attention kernel compilation hangs or crashes on the device, use `SKIP_ATTENTION_KERNELS=1` to activate the CPU fallback for attention while keeping other layers on OpenCL.

### Running on Android
- Build: `./build_android.sh`
- Push and run: Use `adb` to copy the binary to device and execute
- Outputs: `output_tokens.bin` (can be decoded via `tools/decode_tokens.py`)

### Performance Comparison
Expected performance characteristics:
- **GPU Acceleration**: 10-100x speedup over CPU for large batches
- **Memory Bandwidth**: Optimized for high-throughput operations
- **Latency**: Lower latency than PyTorch for custom operations

## Comparison with Existing Backends

| Feature | PyTorch | MLX | Metal | OpenCL |
|---------|---------|-----|-------|--------|
| Platform Support | Cross-platform | Apple only | Apple only | Cross-platform |
| GPU Acceleration | CUDA/ROCm | Metal | Metal | OpenCL |
| Custom Kernels | Limited | Python ops | Metal kernels | OpenCL kernels |
| Performance | Good | Excellent | Excellent | Good-Excellent |
| Development Effort | Low | Medium | High | High |

## Conclusion

Adding OpenCL support to the Edge library would provide:

1. **Cross-platform GPU acceleration** for state-space models
2. **Performance improvements** on a wide range of hardware
3. **Custom kernel optimization** for specific operations
4. **Future-proof architecture** for emerging hardware

The implementation would follow the existing patterns established by the Metal backend while providing the flexibility and portability of OpenCL.


