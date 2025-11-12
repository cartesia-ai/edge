# Cartesia OpenCL Backend

OpenCL backend for cross-platform GPU acceleration of state-space models (SSMs).

## Overview

C++ implementation with OpenCL kernels for efficient SSM inference on GPUs. Provides cross-platform GPU acceleration for a wide range of hardware.

## Architecture

```
cartesia-opencl/
├── CMakeLists.txt                 # Build configuration
├── build_android.sh               # Android build script
├── src/
│   ├── opencl_context.{h,cpp}     # OpenCL context management
│   ├── layers/                    # Model layers with OpenCL kernels
│   ├── embedding.{h,cpp}         # Token embedding
│   ├── lm_head.{h,cpp}           # Output projection
│   └── *.cl                       # OpenCL kernel sources
└── tools/                         # Utility scripts
```

## Key Operations

- **SSM Update**: State-space model updates with matrix operations
- **SSD Update**: State-space dynamics updates
- **Conv1D**: 1D convolutions with Swish activation

## Build

### Dependencies
- OpenCL headers and library (Will be downloaded by script if not found)
- CMake
- Android NDK (for Android builds)

### Build for Android
```bash
./build_android.sh
```

### OpenCL Device Info (Diagnostic)
```bash
./build_android_opencl_info.sh  # Builds diagnostic tool to query OpenCL capabilities
adb push ./build_android/dump_opencl_info /data/local/tmp/
adb shell /data/local/tmp/dump_opencl_info
```

## Usage

### Build and Run
```bash
# Build
./build_android.sh

# Run with copy_and_run.sh helper script
./copy_and_run.sh "Rene Descartes was" mamba2_130m_weights/ output.bin 10

# Or manually:
adb push ./build_android_standalone/cartesia_opencl_test /data/local/tmp/
adb shell '/data/local/tmp/cartesia_opencl_test "Rene Descartes was" \
  /data/local/tmp/mamba2_130m_weights \
  /data/local/tmp/output_opencl_mamba2.bin 10'
```

Binary arguments:
1. `input` - Text prompt or path to binary token file (required)
2. `weights_dir` - Path to model weights directory (required)
3. `output_path` - Output token file path (required)
4. `max_tokens` - Maximum tokens to generate (optional, default: 10)

