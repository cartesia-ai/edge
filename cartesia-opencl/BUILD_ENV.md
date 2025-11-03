# Build Environment Variables

This document describes all environment variables that can be used to customize the Android build process for Cartesia OpenCL.

## Required Variables

### `ANDROID_NDK`
**Description**: Path to the Android NDK installation.  
**Default**: Auto-detected from common locations:
- macOS: `$HOME/Library/Android/sdk/ndk/` (latest version)
- Linux: `$HOME/Android/Sdk/ndk/` (latest version)

**Example**:
```bash
export ANDROID_NDK=$HOME/Library/Android/sdk/ndk/21.1.6352462
```

## Optional Variables

### Build Configuration

#### `BUILD_DIR`
**Description**: Directory where the build will be performed.  
**Default**: `<script_dir>/build_android_standalone`

**Example**:
```bash
export BUILD_DIR=/path/to/custom/build
```

#### `ANDROID_API`
**Description**: Android API level to target.  
**Default**: `21` (for maximum compatibility)

**Example**:
```bash
export ANDROID_API=28
```

#### `ANDROID_ABI`
**Description**: Android ABI to build for.  
**Default**: `arm64-v8a`  
**Options**: `arm64-v8a`, `armeabi-v7a`, `x86_64`, `x86`

**Example**:
```bash
export ANDROID_ABI=arm64-v8a
```

#### `ANDROID_ARCH`
**Description**: Android target architecture for the compiler.  
**Default**: `aarch64`  
**Options**: `aarch64`, `armv7a`, `x86_64`, `i686`

**Example**:
```bash
export ANDROID_ARCH=aarch64
```

#### `CMAKE_BUILD_TYPE`
**Description**: CMake build type.  
**Default**: `Release`  
**Options**: `Release`, `Debug`, `RelWithDebInfo`, `MinSizeRel`

**Example**:
```bash
export CMAKE_BUILD_TYPE=Debug
```

#### `ANDROID_STL`
**Description**: Android C++ standard library to use.  
**Default**: `c++_shared`  
**Options**: `c++_shared`, `c++_static`, `none`

**Example**:
```bash
export ANDROID_STL=c++_shared
```

#### `CMAKE_BUILD_PARALLEL_LEVEL`
**Description**: Number of parallel build jobs.  
**Default**: Auto-detected (number of CPU cores)

**Example**:
```bash
export CMAKE_BUILD_PARALLEL_LEVEL=8
```

### Host Configuration

#### `HOST_ARCH`
**Description**: Host machine architecture (for NDK toolchain selection).  
**Default**: Auto-detected from system  
**Format**: `<os>-<arch>` (e.g., `darwin-arm64`, `darwin-x86_64`, `linux-x86_64`)

**Example**:
```bash
export HOST_ARCH=darwin-arm64
```

### Temporary Files

#### `TEMP_DIR`
**Description**: Directory for temporary files during build.  
**Default**: `/tmp`

**Example**:
```bash
export TEMP_DIR=/var/tmp
```

#### `OPENCL_STUB_LIB`
**Description**: Path to OpenCL stub library for linking.  
**Default**: `$TEMP_DIR/libOpenCL_stub.so`

**Example**:
```bash
export OPENCL_STUB_LIB=/path/to/libOpenCL_stub.so
```

### Homebrew Configuration

#### `HOMEBREW_PREFIX`
**Description**: Homebrew installation prefix.  
**Default**: Auto-detected (`/opt/homebrew` or `/usr/local`)

**Example**:
```bash
export HOMEBREW_PREFIX=/opt/homebrew
```

## Complete Example

```bash
# Minimal setup (using defaults)
export ANDROID_NDK=$HOME/Library/Android/sdk/ndk/21.1.6352462
./build_android.sh

# Full custom configuration
export ANDROID_NDK=$HOME/Library/Android/sdk/ndk/25.2.9519653
export BUILD_DIR=$HOME/builds/cartesia-opencl
export ANDROID_API=28
export ANDROID_ABI=arm64-v8a
export ANDROID_ARCH=aarch64
export CMAKE_BUILD_TYPE=RelWithDebInfo
export ANDROID_STL=c++_shared
export CMAKE_BUILD_PARALLEL_LEVEL=8
export TEMP_DIR=/var/tmp
export HOMEBREW_PREFIX=/opt/homebrew
./build_android.sh
```

## Platform-Specific Notes

### macOS (Apple Silicon)
```bash
# NDK usually auto-detects as darwin-arm64
export ANDROID_NDK=$HOME/Library/Android/sdk/ndk/21.1.6352462
export HOST_ARCH=darwin-arm64
./build_android.sh
```

### macOS (Intel)
```bash
# NDK usually auto-detects as darwin-x86_64
export ANDROID_NDK=$HOME/Library/Android/sdk/ndk/21.1.6352462
export HOST_ARCH=darwin-x86_64
./build_android.sh
```

### Linux
```bash
# NDK usually auto-detects as linux-x86_64
export ANDROID_NDK=$HOME/Android/Sdk/ndk/21.1.6352462
export HOST_ARCH=linux-x86_64
./build_android.sh
```

## Troubleshooting

### NDK Not Found
If the script cannot find your NDK, explicitly set `ANDROID_NDK`:
```bash
export ANDROID_NDK=/path/to/your/ndk
```

### Wrong Host Architecture
If the NDK toolchain is not found, check your host architecture:
```bash
uname -m  # Should output: arm64, x86_64, etc.
export HOST_ARCH=darwin-$(uname -m)  # or linux-$(uname -m)
```

### Build Fails on Rosetta 2 (macOS)
If running on Apple Silicon via Rosetta 2, you may need:
```bash
export HOST_ARCH=darwin-x86_64
```

### OpenCL Stub Creation Fails
If the OpenCL stub library fails to create, specify a different temporary directory:
```bash
export TEMP_DIR=$HOME/tmp
mkdir -p $TEMP_DIR
```

## Environment File

You can create a `.env` file for your project:

```bash
# .env.example
ANDROID_NDK=$HOME/Library/Android/sdk/ndk/21.1.6352462
ANDROID_API=21
ANDROID_ABI=arm64-v8a
CMAKE_BUILD_TYPE=Release
CMAKE_BUILD_PARALLEL_LEVEL=8
```

Then source it before building:
```bash
source .env
./build_android.sh
```

