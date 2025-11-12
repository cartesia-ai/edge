#!/bin/bash

# Build script for dump_opencl_info.cpp on Android
# Requires Android NDK with CMake support

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build_android"

# Hardcoded Android NDK path
NDK_PATH="/Users/alazarshenkute/Library/Android/sdk/ndk/19.2.5345600"

# Allow override via environment variable
if [ -n "$ANDROID_NDK" ]; then
    NDK_PATH="$ANDROID_NDK"
fi

if [ ! -d "$NDK_PATH" ]; then
    echo "Error: Android NDK not found at: $NDK_PATH"
    echo "Please set ANDROID_NDK environment variable to point to your NDK installation"
    exit 1
fi

if [ ! -f "$NDK_PATH/build/cmake/android.toolchain.cmake" ]; then
    echo "Error: android.toolchain.cmake not found in NDK at: $NDK_PATH"
    exit 1
fi

echo "Using NDK: $NDK_PATH"

# Check for OpenCL headers and download if needed
OPENCL_HEADERS_DIR="${SCRIPT_DIR}/opencl-headers"
if [ ! -d "${OPENCL_HEADERS_DIR}/OpenCL-Headers/CL" ]; then
    echo "OpenCL headers not found. Downloading..."
    mkdir -p "${OPENCL_HEADERS_DIR}"
    cd "${OPENCL_HEADERS_DIR}"
    
    # Download OpenCL headers from Khronos
    if command -v git &> /dev/null; then
        if [ -d "OpenCL-Headers" ]; then
            echo "Updating existing OpenCL headers..."
            cd OpenCL-Headers && git pull && cd ..
        else
            echo "Cloning OpenCL headers repository..."
            git clone --depth 1 https://github.com/KhronosGroup/OpenCL-Headers.git
        fi
    else
        echo "Error: git not found. Please install git or manually download OpenCL headers."
        echo "You can also install via Homebrew: brew install opencl-headers"
        exit 1
    fi
    cd "$SCRIPT_DIR"
fi

# Create build directory
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# Find OpenCL library path (adjust these paths based on your Android device)
# PowerVR devices typically have OpenCL in /vendor/lib64
OPENCL_LIB_PATH="${SCRIPT_DIR}/opencl_libs"

# Copy CMakeLists for dump program
cp "${SCRIPT_DIR}/CMakeLists_dump.txt" CMakeLists.txt
# Update source path in CMakeLists to point to parent directory
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS uses BSD sed
    sed -i '' 's|dump_opencl_info.cpp|../dump_opencl_info.cpp|' CMakeLists.txt
    # Add OpenCL headers path (absolute path) - replace the set() line
    sed -i '' "s|set(OPENCL_HEADERS_BASE_DIR .*|set(OPENCL_HEADERS_BASE_DIR \"${OPENCL_HEADERS_DIR}/OpenCL-Headers\")|" CMakeLists.txt
else
    # Linux uses GNU sed
    sed -i 's|dump_opencl_info.cpp|../dump_opencl_info.cpp|' CMakeLists.txt
    # Add OpenCL headers path (absolute path) - replace the set() line
    sed -i "s|set(OPENCL_HEADERS_BASE_DIR .*|set(OPENCL_HEADERS_BASE_DIR \"${OPENCL_HEADERS_DIR}/OpenCL-Headers\")|" CMakeLists.txt
fi

# Verify the replacement worked
echo "OpenCL headers directory: ${OPENCL_HEADERS_DIR}/OpenCL-Headers"
if [ ! -d "${OPENCL_HEADERS_DIR}/OpenCL-Headers/CL" ]; then
    echo "Warning: OpenCL headers directory structure may be incorrect"
fi

# Create OpenCL stub library for linking (symbols resolved at runtime)
OPENCL_STUB_LIB="/tmp/libOpenCL_stub.so"
if [ ! -f "$OPENCL_STUB_LIB" ]; then
    echo "Creating OpenCL stub library for linking..."
    cat > /tmp/opencl_stub.c << 'STUB_EOF'
#include <stddef.h>
#include <stdint.h>
// Stub with proper signatures - symbols resolved at runtime
typedef uint32_t cl_uint;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef int32_t cl_int;
typedef uint64_t cl_ulong;
typedef uint64_t cl_device_type;
#define CL_DEVICE_TYPE_GPU 0
#define CL_DEVICE_TYPE_CPU 0
cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) { return 0; }
cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices) { return 0; }
cl_context clCreateContext(const void* props, cl_uint num_devices, const cl_device_id* devices, void* pfn_notify, void* user_data, cl_int* errcode_ret) { return 0; }
cl_int clReleaseContext(cl_context context) { return 0; }
cl_int clGetPlatformInfo(cl_platform_id platform, uint32_t param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) { return 0; }
cl_int clGetDeviceInfo(cl_device_id device, uint32_t param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) { return 0; }
STUB_EOF
    
    # Find the Android clang compiler  
    ANDROID_CLANG="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/aarch64-linux-android22-clang"
    if [ ! -f "$ANDROID_CLANG" ]; then
        ANDROID_CLANG=$(find "$NDK_PATH/toolchains" -name "*aarch64*clang*" -type f | head -1)
    fi
    if [ ! -f "$ANDROID_CLANG" ]; then
        ANDROID_CLANG="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/bin/clang"
    fi
    
    if [ -f "$ANDROID_CLANG" ]; then
        # Use Android API 21 for maximum compatibility
        "$ANDROID_CLANG" \
            -target aarch64-linux-android21 \
            --sysroot="$NDK_PATH/toolchains/llvm/prebuilt/darwin-x86_64/sysroot" \
            -shared -fPIC \
            -Wl,-soname,libOpenCL.so \
            /tmp/opencl_stub.c -o "$OPENCL_STUB_LIB"
        if [ -f "$OPENCL_STUB_LIB" ]; then
            echo "Created OpenCL stub library: $OPENCL_STUB_LIB"
        else
            echo "Warning: Failed to create stub library"
        fi
    else
        echo "Warning: Could not find Android clang compiler"
    fi
    rm -f /tmp/opencl_stub.c
fi

# Check if Homebrew opencl-headers is installed and add to CMAKE_PREFIX_PATH if needed
if [ -d "/usr/local/opt/opencl-headers" ] || [ -d "/opt/homebrew/opt/opencl-headers" ]; then
    if [ -d "/usr/local/opt/opencl-headers" ]; then
        export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:+$CMAKE_PREFIX_PATH:}/usr/local/opt/opencl-headers"
        echo "Found Homebrew opencl-headers at /usr/local/opt/opencl-headers"
    elif [ -d "/opt/homebrew/opt/opencl-headers" ]; then
        export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:+$CMAKE_PREFIX_PATH:}/opt/homebrew/opt/opencl-headers"
        echo "Found Homebrew opencl-headers at /opt/homebrew/opt/opencl-headers"
    fi
fi

# Configure with Android NDK
# Adjust API level and architecture as needed
ANDROID_API=21
ARCH=arm64

cmake \
    -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-${ANDROID_API} \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_STL=c++_static \
    ${CMAKE_PREFIX_PATH:+-DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"} \
    -S .

# Build
cmake --build . --config Release

echo ""
echo "Build complete!"
echo ""
echo "To copy and run on your Android device:"
echo "  adb push $BUILD_DIR/dump_opencl_info /data/local/tmp/"
echo "  adb shell chmod +x /data/local/tmp/dump_opencl_info"
echo "  adb shell /data/local/tmp/dump_opencl_info"
echo ""
echo "Or run directly:"
echo "  adb push $BUILD_DIR/dump_opencl_info /data/local/tmp/ && adb shell 'chmod +x /data/local/tmp/dump_opencl_info && /data/local/tmp/dump_opencl_info'"
echo ""

