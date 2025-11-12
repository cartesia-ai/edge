#!/bin/bash

# Build script for Cartesia OpenCL standalone executable on Android
# Requires Android NDK with CMake support
# See BUILD_ENV.md for environment variable documentation

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${BUILD_DIR:-${SCRIPT_DIR}/build_android_standalone}"

# Detect host architecture (darwin-x86_64, darwin-arm64, linux-x86_64, etc.)
if [[ "$OSTYPE" == "darwin"* ]]; then
    HOST_ARCH="${HOST_ARCH:-darwin-$(uname -m)}"
elif [[ "$OSTYPE" == "linux"* ]]; then
    HOST_ARCH="${HOST_ARCH:-linux-$(uname -m)}"
else
    HOST_ARCH="${HOST_ARCH:-unknown}"
fi

# Android NDK path - try environment variable first, then common locations
if [ -z "$ANDROID_NDK" ]; then
    # Try common Android SDK locations
    if [ -d "$HOME/Library/Android/sdk/ndk" ]; then
        # macOS default, find latest NDK version
        ANDROID_NDK=$(find "$HOME/Library/Android/sdk/ndk" -maxdepth 1 -type d | sort -V | tail -1)
    elif [ -d "$HOME/Android/Sdk/ndk" ]; then
        # Linux default
        ANDROID_NDK=$(find "$HOME/Android/Sdk/ndk" -maxdepth 1 -type d | sort -V | tail -1)
    fi
fi

if [ -z "$ANDROID_NDK" ] || [ ! -d "$ANDROID_NDK" ]; then
    echo "Error: Android NDK not found"
    echo "Please set ANDROID_NDK environment variable to point to your NDK installation"
    echo "Example: export ANDROID_NDK=\$HOME/Library/Android/sdk/ndk/21.1.6352462"
    exit 1
fi

    NDK_PATH="$ANDROID_NDK"

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

# Copy CMakeLists for standalone program
cp "${SCRIPT_DIR}/CMakeLists_standalone.txt" CMakeLists.txt
# Update source paths in CMakeLists to point to parent directory
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS uses BSD sed
    sed -i '' 's|main_rene.cpp|../main_rene.cpp|' CMakeLists.txt
    sed -i '' 's|src/|../src/|g' CMakeLists.txt
    # Add OpenCL headers path (absolute path) - replace the set() line
    sed -i '' "s|set(OPENCL_HEADERS_BASE_DIR .*|set(OPENCL_HEADERS_BASE_DIR \"${OPENCL_HEADERS_DIR}/OpenCL-Headers\")|" CMakeLists.txt
else
    # Linux uses GNU sed
    sed -i 's|main_rene.cpp|../main_rene.cpp|' CMakeLists.txt
    sed -i 's|src/|../src/|g' CMakeLists.txt
    # Add OpenCL headers path (absolute path) - replace the set() line
    sed -i "s|set(OPENCL_HEADERS_BASE_DIR .*|set(OPENCL_HEADERS_BASE_DIR \"${OPENCL_HEADERS_DIR}/OpenCL-Headers\")|" CMakeLists.txt
fi

# Verify the replacement worked
echo "OpenCL headers directory: ${OPENCL_HEADERS_DIR}/OpenCL-Headers"
if [ ! -d "${OPENCL_HEADERS_DIR}/OpenCL-Headers/CL" ]; then
    echo "Warning: OpenCL headers directory structure may be incorrect"
fi

# Create OpenCL stub library for linking (symbols resolved at runtime)
TEMP_DIR="${TEMP_DIR:-/tmp}"
OPENCL_STUB_LIB="${OPENCL_STUB_LIB:-${TEMP_DIR}/libOpenCL_stub.so}"
OPENCL_STUB_C="${TEMP_DIR}/opencl_stub.c"

if [ ! -f "$OPENCL_STUB_LIB" ]; then
    echo "Creating OpenCL stub library for linking..."
    cat > "$OPENCL_STUB_C" << 'STUB_EOF'
#include <stddef.h>
#include <stdint.h>
// Stub with proper signatures - symbols resolved at runtime
typedef uint32_t cl_uint;
typedef void* cl_platform_id;
typedef void* cl_device_id;
typedef void* cl_context;
typedef void* cl_command_queue;
typedef void* cl_program;
typedef void* cl_kernel;
typedef void* cl_mem;
typedef int32_t cl_int;
typedef uint64_t cl_ulong;
typedef uint64_t cl_device_type;
#define CL_DEVICE_TYPE_GPU 0
#define CL_DEVICE_TYPE_CPU 0
#define CL_MEM_READ_ONLY 0
#define CL_MEM_WRITE_ONLY 0
#define CL_MEM_COPY_HOST_PTR 0
#define CL_SUCCESS 0
#define CL_PROGRAM_BUILD_LOG 0
cl_int clGetPlatformIDs(cl_uint num_entries, cl_platform_id* platforms, cl_uint* num_platforms) { return 0; }
cl_int clGetDeviceIDs(cl_platform_id platform, cl_device_type device_type, cl_uint num_entries, cl_device_id* devices, cl_uint* num_devices) { return 0; }
cl_context clCreateContext(const void* props, cl_uint num_devices, const cl_device_id* devices, void* pfn_notify, void* user_data, cl_int* errcode_ret) { return 0; }
cl_int clReleaseContext(cl_context context) { return 0; }
cl_command_queue clCreateCommandQueue(cl_context context, cl_device_id device, cl_ulong properties, cl_int* errcode_ret) { return 0; }
cl_int clReleaseCommandQueue(cl_command_queue command_queue) { return 0; }
cl_program clCreateProgramWithSource(cl_context context, cl_uint count, const char** strings, const size_t* lengths, cl_int* errcode_ret) { return 0; }
cl_int clBuildProgram(cl_program program, cl_uint num_devices, const cl_device_id* device_list, const char* options, void* pfn_notify, void* user_data) { return 0; }
cl_int clGetProgramBuildInfo(cl_program program, cl_device_id device, uint32_t param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) { return 0; }
cl_int clReleaseProgram(cl_program program) { return 0; }
cl_kernel clCreateKernel(cl_program program, const char* kernel_name, cl_int* errcode_ret) { return 0; }
cl_int clReleaseKernel(cl_kernel kernel) { return 0; }
cl_mem clCreateBuffer(cl_context context, cl_ulong flags, size_t size, void* host_ptr, cl_int* errcode_ret) { return 0; }
cl_int clReleaseMemObject(cl_mem memobj) { return 0; }
cl_int clSetKernelArg(cl_kernel kernel, cl_uint arg_index, size_t arg_size, const void* arg_value) { return 0; }
cl_int clEnqueueNDRangeKernel(cl_command_queue command_queue, cl_kernel kernel, cl_uint work_dim, const size_t* global_work_offset, const size_t* global_work_size, const size_t* local_work_size, cl_uint num_events_in_wait_list, const void* event_wait_list, void* event) { return 0; }
cl_int clEnqueueReadBuffer(cl_command_queue command_queue, cl_mem buffer, cl_int blocking_read, size_t offset, size_t size, void* ptr, cl_uint num_events_in_wait_list, const void* event_wait_list, void* event) { return 0; }
cl_int clGetPlatformInfo(cl_platform_id platform, uint32_t param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) { return 0; }
cl_int clGetDeviceInfo(cl_device_id device, uint32_t param_name, size_t param_value_size, void* param_value, size_t* param_value_size_ret) { return 0; }
STUB_EOF
    
    # Android build configuration
    ANDROID_API="${ANDROID_API:-21}"
    ANDROID_ARCH="${ANDROID_ARCH:-aarch64}"
    
    # Find the Android clang compiler based on detected host architecture
    NDK_TOOLCHAIN_DIR="$NDK_PATH/toolchains/llvm/prebuilt/${HOST_ARCH}"
    if [ ! -d "$NDK_TOOLCHAIN_DIR" ]; then
        # Try x86_64 variant if exact arch not found
        if [[ "$HOST_ARCH" == *"arm64"* ]]; then
            NDK_TOOLCHAIN_DIR="$NDK_PATH/toolchains/llvm/prebuilt/${HOST_ARCH/arm64/x86_64}"
        fi
    fi
    
    ANDROID_CLANG="${NDK_TOOLCHAIN_DIR}/bin/${ANDROID_ARCH}-linux-android${ANDROID_API}-clang"
    if [ ! -f "$ANDROID_CLANG" ]; then
        ANDROID_CLANG=$(find "$NDK_PATH/toolchains" -name "*${ANDROID_ARCH}*clang*" -type f | head -1)
    fi
    if [ ! -f "$ANDROID_CLANG" ]; then
        ANDROID_CLANG="${NDK_TOOLCHAIN_DIR}/bin/clang"
    fi
    
    if [ -f "$ANDROID_CLANG" ]; then
        echo "Using Android clang: $ANDROID_CLANG"
        "$ANDROID_CLANG" \
            -target ${ANDROID_ARCH}-linux-android${ANDROID_API} \
            --sysroot="${NDK_TOOLCHAIN_DIR}/sysroot" \
            -shared -fPIC \
            -Wl,-soname,libOpenCL.so \
            "$OPENCL_STUB_C" -o "$OPENCL_STUB_LIB"
        if [ -f "$OPENCL_STUB_LIB" ]; then
            echo "Created OpenCL stub library: $OPENCL_STUB_LIB"
        else
            echo "Warning: Failed to create stub library"
        fi
    else
        echo "Warning: Could not find Android clang compiler at $ANDROID_CLANG"
    fi
    rm -f "$OPENCL_STUB_C"
fi

# Check if Homebrew opencl-headers is installed and add to CMAKE_PREFIX_PATH if needed
HOMEBREW_PREFIX="${HOMEBREW_PREFIX:-}"
if [ -z "$HOMEBREW_PREFIX" ]; then
    # Try to detect Homebrew prefix
    if [ -d "/opt/homebrew" ]; then
        HOMEBREW_PREFIX="/opt/homebrew"
    elif [ -d "/usr/local" ]; then
        HOMEBREW_PREFIX="/usr/local"
    fi
fi

if [ -n "$HOMEBREW_PREFIX" ]; then
    if [ -d "${HOMEBREW_PREFIX}/opt/opencl-headers" ]; then
        export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH:+$CMAKE_PREFIX_PATH:}${HOMEBREW_PREFIX}/opt/opencl-headers"
        echo "Found Homebrew opencl-headers at ${HOMEBREW_PREFIX}/opt/opencl-headers"
    fi
fi

# Configure with Android NDK
ANDROID_API="${ANDROID_API:-21}"
ANDROID_ABI="${ANDROID_ABI:-arm64-v8a}"
CMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}"
ANDROID_STL="${ANDROID_STL:-c++_static}"

echo "Building with:"
echo "  NDK: $NDK_PATH"
echo "  Android API: $ANDROID_API"
echo "  Android ABI: $ANDROID_ABI"
echo "  Build type: $CMAKE_BUILD_TYPE"
echo "  Host arch: $HOST_ARCH"

cmake \
    -DCMAKE_TOOLCHAIN_FILE=$NDK_PATH/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=${ANDROID_ABI} \
    -DANDROID_PLATFORM=android-${ANDROID_API} \
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DANDROID_STL=${ANDROID_STL} \
    ${CMAKE_PREFIX_PATH:+-DCMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}"} \
    -S .

# Build
CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
cmake --build . --config ${CMAKE_BUILD_TYPE} -j ${CMAKE_BUILD_PARALLEL_LEVEL}

echo ""
echo "Build complete!"
echo ""
echo "Binary location: $BUILD_DIR/cartesia_opencl_test"
echo ""
echo "To use:"
echo "  1. Generate token file: python tools/dump_tokens.py 'Rene Descartes was' -o prompt_tokens.bin"
echo "  2. Copy binary and token file to device:"
echo "     adb push $BUILD_DIR/cartesia_opencl_test /data/local/tmp/"
echo "     adb push prompt_tokens.bin /data/local/tmp/"
echo "  3. Run:"
echo "     adb shell chmod +x /data/local/tmp/cartesia_opencl_test"
echo "     adb shell /data/local/tmp/cartesia_opencl_test /data/local/tmp/prompt_tokens.bin"
echo ""

