# OpenCL Device Info Dumper

This simple program initializes OpenCL and dumps all available device information to help debug and understand OpenCL capabilities on your Android device.

## Files

- `dump_opencl_info.cpp` - Main C++ program that initializes OpenCL and prints device info
- `build_android_dump.sh` - Build script for Android (requires Android NDK)
- `CMakeLists_dump.txt` - CMake configuration for building the dump program

## Quick Start (Android)

### Option 1: Using the Build Script

1. Make sure you have Android NDK installed
2. Set `ANDROID_NDK` environment variable (or it will try to auto-detect):
   ```bash
   export ANDROID_NDK=$HOME/Android/Sdk/ndk/25.2.9519653  # or your NDK path
   ```

3. Run the build script:
   ```bash
   chmod +x build_android_dump.sh
   ./build_android_dump.sh
   ```

4. Copy and run on device:
   ```bash
   adb push build_android/dump_opencl_info /data/local/tmp/
   adb shell chmod +x /data/local/tmp/dump_opencl_info
   adb shell /data/local/tmp/dump_opencl_info
   ```

### Option 2: Manual Build with Android NDK

If you prefer to build manually:

```bash
# Set up NDK paths
export ANDROID_NDK=$HOME/Android/Sdk/ndk/<version>
export ANDROID_API=21

# Create build directory
mkdir -p build_android
cd build_android

# Configure
cmake \
    -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-$ANDROID_API \
    -DCMAKE_BUILD_TYPE=Release \
    -DANDROID_STL=c++_shared \
    -S ..

# Build
cmake --build .

# Deploy and run
adb push dump_opencl_info /data/local/tmp/
adb shell chmod +x /data/local/tmp/dump_opencl_info
adb shell /data/local/tmp/dump_opencl_info
```

### Option 3: Using Standalone Toolchain (Alternative)

```bash
# Set up standalone toolchain
$ANDROID_NDK/build/tools/make_standalone_toolchain.py \
    --arch arm64 --api 21 --install-dir /tmp/android-toolchain

# Compile directly
/tmp/android-toolchain/bin/aarch64-linux-android-clang++ \
    -I/path/to/opencl-headers \
    -L/vendor/lib64 \
    dump_opencl_info.cpp \
    -lOpenCL \
    -o dump_opencl_info \
    -static-libstdc++

# Deploy
adb push dump_opencl_info /data/local/tmp/
adb shell chmod +x /data/local/tmp/dump_opencl_info
adb shell /data/local/tmp/dump_opencl_info
```

## What It Does

The program:
1. Calls `initializeOpenCL()` similar to your main codebase
2. Discovers all OpenCL platforms
3. Discovers all devices (GPU, CPU, etc.)
4. Dumps comprehensive device information:
   - Platform info (name, vendor, version, extensions)
   - Device info (name, type, compute units, memory, etc.)
   - GPU capabilities (work group sizes, vector widths, etc.)
   - Floating point support (half, single, double precision)
   - Image support
   - Extensions and features
5. Tests context creation to verify OpenCL is working

## Output

The program prints detailed information about:
- All available OpenCL platforms
- All devices on each platform (GPU, CPU, etc.)
- Device capabilities and limits
- Memory information
- Compute capabilities
- OpenCL version support
- Extensions available

This helps you understand:
- What OpenCL devices are available
- Their capabilities and limitations
- Whether your code can run on the device
- What features you can use

## Expected Output

When you successfully build and run `build_android_dump.sh` on your Android device, you should see similar output to this:

```
OpenCL Device Information Dump
========================================

Initializing OpenCL...

Found 1 OpenCL platform(s)

========================================
Platform #0 Information:
========================================
  Profile: EMBEDDED_PROFILE
  Version: OpenCL 1.2 
  Name: PowerVR
  Vendor: Imagination Technologies
  Extensions: cl_khr_icd cl_khr_fp16 cl_img_spirv cles_khr_int64 cl_img_yuv_image cl_img_generate_mipmap cl_khr_3d_image_writes cl_img_cached_allocations cl_khr_extended_versioning cl_khr_image2d_from_buffer cl_khr_byte_addressable_store cl_khr_local_int32_base_atomics cl_khr_global_int32_base_atomics cl_arm_non_uniform_work_group_size cl_khr_local_int32_extended_atomics cl_khr_global_int32_extended_atomics cl_khr_spir cl_arm_import_memory cl_arm_import_memory_dma_buf cl_img_use_gralloc_ptr cl_img_protected_content cl_img_use_gralloc_ptr_v2

Found 1 GPU device(s)

========================================
Device #0 Information:
========================================
  Name: PowerVR GE8320
  Vendor: Imagination Technologies
  Vendor ID: 0x1
  Version: OpenCL 1.2 
  Driver Version: 1.13@5776728
  OpenCL C Version: OpenCL C 1.2 
  Type: GPU 
  Max Compute Units: 1
  Max Clock Frequency: 400 MHz
  Global Memory Size: 1367 MB
  Max Memory Allocation: 341 MB
  Local Memory Size: 4 KB
  Unified Memory: Yes
  Max Work Group Size: 512
```

## Troubleshooting

### "OpenCL library not found" at build time
- This is OK for Android! The library exists on the device at runtime
- The build script should handle this, but you may need to adjust paths

### "No platforms found" at runtime
- Make sure your device has OpenCL support
- Check that `/vendor/lib64/libOpenCL.so` exists on device:
  ```bash
  adb shell ls -la /vendor/lib64/libOpenCL.so
  ```

### Permission denied
- Make sure you're using `/data/local/tmp/` which is writable
- Or push to a location you have write access to

### Library not found at runtime
- Set LD_LIBRARY_PATH if needed:
  ```bash
  adb shell "LD_LIBRARY_PATH=/vendor/lib64:/system/lib64 /data/local/tmp/dump_opencl_info"
  ```

