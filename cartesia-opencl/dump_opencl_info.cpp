#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>

// Helper function to get string from OpenCL
std::string getString(cl_platform_id platform, cl_platform_info param) {
    size_t size;
    clGetPlatformInfo(platform, param, 0, nullptr, &size);
    std::vector<char> buffer(size);
    clGetPlatformInfo(platform, param, size, buffer.data(), nullptr);
    return std::string(buffer.data());
}

std::string getString(cl_device_id device, cl_device_info param) {
    size_t size;
    clGetDeviceInfo(device, param, 0, nullptr, &size);
    std::vector<char> buffer(size);
    clGetDeviceInfo(device, param, size, buffer.data(), nullptr);
    return std::string(buffer.data());
}

// Helper function to get numeric values
template<typename T>
T getValue(cl_device_id device, cl_device_info param) {
    T value;
    clGetDeviceInfo(device, param, sizeof(T), &value, nullptr);
    return value;
}

// Helper function to get array values
template<typename T>
std::vector<T> getArray(cl_device_id device, cl_device_info param) {
    size_t size;
    clGetDeviceInfo(device, param, 0, nullptr, &size);
    std::vector<T> result(size / sizeof(T));
    clGetDeviceInfo(device, param, size, result.data(), nullptr);
    return result;
}

void printSeparator() {
    std::cout << "========================================" << std::endl;
}

void dumpPlatformInfo(cl_platform_id platform, int platform_idx) {
    printSeparator();
    std::cout << "Platform #" << platform_idx << " Information:" << std::endl;
    printSeparator();
    
    std::cout << "  Profile: " << getString(platform, CL_PLATFORM_PROFILE) << std::endl;
    std::cout << "  Version: " << getString(platform, CL_PLATFORM_VERSION) << std::endl;
    std::cout << "  Name: " << getString(platform, CL_PLATFORM_NAME) << std::endl;
    std::cout << "  Vendor: " << getString(platform, CL_PLATFORM_VENDOR) << std::endl;
    std::cout << "  Extensions: " << getString(platform, CL_PLATFORM_EXTENSIONS) << std::endl;
    std::cout << std::endl;
}

void dumpDeviceInfo(cl_device_id device, int device_idx, cl_platform_id platform) {
    printSeparator();
    std::cout << "Device #" << device_idx << " Information:" << std::endl;
    printSeparator();
    
    // Basic info
    std::cout << "  Name: " << getString(device, CL_DEVICE_NAME) << std::endl;
    std::cout << "  Vendor: " << getString(device, CL_DEVICE_VENDOR) << std::endl;
    std::cout << "  Vendor ID: 0x" << std::hex << getValue<cl_uint>(device, CL_DEVICE_VENDOR_ID) << std::dec << std::endl;
    std::cout << "  Version: " << getString(device, CL_DEVICE_VERSION) << std::endl;
    std::cout << "  Driver Version: " << getString(device, CL_DRIVER_VERSION) << std::endl;
    std::cout << "  OpenCL C Version: " << getString(device, CL_DEVICE_OPENCL_C_VERSION) << std::endl;
    
    // Device type
    cl_device_type device_type = getValue<cl_device_type>(device, CL_DEVICE_TYPE);
    std::cout << "  Type: ";
    if (device_type & CL_DEVICE_TYPE_CPU) std::cout << "CPU ";
    if (device_type & CL_DEVICE_TYPE_GPU) std::cout << "GPU ";
    if (device_type & CL_DEVICE_TYPE_ACCELERATOR) std::cout << "Accelerator ";
    if (device_type & CL_DEVICE_TYPE_CUSTOM) std::cout << "Custom ";
    std::cout << std::endl;
    
    // Compute units
    std::cout << "  Max Compute Units: " << getValue<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS) << std::endl;
    
    // Clock frequency
    std::cout << "  Max Clock Frequency: " << getValue<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY) << " MHz" << std::endl;
    
    // Memory info
    cl_ulong mem_size = getValue<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
    std::cout << "  Global Memory Size: " << mem_size / (1024 * 1024) << " MB" << std::endl;
    
    cl_ulong alloc_size = getValue<cl_ulong>(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    std::cout << "  Max Memory Allocation: " << alloc_size / (1024 * 1024) << " MB" << std::endl;
    
    cl_ulong local_mem = getValue<cl_ulong>(device, CL_DEVICE_LOCAL_MEM_SIZE);
    std::cout << "  Local Memory Size: " << local_mem / 1024 << " KB" << std::endl;
    
    cl_bool unified = getValue<cl_bool>(device, CL_DEVICE_HOST_UNIFIED_MEMORY);
    std::cout << "  Unified Memory: " << (unified ? "Yes" : "No") << std::endl;
    
    // Work group sizes
    size_t max_wg_size = getValue<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    std::cout << "  Max Work Group Size: " << max_wg_size << std::endl;
    
    std::vector<size_t> max_dims = getArray<size_t>(device, CL_DEVICE_MAX_WORK_ITEM_SIZES);
    std::cout << "  Max Work Item Sizes: ";
    for (size_t dim : max_dims) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;
    std::cout << "  Max Work Item Dimensions: " << getValue<cl_uint>(device, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS) << std::endl;
    
    // Image support
    cl_bool img_support = getValue<cl_bool>(device, CL_DEVICE_IMAGE_SUPPORT);
    std::cout << "  Image Support: " << (img_support ? "Yes" : "No") << std::endl;
    
    if (img_support) {
        std::cout << "    Max Image 2D Width: " << getValue<size_t>(device, CL_DEVICE_IMAGE2D_MAX_WIDTH) << std::endl;
        std::cout << "    Max Image 2D Height: " << getValue<size_t>(device, CL_DEVICE_IMAGE2D_MAX_HEIGHT) << std::endl;
        std::cout << "    Max Image 3D Width: " << getValue<size_t>(device, CL_DEVICE_IMAGE3D_MAX_WIDTH) << std::endl;
        std::cout << "    Max Image 3D Height: " << getValue<size_t>(device, CL_DEVICE_IMAGE3D_MAX_HEIGHT) << std::endl;
        std::cout << "    Max Image 3D Depth: " << getValue<size_t>(device, CL_DEVICE_IMAGE3D_MAX_DEPTH) << std::endl;
    }
    
    // Floating point support
    std::cout << "  Preferred Vector Width Float: " << getValue<cl_uint>(device, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT) << std::endl;
    std::cout << "  Native Vector Width Float: " << getValue<cl_uint>(device, CL_DEVICE_NATIVE_VECTOR_WIDTH_FLOAT) << std::endl;
    
    // Half precision support (OpenCL 1.1+)
    #ifdef CL_DEVICE_HALF_FP_CONFIG
    cl_bool fp16_support = getValue<cl_bool>(device, CL_DEVICE_HALF_FP_CONFIG);
    std::cout << "  Half Precision Support: " << (fp16_support ? "Yes" : "No") << std::endl;
    #else
    std::cout << "  Half Precision Support: Unknown (CL_DEVICE_HALF_FP_CONFIG not available)" << std::endl;
    #endif
    
    // Double precision support
    cl_bool fp64_support = getValue<cl_bool>(device, CL_DEVICE_DOUBLE_FP_CONFIG);
    std::cout << "  Double Precision Support: " << (fp64_support ? "Yes" : "No") << std::endl;
    
    // Extensions
    std::cout << "  Extensions: " << getString(device, CL_DEVICE_EXTENSIONS) << std::endl;
    
    // Execution capabilities
    cl_device_exec_capabilities exec_caps = getValue<cl_device_exec_capabilities>(device, CL_DEVICE_EXECUTION_CAPABILITIES);
    std::cout << "  Execution Capabilities: ";
    if (exec_caps & CL_EXEC_KERNEL) std::cout << "Kernel ";
    if (exec_caps & CL_EXEC_NATIVE_KERNEL) std::cout << "Native ";
    std::cout << std::endl;
    
    // Queue properties
    cl_command_queue_properties queue_props = getValue<cl_command_queue_properties>(device, CL_DEVICE_QUEUE_PROPERTIES);
    std::cout << "  Queue Properties: ";
    if (queue_props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE) std::cout << "OutOfOrder ";
    if (queue_props & CL_QUEUE_PROFILING_ENABLE) std::cout << "Profiling ";
    std::cout << std::endl;
    
    // Profiling timer resolution
    size_t timer_res = getValue<size_t>(device, CL_DEVICE_PROFILING_TIMER_RESOLUTION);
    std::cout << "  Profiling Timer Resolution: " << timer_res << " ns" << std::endl;
    
    // Address bits
    std::cout << "  Address Bits: " << getValue<cl_uint>(device, CL_DEVICE_ADDRESS_BITS) << std::endl;
    
    // Endianness
    cl_bool endian = getValue<cl_bool>(device, CL_DEVICE_ENDIAN_LITTLE);
    std::cout << "  Endianness: " << (endian ? "Little" : "Big") << std::endl;
    
    std::cout << std::endl;
}

void initializeOpenCL() {
    std::cout << "Initializing OpenCL..." << std::endl;
    std::cout << std::endl;
    
    // Get number of platforms
    cl_uint num_platforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        std::cerr << "Error: No OpenCL platforms found (error code: " << err << ")" << std::endl;
        return;
    }
    
    std::cout << "Found " << num_platforms << " OpenCL platform(s)" << std::endl;
    std::cout << std::endl;
    
    // Get all platforms
    std::vector<cl_platform_id> platforms(num_platforms);
    clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    
    // Iterate through platforms
    for (cl_uint i = 0; i < num_platforms; i++) {
        dumpPlatformInfo(platforms[i], i);
        
        // Get devices for this platform
        cl_uint num_devices;
        
        // Try GPU first
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            std::vector<cl_device_id> devices(num_devices);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, num_devices, devices.data(), nullptr);
            
            std::cout << "Found " << num_devices << " GPU device(s)" << std::endl;
            std::cout << std::endl;
            
            for (cl_uint j = 0; j < num_devices; j++) {
                dumpDeviceInfo(devices[j], j, platforms[i]);
            }
        }
        
        // Try CPU
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, nullptr, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            std::vector<cl_device_id> devices(num_devices);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, num_devices, devices.data(), nullptr);
            
            std::cout << "Found " << num_devices << " CPU device(s)" << std::endl;
            std::cout << std::endl;
            
            for (cl_uint j = 0; j < num_devices; j++) {
                dumpDeviceInfo(devices[j], j, platforms[i]);
            }
        }
        
        // Try ALL devices
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &num_devices);
        if (err == CL_SUCCESS && num_devices > 0) {
            std::vector<cl_device_id> devices(num_devices);
            clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, num_devices, devices.data(), nullptr);
            
            std::cout << "Total devices on platform: " << num_devices << std::endl;
            std::cout << std::endl;
        }
    }
    
    // Try to create a context with the first GPU device (if available)
    std::cout << "========================================" << std::endl;
    std::cout << "Testing Context Creation:" << std::endl;
    std::cout << "========================================" << std::endl;
    
    cl_uint num_platforms_test;
    clGetPlatformIDs(1, &platforms[0], &num_platforms_test);
    if (num_platforms_test > 0) {
        cl_device_id test_device;
        err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 1, &test_device, nullptr);
        if (err != CL_SUCCESS) {
            err = clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_CPU, 1, &test_device, nullptr);
        }
        
        if (err == CL_SUCCESS) {
            cl_context context = clCreateContext(nullptr, 1, &test_device, nullptr, nullptr, &err);
            if (err == CL_SUCCESS) {
                std::cout << "✓ Successfully created OpenCL context!" << std::endl;
                clReleaseContext(context);
            } else {
                std::cout << "✗ Failed to create context (error: " << err << ")" << std::endl;
            }
        } else {
            std::cout << "✗ No devices available for context creation" << std::endl;
        }
    }
    
    std::cout << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "OpenCL Device Information Dump" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;
    
    initializeOpenCL();
    
    std::cout << "========================================" << std::endl;
    std::cout << "Done!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}

