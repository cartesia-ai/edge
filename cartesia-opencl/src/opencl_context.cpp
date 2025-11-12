#include "opencl_context.h"
#include <stdexcept>
#include <sstream>
#include <fstream>
#include <iostream>
#include <cstring>
#include <functional>
#include <iomanip>
#include <cstdlib>

namespace cartesia_opencl {

OpenCLContextManager& OpenCLContextManager::getInstance() {
    static OpenCLContextManager instance;
    return instance;
}

void OpenCLContextManager::initialize() {
    if (initialized_) return;
    initializeOpenCL();
    initialized_ = true;
}

void OpenCLContextManager::initializeOpenCL() {
    cl_int err;
    
    // Get platform
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, nullptr, &num_platforms);
    if (err != CL_SUCCESS || num_platforms == 0) {
        throw std::runtime_error("No OpenCL platforms found");
    }
    
    std::vector<cl_platform_id> platforms(num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        throw std::runtime_error("Failed to get OpenCL platforms");
    }
    platform_ = platforms[0];
    
    // Get device (prefer GPU, fallback to CPU)
    cl_uint num_devices;
    err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
    if (num_devices == 0 || err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, 1, &device_, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("No OpenCL devices found");
        }
    } else {
        err = clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
        if (err != CL_SUCCESS) {
            throw std::runtime_error("Failed to get GPU device");
        }
    }
    
    // Create context
    context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, &err);
    if (err != CL_SUCCESS || !context_) {
        throw std::runtime_error("Failed to create OpenCL context: " + std::to_string(err));
    }
    
    // Create command queue with in-order execution (flags=0) for determinism
    // In-order execution ensures commands complete in submission order
    queue_ = clCreateCommandQueue(context_, device_, 0, &err);
    if (err != CL_SUCCESS || !queue_) {
        clReleaseContext(context_);
        throw std::runtime_error("Failed to create command queue: " + std::to_string(err));
    }

    // Print device info for diagnostics
    char platform_name[256] = {0};
    char device_name[256] = {0};
    char device_vendor[256] = {0};
    char driver_version[256] = {0};
    clGetPlatformInfo(platform_, CL_PLATFORM_NAME, sizeof(platform_name), platform_name, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(device_name), device_name, nullptr);
    clGetDeviceInfo(device_, CL_DEVICE_VENDOR, sizeof(device_vendor), device_vendor, nullptr);
    clGetDeviceInfo(device_, CL_DRIVER_VERSION, sizeof(driver_version), driver_version, nullptr);
    std::cout << "OpenCL Platform: " << platform_name << "\n"
              << "OpenCL Device:   " << device_name << " (" << device_vendor << ")\n"
              << "Driver Version:  " << driver_version << std::endl;
}

cl_program OpenCLContextManager::buildProgram(const std::vector<std::string>& sources, const std::string& cache_key) {
    if (!initialized_) {
        throw std::runtime_error("OpenCLContextManager not initialized");
    }
    
    // Try to load from cache if cache_key is provided
    if (!cache_key.empty()) {
        std::string cache_filename = getCacheFilename(cache_key);
        std::vector<unsigned char> binary;
        
        if (loadProgramBinary(cache_filename, binary)) {
            // std::cout << "\n      Loading program binary from cache..." << std::flush;
            try {
                cl_program program = createProgramFromBinary(binary);
                // Some vendors require a finalize step, but most binaries work immediately
                // Try to build anyway (will succeed immediately for valid binaries)
                cl_int err = clBuildProgram(program, 1, &device_, nullptr, nullptr, nullptr);
                if (err == CL_SUCCESS) {
                    // std::cout << " ✓ (loaded from cache, " << (binary.size() / 1024) << " KB)" << std::flush;
                    return program;
                } else {
                    // Binary is invalid (e.g., driver updated), fall through to rebuild
                    std::cout << "\n      Cache binary invalid (err=" << err << "), rebuilding..." << std::flush;
                    clReleaseProgram(program);
                }
            } catch (const std::exception& e) {
                // Cache load failed, fall through to rebuild
                std::cout << "\n      Cache load failed: " << e.what() << ", rebuilding..." << std::flush;
            } catch (...) {
                // Cache load failed, fall through to rebuild
                std::cout << "\n      Cache load failed (unknown error), rebuilding..." << std::flush;
            }
        }
    }
    
    // Build from source
    std::vector<const char*> source_ptrs;
    std::vector<size_t> source_sizes;
    
    for (const auto& source : sources) {
        source_ptrs.push_back(source.c_str());
        source_sizes.push_back(source.length());
    }
    
    cl_int err;
    cl_program program = clCreateProgramWithSource(
        context_, sources.size(), source_ptrs.data(), source_sizes.data(), &err
    );
    
    if (err != CL_SUCCESS || !program) {
        throw std::runtime_error("Failed to create OpenCL program: " + std::to_string(err));
    }
    
    // Build options: allow override via env, default to deterministic settings
    // Use conservative flags to ensure deterministic floating-point behavior:
    // - cl-opt-disable: Disable optimizations that may cause non-determinism
    // - cl-fp32-correctly-rounded-divide-sqrt: Ensure correct rounding for div/sqrt
    // - cl-no-signed-zeros: disabled (omitted) to preserve signed zeros
    // - cl-mad-enable: disabled (omitted) to avoid fused multiply-add non-determinism
    // - cl-finite-math-only: disabled (omitted) to handle NaN/Inf consistently
    const char* env_opts = std::getenv("OPENCL_BUILD_OPTS");
    std::string build_opts = env_opts ? std::string(env_opts) : 
        std::string("-cl-std=CL1.2 -cl-opt-disable");
    err = clBuildProgram(program, 1, &device_, build_opts.c_str(), nullptr, nullptr);
    if (err != CL_SUCCESS) {
        std::string build_log = getBuildLog(program);
        clReleaseProgram(program);
        throw std::runtime_error("Failed to build OpenCL program: " + std::to_string(err) + "\nBuild options: " + build_opts + "\n" + build_log);
    }
    // Optionally print build log on success when verbose requested
    if (const char* verbose = std::getenv("OPENCL_BUILD_VERBOSE"); verbose && std::string(verbose) == "1") {
        std::string log = getBuildLog(program);
        if (!log.empty()) {
            std::cout << "[OpenCL] Build log (success):\n" << log << std::endl;
        }
    }
    
    // Save to cache if cache_key is provided
    if (!cache_key.empty()) {
        std::string cache_filename = getCacheFilename(cache_key);
        if (saveProgramBinary(cache_filename, program)) {
            std::cout << "\n      Saved program binary to cache" << std::flush;
        }
    }
    
    return program;
}

std::string OpenCLContextManager::generateCacheKey(const std::vector<std::string>& sources) const {
    // Simple hash based on concatenated source strings
    std::string combined;
    for (const auto& src : sources) {
        combined += src;
    }
    
    // Compute simple hash
    std::hash<std::string> hasher;
    size_t hash = hasher(combined);
    
    std::stringstream ss;
    ss << std::hex << std::setfill('0') << std::setw(16) << hash;
    return ss.str();
}

std::string OpenCLContextManager::getCacheFilename(const std::string& cache_key) const {
    // Get device name for cache directory
    char device_name[256] = {0};
    size_t name_size = 0;
    clGetDeviceInfo(device_, CL_DEVICE_NAME, sizeof(device_name), device_name, &name_size);
    
    // Create cache directory path
    std::string cache_dir = "/data/local/tmp/cartesia_kernel_cache";
    std::string device_str = std::string(device_name);
    // Sanitize device name for filesystem
    for (char& c : device_str) {
        if (c == ' ' || c == '/') c = '_';
    }
    
    return cache_dir + "/" + device_str + "_" + cache_key + ".bin";
}

bool OpenCLContextManager::loadProgramBinary(const std::string& filename, std::vector<unsigned char>& binary) const {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return false;
    }
    
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    binary.resize(size);
    file.read(reinterpret_cast<char*>(binary.data()), size);
    
    return file.good();
}

bool OpenCLContextManager::saveProgramBinary(const std::string& filename, cl_program program) const {
    // Get binary sizes
    size_t num_devices = 1;
    size_t binary_size = 0;
    cl_int err = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &binary_size, nullptr);
    if (err != CL_SUCCESS || binary_size == 0) {
        return false;
    }
    
    // Get binary
    unsigned char* binary_data = new unsigned char[binary_size];
    unsigned char* binaries[] = {binary_data};
    size_t binary_sizes[] = {binary_size};
    
    err = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(unsigned char*), binaries, nullptr);
    if (err != CL_SUCCESS) {
        delete[] binary_data;
        return false;
    }
    
    // Create cache directory
    size_t last_slash = filename.find_last_of('/');
    if (last_slash != std::string::npos) {
        std::string dir = filename.substr(0, last_slash);
        // Simple mkdir (requires creating parent dirs manually or using system call)
        std::string cmd = "mkdir -p " + dir;
        system(cmd.c_str());
    }
    
    // Save binary to file
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        delete[] binary_data;
        return false;
    }
    
    file.write(reinterpret_cast<const char*>(binary_data), binary_size);
    file.close();
    
    delete[] binary_data;
    return file.good();
}

cl_program OpenCLContextManager::createProgramFromBinary(const std::vector<unsigned char>& binary) {
    const unsigned char* binary_ptr = binary.data();
    size_t binary_size = binary.size();
    
    cl_int err;
    cl_int binary_status = CL_SUCCESS;
    cl_program program = clCreateProgramWithBinary(
        context_, 1, &device_, &binary_size, &binary_ptr, &binary_status, &err
    );
    
    if (err != CL_SUCCESS || !program || binary_status != CL_SUCCESS) {
        throw std::runtime_error("Failed to create program from binary: " + std::to_string(err));
    }
    
    return program;
}

cl_kernel OpenCLContextManager::getKernel(cl_program program, const std::string& kernel_name) {
    cl_int err;
    cl_kernel kernel = clCreateKernel(program, kernel_name.c_str(), &err);
    if (err != CL_SUCCESS || !kernel) {
        throw std::runtime_error("Failed to create kernel '" + kernel_name + "': " + std::to_string(err));
    }
    return kernel;
}

std::string OpenCLContextManager::getBuildLog(cl_program program) {
    size_t log_size;
    clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
    
    if (log_size == 0) return "";
    
    std::vector<char> log(log_size);
    clGetProgramBuildInfo(program, device_, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
    return std::string(log.data());
}

void OpenCLContextManager::cleanup() {
    if (queue_) clReleaseCommandQueue(queue_);
    if (context_) clReleaseContext(context_);
    queue_ = nullptr;
    context_ = nullptr;
    device_ = nullptr;
    platform_ = nullptr;
    initialized_ = false;
}

OpenCLContextManager::~OpenCLContextManager() {
    cleanup();
}

} // namespace cartesia_opencl

