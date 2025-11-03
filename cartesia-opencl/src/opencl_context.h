#pragma once

#include <CL/cl.h>
#include <string>
#include <vector>
#include <memory>

namespace cartesia_opencl {

/**
 * Shared OpenCL context manager for the model.
 * Provides context, device, command queue, and program/kernel management.
 */
class OpenCLContextManager {
public:
    static OpenCLContextManager& getInstance();
    
    // Initialize OpenCL (call once at startup)
    void initialize();
    
    // Get OpenCL objects
    cl_context getContext() const { return context_; }
    cl_device_id getDevice() const { return device_; }
    cl_command_queue getQueue() const { return queue_; }
    cl_platform_id getPlatform() const { return platform_; }
    
    // Build program from source strings (kernels)
    // If cache_key is provided, tries to load from cache first, saves after successful build
    cl_program buildProgram(const std::vector<std::string>& sources, const std::string& cache_key = "");
    
    // Get kernel from program
    cl_kernel getKernel(cl_program program, const std::string& kernel_name);
    
    // Program binary caching utilities
    std::string generateCacheKey(const std::vector<std::string>& sources) const;
    std::string getCacheFilename(const std::string& cache_key) const;
    bool loadProgramBinary(const std::string& filename, std::vector<unsigned char>& binary) const;
    bool saveProgramBinary(const std::string& filename, cl_program program) const;
    cl_program createProgramFromBinary(const std::vector<unsigned char>& binary);
    
    // Cleanup (call at shutdown)
    void cleanup();
    
    ~OpenCLContextManager();

private:
    OpenCLContextManager() = default;
    OpenCLContextManager(const OpenCLContextManager&) = delete;
    OpenCLContextManager& operator=(const OpenCLContextManager&) = delete;
    
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    
    bool initialized_;
    
    void initializeOpenCL();
    std::string getBuildLog(cl_program program);
};

// Convenience typedef
using OpenCLContext = OpenCLContextManager;

} // namespace cartesia_opencl

