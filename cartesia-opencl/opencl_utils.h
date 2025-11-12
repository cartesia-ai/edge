#pragma once

#include <CL/cl.h>
#include <vector>

namespace cartesia_opencl {

/**
 * OpenCL buffer utility functions
 */

/**
 * Create and zero-initialize an OpenCL buffer for deterministic behavior.
 * 
 * @param context OpenCL context
 * @param queue Command queue for initialization
 * @param size_in_bytes Size of buffer in bytes
 * @param err_out Pointer to store error code (optional)
 * @return Created and zero-initialized buffer, or nullptr on failure
 */
inline cl_mem createAndZeroBuffer(cl_context context, cl_command_queue queue, 
                                  size_t size_in_bytes, cl_int* err_out) {
    cl_int err;
    cl_mem buffer = clCreateBuffer(context, CL_MEM_READ_WRITE, size_in_bytes, nullptr, &err);
    if (err != CL_SUCCESS || !buffer) {
        if (err_out) *err_out = err;
        return nullptr;
    }
    
    // Zero-initialize the buffer
    size_t num_floats = size_in_bytes / sizeof(float);
    std::vector<float> zeros(num_floats, 0.0f);
    err = clEnqueueWriteBuffer(queue, buffer, CL_TRUE, 0, size_in_bytes, 
                               zeros.data(), 0, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        clReleaseMemObject(buffer);
        if (err_out) *err_out = err;
        return nullptr;
    }
    
    // Ensure zero-initialization completes before buffer is used
    clFinish(queue);
    
    if (err_out) *err_out = CL_SUCCESS;
    return buffer;
}

} // namespace cartesia_opencl

