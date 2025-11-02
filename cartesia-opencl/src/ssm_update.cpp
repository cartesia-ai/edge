#include "ssm_update.h"
#include <CL/cl.h>
#include <vector>
#include <string>
#include <stdexcept>

namespace cartesia_opencl {

class OpenCLContext {
private:
    cl_platform_id platform_;
    cl_device_id device_;
    cl_context context_;
    cl_command_queue queue_;
    cl_program program_;
    
    // Kernel handles
    cl_kernel ssm_update_kernel_fp32_;
    cl_kernel ssm_update_kernel_fp16_;

public:
    OpenCLContext() {
        initializeOpenCL();
        buildProgram();
        createKernels();
    }
    
    ~OpenCLContext() {
        cleanup();
    }
    
    void ssm_update(
        const std::vector<float>& x,
        const std::vector<float>& dt,
        const std::vector<float>& A,
        const std::vector<float>& B,
        const std::vector<float>& C,
        const std::vector<float>& D,
        const std::vector<float>& z,
        const std::vector<float>& state,
        std::vector<float>& y,
        std::vector<float>& next_state,
        int batch_size,
        int channel_size,
        int state_size
    ) {
        // Create buffers
        cl_mem x_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     x.size() * sizeof(float), (void*)x.data(), nullptr);
        cl_mem dt_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      dt.size() * sizeof(float), (void*)dt.data(), nullptr);
        cl_mem A_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     A.size() * sizeof(float), (void*)A.data(), nullptr);
        cl_mem B_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     B.size() * sizeof(float), (void*)B.data(), nullptr);
        cl_mem C_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     C.size() * sizeof(float), (void*)C.data(), nullptr);
        cl_mem D_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     D.size() * sizeof(float), (void*)D.data(), nullptr);
        cl_mem z_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     z.size() * sizeof(float), (void*)z.data(), nullptr);
        cl_mem state_buf = clCreateBuffer(context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                         state.size() * sizeof(float), (void*)state.data(), nullptr);
        cl_mem y_buf = clCreateBuffer(context_, CL_MEM_WRITE_ONLY,
                                     y.size() * sizeof(float), nullptr, nullptr);
        cl_mem next_state_buf = clCreateBuffer(context_, CL_MEM_WRITE_ONLY,
                                              next_state.size() * sizeof(float), nullptr, nullptr);

        // Set kernel arguments
        cl_kernel kernel = ssm_update_kernel_fp32_;
        clSetKernelArg(kernel, 0, sizeof(cl_mem), &x_buf);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), &dt_buf);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), &A_buf);
        clSetKernelArg(kernel, 3, sizeof(cl_mem), &B_buf);
        clSetKernelArg(kernel, 4, sizeof(cl_mem), &C_buf);
        clSetKernelArg(kernel, 5, sizeof(cl_mem), &D_buf);
        clSetKernelArg(kernel, 6, sizeof(cl_mem), &z_buf);
        clSetKernelArg(kernel, 7, sizeof(cl_mem), &state_buf);
        clSetKernelArg(kernel, 8, sizeof(cl_mem), &y_buf);
        clSetKernelArg(kernel, 9, sizeof(cl_mem), &next_state_buf);
        clSetKernelArg(kernel, 10, sizeof(int), &state_size);
        clSetKernelArg(kernel, 11, sizeof(int), &channel_size);

        // Execute kernel
        size_t global_size[2] = {static_cast<size_t>(batch_size), static_cast<size_t>(channel_size)};
        size_t local_size[2] = {32, 32}; // Adjust based on device capabilities
        
        clEnqueueNDRangeKernel(queue_, kernel, 2, nullptr, global_size, local_size, 0, nullptr, nullptr);
        
        // Read results
        clEnqueueReadBuffer(queue_, y_buf, CL_TRUE, 0, y.size() * sizeof(float), y.data(), 0, nullptr, nullptr);
        clEnqueueReadBuffer(queue_, next_state_buf, CL_TRUE, 0, next_state.size() * sizeof(float), next_state.data(), 0, nullptr, nullptr);

        // Cleanup buffers
        clReleaseMemObject(x_buf);
        clReleaseMemObject(dt_buf);
        clReleaseMemObject(A_buf);
        clReleaseMemObject(B_buf);
        clReleaseMemObject(C_buf);
        clReleaseMemObject(D_buf);
        clReleaseMemObject(z_buf);
        clReleaseMemObject(state_buf);
        clReleaseMemObject(y_buf);
        clReleaseMemObject(next_state_buf);
    }

private:
    void initializeOpenCL() {
        // Get platform
        cl_uint num_platforms;
        clGetPlatformIDs(0, nullptr, &num_platforms);
        std::vector<cl_platform_id> platforms(num_platforms);
        clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
        platform_ = platforms[0]; // Use first platform

        // Get device
        cl_uint num_devices;
        clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
        if (num_devices == 0) {
            // Fallback to CPU if no GPU available
            clGetDeviceIDs(platform_, CL_DEVICE_TYPE_CPU, 1, &device_, nullptr);
        } else {
            clGetDeviceIDs(platform_, CL_DEVICE_TYPE_GPU, 1, &device_, nullptr);
        }

        // Create context and command queue
        context_ = clCreateContext(nullptr, 1, &device_, nullptr, nullptr, nullptr);
        queue_ = clCreateCommandQueue(context_, device_, 0, nullptr);
    }

    void buildProgram() {
        // Load OpenCL source code
        std::string source = R"(
            // Include the OpenCL kernel source here
            #define SILU(x) ({ \
                float y = 1.0f / (1.0f + exp(-fabs(x))); \
                (x < 0.0f) ? (1.0f - y) * x : y * x; \
            })

            #define SOFTPLUS(x) ({ \
                float y = log1p(exp(x)); \
                (x > 20.0f) ? x : y; \
            })

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
                
                const int cb_start_idx = batch_idx * state_size;
                const int x_idx = batch_idx * channel_size + channel_idx;
                const int state_start_idx = x_idx * state_size;

                float this_x = x[x_idx];
                this_x = SILU(this_x);

                float this_z = z[x_idx];
                this_z = SILU(this_z);

                float delta = SOFTPLUS(dt[x_idx]);

                float temp = 0.0f;
                for (int i = 0; i < state_size; ++i) {
                    int cb_idx = cb_start_idx + i;
                    int state_idx = state_start_idx + i;
                    float this_new_state = state[state_idx] * exp(A[i] * delta) + B[cb_idx] * delta * this_x; 
                    next_state[state_idx] = this_new_state;
                    temp = temp + this_new_state * C[cb_idx];
                }
                temp = temp + D[channel_idx] * this_x;
                temp = temp * this_z;
                y[x_idx] = temp; 
            }
        )";

        const char* source_ptr = source.c_str();
        program_ = clCreateProgramWithSource(context_, 1, &source_ptr, nullptr, nullptr);
        clBuildProgram(program_, 1, &device_, nullptr, nullptr, nullptr);
    }

    void createKernels() {
        ssm_update_kernel_fp32_ = clCreateKernel(program_, "ssm_update_kernel", nullptr);
        // Add fp16 kernel creation if supported
    }

    void cleanup() {
        clReleaseKernel(ssm_update_kernel_fp32_);
        clReleaseKernel(ssm_update_kernel_fp16_);
        clReleaseProgram(program_);
        clReleaseCommandQueue(queue_);
        clReleaseContext(context_);
    }
};

// Global OpenCL context
static OpenCLContext* g_opencl_context = nullptr;

void initialize_opencl() {
    if (!g_opencl_context) {
        g_opencl_context = new OpenCLContext();
    }
}

void cleanup_opencl() {
    if (g_opencl_context) {
        delete g_opencl_context;
        g_opencl_context = nullptr;
    }
}

std::vector<array> ssm_update(
    const array& x,
    const array& dt,
    const array& A,
    const array& B,
    const array& C,
    const array& D,
    const array& z,
    const array& state
) {
    if (!g_opencl_context) {
        initialize_opencl();
    }

    // Convert arrays to vectors (simplified - in practice you'd want more efficient data handling)
    std::vector<float> x_vec(x.data(), x.data() + x.size());
    std::vector<float> dt_vec(dt.data(), dt.data() + dt.size());
    std::vector<float> A_vec(A.data(), A.data() + A.size());
    std::vector<float> B_vec(B.data(), B.data() + B.size());
    std::vector<float> C_vec(C.data(), C.data() + C.size());
    std::vector<float> D_vec(D.data(), D.data() + D.size());
    std::vector<float> z_vec(z.data(), z.data() + z.size());
    std::vector<float> state_vec(state.data(), state.data() + state.size());

    std::vector<float> y_vec(x.size());
    std::vector<float> next_state_vec(state.size());

    int batch_size = x.shape(0);
    int channel_size = x.shape(1);
    int state_size = A.shape(0);

    g_opencl_context->ssm_update(
        x_vec, dt_vec, A_vec, B_vec, C_vec, D_vec, z_vec, state_vec,
        y_vec, next_state_vec, batch_size, channel_size, state_size
    );

    // Convert back to arrays (simplified)
    array y(y_vec.data(), x.shape());
    array next_state(next_state_vec.data(), state.shape());

    return {y, next_state};
}

} // namespace cartesia_opencl


