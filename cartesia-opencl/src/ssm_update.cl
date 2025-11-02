// OpenCL kernel for SSM update operation
// Based on the Metal implementation from cartesia-metal

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
    
    const int cb_start_idx = batch_idx * state_size;  // CB are data controlled
    const int x_idx = batch_idx * channel_size + channel_idx;
    const int state_start_idx = x_idx * state_size;

    float this_x = x[x_idx];
    this_x = SILU(this_x); // SILU activation 

    float this_z = z[x_idx];
    this_z = SILU(this_z); // SILU activation 

    float delta = SOFTPLUS(dt[x_idx]);  // Softplus log(1 + exp(dt))

    float temp = 0.0f;
    for (int i = 0; i < state_size; ++i) {
        int cb_idx = cb_start_idx + i;
        int state_idx = state_start_idx + i;
        float this_new_state = state[state_idx] * exp(A[i] * delta) + B[cb_idx] * delta * this_x; 
        next_state[state_idx] = this_new_state;
        temp = temp + this_new_state * C[cb_idx];
    }
    temp = temp + D[channel_idx] * this_x;  // Skip connection
    temp = temp * this_z; // Out gate with z
    y[x_idx] = temp; 
}

// Half precision version for better performance on supported devices
__kernel void ssm_update_kernel_fp16(
    __global const half* x,
    __global const half* dt,
    __global const half* A,
    __global const half* B,
    __global const half* C,
    __global const half* D,
    __global const half* z,
    __global const half* state,
    __global half* y,
    __global half* next_state,
    const int state_size,
    const int channel_size
) {
    const int batch_idx = get_global_id(0);
    const int channel_idx = get_global_id(1);
    
    const int cb_start_idx = batch_idx * state_size;
    const int x_idx = batch_idx * channel_size + channel_idx;
    const int state_start_idx = x_idx * state_size;

    half this_x = x[x_idx];
    this_x = SILU(this_x);

    half this_z = z[x_idx];
    this_z = SILU(this_z);

    half delta = SOFTPLUS(dt[x_idx]);

    half temp = 0.0h;
    for (int i = 0; i < state_size; ++i) {
        int cb_idx = cb_start_idx + i;
        int state_idx = state_start_idx + i;
        half this_new_state = state[state_idx] * exp(A[i] * delta) + B[cb_idx] * delta * this_x; 
        next_state[state_idx] = this_new_state;
        temp = temp + this_new_state * C[cb_idx];
    }
    temp = temp + D[channel_idx] * this_x;
    temp = temp * this_z;
    y[x_idx] = temp; 
}
