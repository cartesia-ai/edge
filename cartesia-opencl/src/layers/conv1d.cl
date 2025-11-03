// 1D Convolution kernels for SSD layer

#define SILU(x) ({ \
    float y = 1.0f / (1.0f + exp(-fabs(x))); \
    (x < 0.0f) ? (1.0f - y) * x : y * x; \
})

// Convolution forward pass (for prefill)
// x: [batch_size, n_channels, seq_len]
// w: [n_channels, kernel_size]
// b: [n_channels]
// y: [batch_size, n_channels, seq_len]
__kernel void conv1d_forward_kernel(
    __global const float* x,
    __global const float* w,
    __global const float* b,
    __global float* y,
    const int batch_size,
    const int n_channels,
    const int seq_len,
    const int kernel_size,
    const int swish_activation  // 1 if apply Swish, 0 otherwise
) {
    const int batch_idx = get_global_id(0);
    const int channel_idx = get_global_id(1);
    const int seq_idx = get_global_id(2);
    
    if (batch_idx >= batch_size || channel_idx >= n_channels || seq_idx >= seq_len) return;
    
    // Check bounds: can only compute if we have enough sequence length
    if (seq_idx + kernel_size > seq_len) {
        y[batch_idx * n_channels * seq_len + channel_idx * seq_len + seq_idx] = 0.0f;
        return;
    }
    
    float sum = 0.0f;
    int w_start = channel_idx * kernel_size;
    
    for (int k = 0; k < kernel_size; ++k) {
        int x_idx = batch_idx * n_channels * seq_len + channel_idx * seq_len + seq_idx + k;
        sum += w[w_start + k] * x[x_idx];
    }
    
    sum += b[channel_idx];
    
    if (swish_activation) {
        sum = SILU(sum);
    }
    
    int y_idx = batch_idx * n_channels * seq_len + channel_idx * seq_len + seq_idx;
    y[y_idx] = sum;
}

// Convolution update (for step function)
// x: [batch_size, n_channels] (single token)
// w: [n_channels, kernel_size]
// b: [n_channels]
// state: [batch_size, n_channels, kernel_size - 1] (conv state)
// y: [batch_size, n_channels]
// next_state: [batch_size, n_channels, kernel_size - 1]
__kernel void conv1d_update_kernel(
    __global const float* x,
    __global const float* w,
    __global const float* b,
    __global const float* state,
    __global float* y,
    __global float* next_state,
    const int batch_size,
    const int n_channels,
    const int kernel_size,
    const int swish_activation
) {
    const int batch_idx = get_global_id(0);
    const int channel_idx = get_global_id(1);
    
    if (batch_idx >= batch_size || channel_idx >= n_channels) return;
    
    int x_idx = batch_idx * n_channels + channel_idx;
    int w_start = channel_idx * kernel_size;
    int state_start = batch_idx * n_channels * (kernel_size - 1) + channel_idx * (kernel_size - 1);
    
    float sum = 0.0f;
    
    // Use state for first (kernel_size - 1) elements
    for (int k = 0; k < kernel_size - 1; ++k) {
        sum += w[w_start + k] * state[state_start + k];
    }
    
    // Use current input for last element
    sum += w[w_start + kernel_size - 1] * x[x_idx];
    sum += b[channel_idx];
    
    if (swish_activation) {
        sum = SILU(sum);
    }
    
    y[x_idx] = sum;
    
    // Update state: shift left and append new value
    for (int k = 0; k < kernel_size - 2; ++k) {
        next_state[state_start + k] = state[state_start + k + 1];
    }
    next_state[state_start + kernel_size - 2] = x[x_idx];
}

