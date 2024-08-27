#include <metal_integer>
#include <metal_math>
#include <metal_stdlib>
#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/utils.h"

struct SILU {
  template <typename T>
  T operator()(T x) {
    auto y = 1 / (1 + metal::exp(-metal::abs(x)));
    return (x < 0) ? (1 - y) * x : y * x;
  }
};

template <typename T>
[[kernel]] void conv1d_update_kernel(
    device const T* x [[buffer(0)]], // b, d
    device const T* w [[buffer(1)]], // d, k
    device const T* b [[buffer(2)]], // d
    device const T* state [[buffer(3)]], // b, d, k-1
    device T* y [[buffer(4)]],
    device T* next_state [[buffer(5)]],
    constant const int& kernel_size [[buffer(6)]],
    constant const size_t* x_strides [[buffer(7)]],
    constant const size_t* state_strides [[buffer(8)]],
    uint3 grid_idx [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]) {

    const int batch_idx = grid_idx.x;
    const int channel_idx = grid_idx.y;
    const int x_idx = batch_idx * x_strides[0] + channel_idx;
    const int w_start_idx = channel_idx * kernel_size;
    const int state_start_idx = batch_idx * state_strides[0] + channel_idx * state_strides[1];

    T temp = 0;
    #pragma unroll
    for (int i = 0; i < kernel_size - 1; ++i) {
        temp = temp + w[w_start_idx + i] * state[state_start_idx + i];
    }

    temp = temp + w[w_start_idx + kernel_size - 1] * x[x_idx];
    temp = temp + b[channel_idx];
    temp = SILU()(temp);
    y[x_idx] = temp;

    #pragma unroll
    for (int i = 0; i < kernel_size - 2; ++i) {
        next_state[state_start_idx + i] = state[state_start_idx + i + 1]; 
    }
    next_state[state_start_idx + kernel_size - 2] = x[x_idx];
} 


#define instantiate_conv1d_update_kernel(type_name, type)       \
  template [[host_name("conv1d_update_kernel_" #type_name)]]    \
  [[kernel]] void conv1d_update_kernel<type>(                   \
    device const type* x [[buffer(0)]],                         \
    device const type* w [[buffer(1)]],                         \
    device const type* b [[buffer(2)]],                         \
    device const type* state [[buffer(3)]],                     \
    device type* y [[buffer(4)]],                               \
    device type* next_state [[buffer(5)]],                      \
    constant const int& kernel_size [[buffer(6)]],              \
    constant const size_t* x_strides [[buffer(7)]],             \
    constant const size_t* state_strides [[buffer(8)]],         \ 
    uint3 grid_idx [[thread_position_in_grid]],                 \
    uint3 grid_size [[threads_per_grid]]); 

instantiate_conv1d_update_kernel(float32, float);
instantiate_conv1d_update_kernel(float16, half);
//instantiate_conv1d_update_kernel(bfloat16, bfloat16_t);
//instantiate_conv1d_update_kernel(complex64, complex64_t);