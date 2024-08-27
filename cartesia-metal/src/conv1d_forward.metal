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
[[kernel]] void conv1d_forward_kernel(
    device const T* x [[buffer(0)]],      // (b, d, l)
    device const T* w [[buffer(1)]],      // (d, k)
    device const T* b [[buffer(2)]],      // (d)
    device T* y [[buffer(3)]],
    constant const size_t* x_strides [[buffer(4)]],
    constant const int& kernel_size [[buffer(5)]],
    uint3 grid_idx [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]) {

    const int length_size = grid_size.z;
    const int batch_idx = grid_idx.x;
    const int channel_idx = grid_idx.y;
    const int length_idx = grid_idx.z;

    const int y_idx = batch_idx * x_strides[0] + channel_idx * x_strides[1] + length_idx;
    const int w_start_idx = channel_idx * kernel_size;

    T acc = 0;

    if (length_idx + kernel_size - 1 < length_size) {  // prevents out of bounds access and assumes last k-1 elements at end are dropped
      #pragma unroll
      for (int i = 0; i < kernel_size; ++i) {
          acc = acc + w[w_start_idx + i] * x[y_idx + i];
      }
      acc = acc + b[channel_idx];
      acc = SILU()(acc);
    }

     y[y_idx] = acc; 
} 


#define instantiate_conv1d_forward_kernel(type_name, type)       \
  template [[host_name("conv1d_forward_kernel_" #type_name)]]    \
  [[kernel]] void conv1d_forward_kernel<type>(                   \
    device const type* x [[buffer(0)]],                         \
    device const type* w [[buffer(1)]],                         \
    device const type* b [[buffer(2)]],                         \
    device type* y [[buffer(3)]],                               \
    constant const size_t* x_strides [[buffer(4)]],             \
    constant const int& kernel_size [[buffer(5)]],              \
    uint3 grid_idx [[thread_position_in_grid]],                 \
    uint3 grid_size [[threads_per_grid]]); 

instantiate_conv1d_forward_kernel(float32, float);
instantiate_conv1d_forward_kernel(float16, half);
//instantiate_conv1d_forward_kernel(bfloat16, bfloat16_t);
//instantiate_conv1d_forward_kernel(complex64, complex64_t);