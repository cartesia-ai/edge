#include <metal_integer>
#include <metal_math>

#include "mlx/backend/metal/kernels/bf16.h"
#include "mlx/backend/metal/kernels/erf.h"
#include "mlx/backend/metal/kernels/utils.h"


struct SILU {
  template <typename T>
  T operator()(T x) {
    auto y = 1 / (1 + metal::exp(-metal::abs(x)));
    return (x < 0) ? (1 - y) * x : y * x;
  }
};

// Stable softplus https://pytorch.org/docs/stable/generated/torch.nn.Softplus.html
struct Softplus {
  template <typename T>
  T operator()(T x) {
    auto y = log1p(metal::fast::exp(x));
    return (x > 20) ? x : y;
  };
};

// Could also be included from #include "mlx/backend/metal/kernels/unary.h" but somehow this doesnt work for multiple files
struct Exp {
  template <typename T>
  T operator()(T x) {
    return metal::fast::exp(x);
  };
};


template <typename T>
[[kernel]] void ssm_update_kernel(
    device const T* x [[buffer(0)]],
    device const T* dt [[buffer(1)]],
    device const T* A [[buffer(2)]],
    device const T* B [[buffer(3)]],
    device const T* C [[buffer(4)]],
    device const T* D [[buffer(5)]],
    device const T* z [[buffer(6)]],
    device const T* state [[buffer(7)]],
    device T* y [[buffer(8)]],
    device T* next_state [[buffer(9)]],
    uint3 grid_idx [[thread_position_in_grid]],
    uint3 grid_size [[threads_per_grid]]) {


    const int state_size = 16;
    //const int batch_size = grid_size.x;
    const int channel_size = grid_size.y;
    const int batch_idx = grid_idx.x;
    const int channel_idx = grid_idx.y;
    const int cb_start_idx = batch_idx * state_size;  // CB are data controlled

    const int x_idx = batch_idx * channel_size + channel_idx;
    const int state_start_idx = x_idx * state_size;

    T this_x = x[x_idx];
    this_x = SILU()(this_x); // SILU activation 

    T this_z = z[x_idx];
    this_z = SILU()(this_z); // SILU activation 

    T delta = Softplus()(dt[x_idx]);  // Softplus log(1 + exp(dt))

    T temp = 0;
    #pragma unroll
    for (int i = 0; i < state_size; ++i) {
        int cb_idx = cb_start_idx + i;
        int state_idx = state_start_idx + i;
        T this_new_state = state[state_idx] * Exp()(A[i] * delta) + B[cb_idx] * delta * this_x; 
        next_state[state_idx] = this_new_state;
        temp = temp + this_new_state * C[cb_idx];
    }
    temp = temp + D[channel_idx] * this_x;  // Skip connection
    temp = temp * this_z; // Out gate with z
    y[x_idx] = temp; 
} 

#define instantiate_ssm_update_kernel(type_name, type)            \
  template [[host_name("ssm_update_kernel_" #type_name)]]         \
  [[kernel]] void ssm_update_kernel<type>(                        \
    device const type* x [[buffer(0)]],                     \
    device const type* dt [[buffer(1)]],                     \
    device const type* A [[buffer(2)]],                     \
    device const type* B [[buffer(3)]],                     \
    device const type* C [[buffer(4)]],                     \
    device const type* D [[buffer(5)]],                     \
    device const type* z [[buffer(6)]],                     \
    device const type* state [[buffer(7)]],                 \
    device type* y [[buffer(8)]],                           \
    device type* next_state [[buffer(9)]],                  \
    uint3 grid_idx [[thread_position_in_grid]],             \
    uint3 grid_size [[threads_per_grid]]); 

instantiate_ssm_update_kernel(float32, float);
instantiate_ssm_update_kernel(float16, half);
//instantiate_ssm_update_kernel(bfloat16, bfloat16_t);
//instantiate_ssm_update_kernel(complex64, complex64_t);
