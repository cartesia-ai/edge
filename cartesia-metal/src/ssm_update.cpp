#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "src/ssm_update.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace mlx::core {

std::vector<array> ssm_update(
    const array& x,
    const array& dt,
    const array& A,
    const array& B,
    const array& C,
    const array& D,
    const array& z,
    const array& state,
    StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
  auto y_dtype = x.dtype();
  // TODO: Also make sure state is of the same dtype

  auto y_shape = x.shape();
  auto state_shape = state.shape();

  return array::make_arrays(
      {y_shape, state_shape},
      {y_dtype, y_dtype},
      std::make_shared<SSMUpdate>(to_stream(s)),
      {x, dt, A, B, C, D, z, state});
}

void SSMUpdate::eval(const std::vector<array>& inputs,  std::vector<array>& outputs) {
    throw std::runtime_error("eval not implemented!");
}


#ifdef ACCELERATE_NEW_LAPACK

void SSMUpdate::eval_cpu(const std::vector<array>& inputs,  std::vector<array>& outputs) {
    throw std::runtime_error("eval_cpu not implemented!");
}

#endif


void SSMUpdate::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {

  assert(inputs.size() == 8);
  assert(outputs.size() == 2);

  auto x = inputs[0];
  auto dt = inputs[1];
  auto A = inputs[2];
  auto B = inputs[3];
  auto C = inputs[4];
  auto D = inputs[5];
  auto z = inputs[6];
  auto state = inputs[7];

  auto y = outputs[0];
  auto next_state = outputs[1];

  auto& s = stream();
  auto& d = metal::device(s.device);

  y.set_data(
    allocator::malloc_or_wait(x.data_size() * y.itemsize()),
    x.data_size(),
    x.strides(),
    x.flags()
  );

  next_state.set_data(
    allocator::malloc_or_wait(state.data_size() * state.itemsize()),
    state.data_size(),
    state.strides(),
    state.flags()
  );

  std::ostringstream kname;
  kname << "ssm_update_kernel_";
  kname << type_to_name(x);
  
  d.register_library("mlx_ext", metal::get_colocated_mtllib_path);
  auto kernel = d.get_kernel(kname.str(), "mlx_ext");
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);

  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(dt, 1);
  compute_encoder.set_input_array(A, 2);
  compute_encoder.set_input_array(B, 3);
  compute_encoder.set_input_array(C, 4);
  compute_encoder.set_input_array(D, 5);
  compute_encoder.set_input_array(z, 6);
  compute_encoder.set_input_array(state, 7);

  compute_encoder.set_output_array(y, 8);
  compute_encoder.set_output_array(next_state, 9);

  auto batch_size = x.shape(0);
  auto n_channels = x.shape(1);
  //auto n_state = A.shape(0);

  // https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes
  MTL::Size grid_dims = MTL::Size(batch_size, n_channels, 1);
  // size_t width = kernel->threadExecutionWidth();
  // size_t height = kernel->maxTotalThreadsPerThreadgroup() / width; 
  MTL::Size group_dims = MTL::Size(32, 32, 1);
  
  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

} // namespace mlx::core