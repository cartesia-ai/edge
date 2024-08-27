#include <cassert>
#include <iostream>
#include <sstream>

#include "mlx/backend/common/copy.h"
#include "mlx/backend/common/utils.h"
#include "mlx/utils.h"

#include "src/conv1d_forward.h"

#ifdef ACCELERATE_NEW_LAPACK
#include <vecLib/cblas_new.h>
#endif

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/utils.h"

namespace mlx::core {

std::vector<array> conv1d_forward(
    const array& x,
    const array& w, 
    const array& b, 
    StreamOrDevice s /* = {} */ // Stream on which to schedule the operation
) {
  auto y_dtype = x.dtype();
  // TODO: Also make sure state is of the same dtype

  auto y_shape = x.shape();

  return array::make_arrays(
      {y_shape},
      {y_dtype},
      std::make_shared<Conv1dForward>(to_stream(s)),
      {x, w, b});
}

void Conv1dForward::eval(const std::vector<array>& inputs,  std::vector<array>& outputs) {
    throw std::runtime_error("eval not implemented!");
}

#ifdef ACCELERATE_NEW_LAPACK
void Conv1dForward::eval_cpu(const std::vector<array>& inputs,  std::vector<array>& outputs) {
    throw std::runtime_error("eval_cpu not implemented!");
}
#endif

void Conv1dForward::eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) {

  assert(inputs.size() == 3);
  assert(outputs.size() == 1);

  auto x = inputs[0];       // (b, d, l)
  auto w = inputs[1];       // (d, k)
  auto b = inputs[2];       // (d)

  auto y = outputs[0];

  auto& s = stream();
  auto& d = metal::device(s.device);

  y.set_data(
    allocator::malloc_or_wait(x.data_size() * y.itemsize()),
    x.data_size(),
    x.strides(),
    x.flags()
  );

  std::ostringstream kname;
  kname << "conv1d_forward_kernel_";
  kname << type_to_name(x);
  
  d.register_library("mlx_ext", metal::get_colocated_mtllib_path);
  auto kernel = d.get_kernel(kname.str(), "mlx_ext");
  auto& compute_encoder = d.get_command_encoder(s.index);
  compute_encoder->setComputePipelineState(kernel);

  auto kernel_size = w.shape(1);
  auto batch_size = x.shape(0);
  auto n_channels = x.shape(1);
  auto seq_len = x.shape(2);

  compute_encoder.set_input_array(x, 0);
  compute_encoder.set_input_array(w, 1);
  compute_encoder.set_input_array(b, 2);
  compute_encoder.set_output_array(y, 3);
  compute_encoder->setBytes(x.strides().data(), 3 * sizeof(size_t), 4);
  compute_encoder->setBytes(&kernel_size, kernel_size * sizeof(int), 5);
  

  // https://developer.apple.com/documentation/metal/compute_passes/calculating_threadgroup_and_grid_sizes
  MTL::Size grid_dims = MTL::Size(batch_size, n_channels, seq_len);
  size_t width = kernel->threadExecutionWidth();
  size_t height = kernel->maxTotalThreadsPerThreadgroup() / width; 
  MTL::Size group_dims = MTL::Size(width, height, 1);
  compute_encoder->dispatchThreads(grid_dims, group_dims);
}

} // namespace mlx::core