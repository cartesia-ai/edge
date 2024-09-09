#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::vector<array> conv1d_swish_update(
    const array& x,
    const array& w,
    const array& b,
    const array& state,
    StreamOrDevice s = {} // Stream on which to schedule the operation
);

class Conv1dSwishUpdate : public Primitive {
 public:
  explicit Conv1dSwishUpdate(Stream stream)
      : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;

  void print(std::ostream& os) override {
    os << "Conv1dSwishUpdate";
  }

  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);
};

} // namespace mlx::core