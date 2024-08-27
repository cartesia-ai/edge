#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::vector<array> conv1d_forward(
    const array& x,
    const array& w,
    const array& b,
    StreamOrDevice s = {} // Stream on which to schedule the operation
);

class Conv1dForward : public Primitive {
 public:
  explicit Conv1dForward(Stream stream)
      : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;

  void print(std::ostream& os) override {
    os << "Conv1dForward";
  }

  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);
};

} // namespace mlx::core