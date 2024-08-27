#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

namespace mlx::core {

std::vector<array> conv1d_update(
    const array& x,
    const array& w,
    const array& b,
    const array& state,
    StreamOrDevice s = {} // Stream on which to schedule the operation
);

class Conv1dUpdate : public Primitive {
 public:
  explicit Conv1dUpdate(Stream stream)
      : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;

  void print(std::ostream& os) override {
    os << "Conv1dUpdate";
  }

  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);
};

} // namespace mlx::core