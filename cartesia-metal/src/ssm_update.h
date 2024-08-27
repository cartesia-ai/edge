#pragma once

#include "mlx/ops.h"
#include "mlx/primitives.h"

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
    StreamOrDevice s = {} // Stream on which to schedule the operation
);

class SSMUpdate : public Primitive {
 public:
  explicit SSMUpdate(Stream stream)
      : Primitive(stream){};

  void eval_cpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;
  void eval_gpu(const std::vector<array>& inputs, std::vector<array>& outputs) override;

  void print(std::ostream& os) override {
    os << "SSMUpdate";
  }

  void eval(const std::vector<array>& inputs, std::vector<array>& outputs);
};

} // namespace mlx::core