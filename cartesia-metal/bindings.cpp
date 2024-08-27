#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "src/conv1d_update.h"
#include "src/conv1d_forward.h"
#include "src/ssm_update.h"
#include "src/ssd_update.h"
#include "src/ssd_update_no_z.h"

namespace nb = nanobind;
using namespace nb::literals;

using namespace mlx::core;

NB_MODULE(_ext, m) {

  m.def(
      "conv1d_update",
      [](const array& x, const array& w, const array& b, const array& state, StreamOrDevice s) {
        return conv1d_update(x, w, b, state, s);
      },
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none()
  );

  m.def(
      "conv1d_forward_",
      [](const array& x, const array& w, const array& b, StreamOrDevice s) {
        return conv1d_forward(x, w, b, s);
      },
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none()
  );

  m.def(
      "ssm_update",
      [](const array& x, const array& dt, const array& A, const array& B, const array& C, const array& D, const array& z, const array& state, StreamOrDevice s) {
        return ssm_update(x, dt, A, B, C, D, z, state, s);
      },
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none()
  );

  m.def(
      "ssd_update_",
      [](const array& x, const array& dt, const array& decay, const array& B, const array& C, const array& D, const array& z, const array& state, StreamOrDevice s) {
        return ssd_update(x, dt, decay, B, C, D, z, state, s);
      },
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none()
  );

  m.def(
      "ssd_update_no_z_",
      [](const array& x, const array& dt, const array& decay, const array& B, const array& C, const array& D, const array& state, StreamOrDevice s) {
        return ssd_update_no_z(x, dt, decay, B, C, D, state, s);
      },
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::arg(),
      nb::kw_only(),
      "stream"_a = nb::none()
  );
}
