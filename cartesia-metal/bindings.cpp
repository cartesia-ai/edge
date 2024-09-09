#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "src/conv1d_update.h"
#include "src/conv1d_forward.h"
#include "src/conv1d_swish_update.h"
#include "src/conv1d_swish_forward.h"
#include "src/ssm_update.h"
#include "src/ssd_update.h"
#include "src/ssd_update_no_z.h"

namespace nb = nanobind;

using namespace mlx::core;

NB_MODULE(_ext, m) {
    m.def("conv1d_update_", &conv1d_update, 
          nb::arg("x"), nb::arg("w"), nb::arg("b"), nb::arg("state"), 
          nb::kw_only(), nb::arg("stream") = nb::none());

    m.def("conv1d_forward_", &conv1d_forward, 
          nb::arg("x"), nb::arg("w"), nb::arg("b"), 
          nb::kw_only(), nb::arg("stream") = nb::none());

    m.def("conv1d_swish_update_", &conv1d_swish_update, 
          nb::arg("x"), nb::arg("w"), nb::arg("b"), nb::arg("state"), 
          nb::kw_only(), nb::arg("stream") = nb::none());

    m.def("conv1d_swish_forward_", &conv1d_swish_forward, 
          nb::arg("x"), nb::arg("w"), nb::arg("b"), 
          nb::kw_only(), nb::arg("stream") = nb::none());

    m.def("ssm_update", &ssm_update, 
          nb::arg("x"), nb::arg("dt"), nb::arg("A"), nb::arg("B"), nb::arg("C"), nb::arg("D"), nb::arg("z"), nb::arg("state"), 
          nb::kw_only(), nb::arg("stream") = nb::none());

    m.def("ssd_update_", &ssd_update, 
          nb::arg("x"), nb::arg("dt"), nb::arg("decay"), nb::arg("B"), nb::arg("C"), nb::arg("D"), nb::arg("z"), nb::arg("state"), 
          nb::kw_only(), nb::arg("stream") = nb::none());

    m.def("ssd_update_no_z_", &ssd_update_no_z, 
          nb::arg("x"), nb::arg("dt"), nb::arg("decay"), nb::arg("B"), nb::arg("C"), nb::arg("D"), nb::arg("state"), 
          nb::kw_only(), nb::arg("stream") = nb::none());
}
