// Copyright © 2023-2024 Cartesia AI

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "src/conv1d_update.h"
#include "src/conv1d_forward.h"
#include "src/conv1d_swish_update.h"
#include "src/conv1d_swish_forward.h"
#include "src/ssm_update.h"
#include "src/ssd_update.h"
#include "src/ssd_update_no_z.h"

namespace nb = nanobind;
using namespace nb::literals;

using namespace mlx::core;

NB_MODULE(_ext, m) {
    m.doc() = "Cartesia Metal extension for MLX";

    m.def("conv1d_update_", &conv1d_update,
          "x"_a, "w"_a, "b"_a, "state"_a,
          nb::kw_only(), "stream"_a = nb::none(),
          R"(
            Perform 1D convolution update.

            Args:
                x (array): Input array.
                w (array): Weight array.
                b (array): Bias array.
                state (array): State array.
                stream (optional): Stream on which to schedule the operation.

            Returns:
                array: Result of the 1D convolution update.
          )");

    m.def("conv1d_forward_", &conv1d_forward,
          "x"_a, "w"_a, "b"_a,
          nb::kw_only(), "stream"_a = nb::none(),
          R"(
            Perform 1D convolution forward pass.

            Args:
                x (array): Input array.
                w (array): Weight array.
                b (array): Bias array.
                stream (optional): Stream on which to schedule the operation.

            Returns:
                array: Result of the 1D convolution forward pass.
          )");

    m.def("conv1d_swish_update_", &conv1d_swish_update,
          "x"_a, "w"_a, "b"_a, "state"_a,
          nb::kw_only(), "stream"_a = nb::none(),
          R"(
            Perform 1D convolution with Swish activation update.

            Args:
                x (array): Input array.
                w (array): Weight array.
                b (array): Bias array.
                state (array): State array.
                stream (optional): Stream on which to schedule the operation.

            Returns:
                array: Result of the 1D convolution with Swish activation update.
          )");

    m.def("conv1d_swish_forward_", &conv1d_swish_forward,
          "x"_a, "w"_a, "b"_a,
          nb::kw_only(), "stream"_a = nb::none(),
          R"(
            Perform 1D convolution with Swish activation forward pass.

            Args:
                x (array): Input array.
                w (array): Weight array.
                b (array): Bias array.
                stream (optional): Stream on which to schedule the operation.

            Returns:
                array: Result of the 1D convolution with Swish activation forward pass.
          )");

    m.def("ssm_update", &ssm_update,
          "x"_a, "dt"_a, "A"_a, "B"_a, "C"_a, "D"_a, "z"_a, "state"_a,
          nb::kw_only(), "stream"_a = nb::none(),
          R"(
            Perform State Space Model (SSM) update.

            Args:
                x (array): Input array.
                dt (array): Time step array.
                A (array): State transition matrix.
                B (array): Input matrix.
                C (array): Output matrix.
                D (array): Feedthrough matrix.
                z (array): Input modulation array.
                state (array): State array.
                stream (optional): Stream on which to schedule the operation.

            Returns:
                array: Result of the SSM update.
          )");

    m.def("ssd_update_", &ssd_update,
          "x"_a, "dt"_a, "decay"_a, "B"_a, "C"_a, "D"_a, "z"_a, "state"_a,
          nb::kw_only(), "stream"_a = nb::none(),
          R"(
            Perform State Space Decay (SSD) update.

            Args:
                x (array): Input array.
                dt (array): Time step array.
                decay (array): Decay array.
                B (array): Input matrix.
                C (array): Output matrix.
                D (array): Feedthrough matrix.
                z (array): Input modulation array.
                state (array): State array.
                stream (optional): Stream on which to schedule the operation.

            Returns:
                array: Result of the SSD update.
          )");

    m.def("ssd_update_no_z_", &ssd_update_no_z,
          "x"_a, "dt"_a, "decay"_a, "B"_a, "C"_a, "D"_a, "state"_a,
          nb::kw_only(), "stream"_a = nb::none(),
          R"(
            Perform State Space Decay (SSD) update without input modulation.

            Args:
                x (array): Input array.
                dt (array): Time step array.
                decay (array): Decay array.
                B (array): Input matrix.
                C (array): Output matrix.
                D (array): Feedthrough matrix.
                state (array): State array.
                stream (optional): Stream on which to schedule the operation.

            Returns:
                array: Result of the SSD update without input modulation.
          )");
}
