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
using namespace nb::literals;

using namespace mlx::core;

NB_MODULE(_ext, m) {
    m.doc() = "Extension module for MLX";

    m.def(
        "conv1d_update_",
        &conv1d_update,
        "x"_a, "w"_a, "b"_a, "state"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        R"(
            Perform 1D convolution update.

            Args:
                x (array): Input tensor.
                w (array): Weights tensor.
                b (array): Bias tensor.
                state (array): State tensor.
                stream (optional): Stream or device (default: None).

            Returns:
                array: Updated convolution results.
        )"
    );

    m.def(
        "conv1d_forward_",
        &conv1d_forward,
        "x"_a, "w"_a, "b"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        R"(
            Perform 1D convolution forward pass.

            Args:
                x (array): Input tensor.
                w (array): Weights tensor.
                b (array): Bias tensor.
                stream (optional): Stream or device (default: None).

            Returns:
                array: Forward pass results.
        )"
    );

    m.def(
        "conv1d_swish_update_",
        &conv1d_swish_update,
        "x"_a, "w"_a, "b"_a, "state"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        R"(
            Perform 1D convolution with Swish activation update.

            Args:
                x (array): Input tensor.
                w (array): Weights tensor.
                b (array): Bias tensor.
                state (array): State tensor.
                stream (optional): Stream or device (default: None).

            Returns:
                array: Updated convolution results with Swish activation.
        )"
    );

    m.def(
        "conv1d_swish_forward_",
        &conv1d_swish_forward,
        "x"_a, "w"_a, "b"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        R"(
            Perform 1D convolution with Swish activation forward pass.

            Args:
                x (array): Input tensor.
                w (array): Weights tensor.
                b (array): Bias tensor.
                stream (optional): Stream or device (default: None).

            Returns:
                array: Forward pass results with Swish activation.
        )"
    );

    m.def(
        "ssm_update",
        &ssm_update,
        "x"_a, "dt"_a, "A"_a, "B"_a, "C"_a, "D"_a, "z"_a, "state"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        R"(
            Perform state-space model (SSM) update.

            Args:
                x (array): Input array.
                dt (array): Time step array.
                A (array): Matrix A.
                B (array): Matrix B.
                C (array): Matrix C.
                D (array): Matrix D.
                z (array): State variable.
                state (array): State tensor.
                stream (optional): Stream or device (default: None).

            Returns:
                array: Updated SSM state.
        )"
    );

    m.def(
        "ssd_update_",
        &ssd_update,
        "x"_a, "dt"_a, "decay"_a, "B"_a, "C"_a, "D"_a, "z"_a, "state"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        R"(
            Perform state-space dynamics (SSD) update.

            Args:
                x (array): Input array.
                dt (array): Time step array.
                decay (array): Decay parameter.
                B (array): Matrix B.
                C (array): Matrix C.
                D (array): Matrix D.
                z (array): State variable.
                state (array): State tensor.
                stream (optional): Stream or device (default: None).

            Returns:
                array: Updated SSD state.
        )"
    );

    m.def(
        "ssd_update_no_z_",
        &ssd_update_no_z,
        "x"_a, "dt"_a, "decay"_a, "B"_a, "C"_a, "D"_a, "state"_a,
        nb::kw_only(), "stream"_a = nb::none(),
        R"(
            Perform SSD update without state variable 'z'.

            Args:
                x (array): Input array.
                dt (array): Time step array.
                decay (array): Decay parameter.
                B (array): Matrix B.
                C (array): Matrix C.
                D (array): Matrix D.
                state (array): State tensor.
                stream (optional): Stream or device (default: None).

            Returns:
                array: Updated SSD state without 'z'.
        )"
    );
}
