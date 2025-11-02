#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "src/ssm_update.h"
#include "src/ssd_update.h"
#include "src/conv1d_forward.h"
#include "src/conv1d_update.h"
#include "src/conv1d_swish_forward.h"
#include "src/conv1d_swish_update.h"
#include "src/ssd_update_no_z.h"

namespace nb = nanobind;
using namespace nb::literals;

using namespace cartesia_opencl;

NB_MODULE(_ext, m) {
    m.doc() = "OpenCL Extension module for Cartesia State Space Models";

    // Initialize and cleanup functions
    m.def("initialize_opencl", &initialize_opencl, "Initialize OpenCL context");
    m.def("cleanup_opencl", &cleanup_opencl, "Cleanup OpenCL context");

    m.def(
        "ssm_update",
        &ssm_update,
        "x"_a, "dt"_a, "A"_a, "B"_a, "C"_a, "D"_a, "z"_a, "state"_a,
        R"(
            Perform state-space model (SSM) update using OpenCL.

            Args:
                x (array): Input array.
                dt (array): Time step array.
                A (array): Matrix A.
                B (array): Matrix B.
                C (array): Matrix C.
                D (array): Matrix D.
                z (array): State variable.
                state (array): State tensor.

            Returns:
                tuple: (y, next_state) - Output and updated state.
        )"
    );

    m.def(
        "ssd_update",
        &ssd_update,
        "x"_a, "dt"_a, "decay"_a, "B"_a, "C"_a, "D"_a, "z"_a, "state"_a,
        R"(
            Perform state-space dynamics (SSD) update using OpenCL.

            Args:
                x (array): Input array.
                dt (array): Time step array.
                decay (array): Decay parameter.
                B (array): Matrix B.
                C (array): Matrix C.
                D (array): Matrix D.
                z (array): State variable.
                state (array): State tensor.

            Returns:
                tuple: (y, next_state) - Output and updated state.
        )"
    );

    m.def(
        "ssd_update_no_z",
        &ssd_update_no_z,
        "x"_a, "dt"_a, "decay"_a, "B"_a, "C"_a, "D"_a, "state"_a,
        R"(
            Perform SSD update without state variable 'z' using OpenCL.

            Args:
                x (array): Input array.
                dt (array): Time step array.
                decay (array): Decay parameter.
                B (array): Matrix B.
                C (array): Matrix C.
                D (array): Matrix D.
                state (array): State tensor.

            Returns:
                tuple: (y, next_state) - Output and updated state.
        )"
    );

    m.def(
        "conv1d_forward",
        &conv1d_forward,
        "x"_a, "w"_a, "b"_a,
        R"(
            Perform 1D convolution forward pass using OpenCL.

            Args:
                x (array): Input tensor.
                w (array): Weights tensor.
                b (array): Bias tensor.

            Returns:
                array: Forward pass results.
        )"
    );

    m.def(
        "conv1d_update",
        &conv1d_update,
        "x"_a, "w"_a, "b"_a, "state"_a,
        R"(
            Perform 1D convolution update using OpenCL.

            Args:
                x (array): Input tensor.
                w (array): Weights tensor.
                b (array): Bias tensor.
                state (array): State tensor.

            Returns:
                tuple: (y, next_state) - Output and updated state.
        )"
    );

    m.def(
        "conv1d_swish_forward",
        &conv1d_swish_forward,
        "x"_a, "w"_a, "b"_a,
        R"(
            Perform 1D convolution with Swish activation forward pass using OpenCL.

            Args:
                x (array): Input tensor.
                w (array): Weights tensor.
                b (array): Bias tensor.

            Returns:
                array: Forward pass results with Swish activation.
        )"
    );

    m.def(
        "conv1d_swish_update",
        &conv1d_swish_update,
        "x"_a, "w"_a, "b"_a, "state"_a,
        R"(
            Perform 1D convolution with Swish activation update using OpenCL.

            Args:
                x (array): Input tensor.
                w (array): Weights tensor.
                b (array): Bias tensor.
                state (array): State tensor.

            Returns:
                tuple: (y, next_state) - Output and updated state with Swish activation.
        )"
    );

    // Device information functions
    m.def(
        "get_opencl_platforms",
        []() {
            // Implementation to get available OpenCL platforms
            return std::vector<std::string>{"OpenCL Platform"};
        },
        R"(
            Get available OpenCL platforms.

            Returns:
                list: List of platform names.
        )"
    );

    m.def(
        "get_opencl_devices",
        []() {
            // Implementation to get available OpenCL devices
            return std::vector<std::string>{"OpenCL Device"};
        },
        R"(
            Get available OpenCL devices.

            Returns:
                list: List of device names.
        )"
    );
}


