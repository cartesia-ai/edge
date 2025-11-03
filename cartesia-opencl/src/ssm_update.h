#pragma once

#include <vector>

namespace cartesia_opencl {

// Initialize and cleanup OpenCL context
void initialize_opencl();
void cleanup_opencl();

// SSM update function that works with raw vectors
// This is the standalone version for command-line usage
void ssm_update_standalone(
    const std::vector<float>& x,
    const std::vector<float>& dt,
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C,
    const std::vector<float>& D,
    const std::vector<float>& z,
    const std::vector<float>& state,
    std::vector<float>& y,
    std::vector<float>& next_state,
    int batch_size,
    int channel_size,
    int state_size
);

} // namespace cartesia_opencl

