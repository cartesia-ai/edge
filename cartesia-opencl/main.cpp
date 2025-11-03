#include <iostream>
#include <vector>
#include <iomanip>
#include <random>
#include "src/ssm_update.h"

using namespace cartesia_opencl;

int main(int argc, char* argv[]) {
    std::cout << "========================================" << std::endl;
    std::cout << "Cartesia OpenCL SSM Update Test" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    try {
        // Initialize OpenCL
        std::cout << "Initializing OpenCL..." << std::endl;
        initialize_opencl();
        std::cout << "✓ OpenCL initialized successfully!" << std::endl;
        std::cout << std::endl;

        // Test parameters
        int batch_size = 2;
        int channel_size = 4;
        int state_size = 8;

        std::cout << "Test Configuration:" << std::endl;
        std::cout << "  Batch size: " << batch_size << std::endl;
        std::cout << "  Channel size: " << channel_size << std::endl;
        std::cout << "  State size: " << state_size << std::endl;
        std::cout << std::endl;

        // Initialize test data with small random values
        std::mt19937 gen(42); // Fixed seed for reproducibility
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        // Input data: x, dt, z
        std::vector<float> x(batch_size * channel_size);
        std::vector<float> dt(batch_size * channel_size);
        std::vector<float> z(batch_size * channel_size);

        // State space matrices: A, B, C, D
        std::vector<float> A(state_size);
        std::vector<float> B(batch_size * state_size);
        std::vector<float> C(batch_size * state_size);
        std::vector<float> D(channel_size);

        // Initial state
        std::vector<float> state(batch_size * channel_size * state_size);

        // Initialize with small random values
        for (float& val : x) val = dist(gen);
        for (float& val : dt) val = dist(gen) * 0.5f + 0.5f; // Positive values
        for (float& val : z) val = dist(gen);
        for (float& val : A) val = dist(gen) * 0.1f; // Small values for stability
        for (float& val : B) val = dist(gen) * 0.1f;
        for (float& val : C) val = dist(gen) * 0.1f;
        for (float& val : D) val = dist(gen) * 0.1f;
        for (float& val : state) val = dist(gen) * 0.1f;

        // Output vectors
        std::vector<float> y(batch_size * channel_size);
        std::vector<float> next_state(batch_size * channel_size * state_size);

        std::cout << "Running SSM update..." << std::endl;

        // Run SSM update
        ssm_update_standalone(
            x, dt, A, B, C, D, z, state,
            y, next_state,
            batch_size, channel_size, state_size
        );

        std::cout << "✓ SSM update completed successfully!" << std::endl;
        std::cout << std::endl;

        // Print some results
        std::cout << "Output (y) - first few values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(8), y.size()); ++i) {
            std::cout << "  y[" << i << "] = " << std::setprecision(6) << y[i] << std::endl;
        }
        std::cout << std::endl;

        std::cout << "Next state - first few values:" << std::endl;
        for (size_t i = 0; i < std::min(static_cast<size_t>(8), next_state.size()); ++i) {
            std::cout << "  next_state[" << i << "] = " << std::setprecision(6) << next_state[i] << std::endl;
        }
        std::cout << std::endl;

        // Cleanup
        cleanup_opencl();
        std::cout << "✓ Cleanup completed" << std::endl;
        std::cout << std::endl;

        std::cout << "========================================" << std::endl;
        std::cout << "Test completed successfully!" << std::endl;
        std::cout << "========================================" << std::endl;

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        cleanup_opencl();
        return 1;
    }
}

