#include "residual_block.h"
#include "opencl_context.h"
#include "layers/rms_norm_layer.h"
#include <stdexcept>
#include <CL/cl.h>
#include <cstring>
#include <vector>

namespace cartesia_opencl {

ResidualBlock::ResidualBlock(
    OpenCLContextManager* ctx,
    Layer* layer,
    int d_model,
    const std::string& norm_point,
    bool stateful
)
    : ctx_(ctx)
    , layer_(layer)
    , d_model_(d_model)
    , norm_point_(norm_point)
    , stateful_(stateful)
{
    if (!ctx_ || !layer_) {
        throw std::runtime_error("Invalid ResidualBlock parameters");
    }
    
    // Initialize norm layer if needed
    if (!norm_point_.empty()) {
        norm_layer_ = std::make_unique<RMSNormLayer>(ctx, d_model);
        // Initialize with ones (will be set from weights later if needed)
        norm_weights_.resize(d_model, 1.0f);
        norm_layer_->initializeWeights(norm_weights_);
    }
}

ResidualBlock::~ResidualBlock() {
    // norm_layer_ will clean itself up
    // Note: We don't delete layer_ - it's managed externally
}

cl_mem ResidualBlock::forward(
    cl_mem input,
    int batch_size,
    int seq_len,
    LayerState* state,
    cl_command_queue queue
) {
    cl_context context = ctx_->getContext();
    cl_mem residual = input;  // Save for residual connection
    
    // Pre-norm
    if (!norm_point_.empty() && norm_point_ == "pre") {
        input = applyNorm(input, batch_size, seq_len, queue);
    }
    
    // Apply layer
    cl_mem output;
    if (stateful_) {
        output = layer_->forward(input, batch_size, seq_len, state, queue);
    } else {
        LayerState dummy_state = LayerState::null();
        output = layer_->forward(input, batch_size, seq_len, &dummy_state, queue);
    }
    
    // Pre-resid norm
    if (!norm_point_.empty() && norm_point_ == "pre_resid") {
        output = applyNorm(output, batch_size, seq_len, queue);
    }
    
    // Residual connection: output = output + residual
    // Allocate temporary buffer for residual addition
    size_t output_size = batch_size * seq_len * d_model_ * sizeof(float);
    
    // Read output and residual, add them, write back
    // For now, we'll use a simple CPU fallback (TODO: implement OpenCL kernel)
    std::vector<float> output_cpu(batch_size * seq_len * d_model_);
    std::vector<float> residual_cpu(batch_size * seq_len * d_model_);
    
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, output_size, output_cpu.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, residual, CL_TRUE, 0, output_size, residual_cpu.data(), 0, nullptr, nullptr);
    
    for (size_t i = 0; i < output_cpu.size(); ++i) {
        output_cpu[i] += residual_cpu[i];
    }
    
    clEnqueueWriteBuffer(queue, output, CL_TRUE, 0, output_size, output_cpu.data(), 0, nullptr, nullptr);
    
    // Post-norm
    if (!norm_point_.empty() && norm_point_ == "post") {
        output = applyNorm(output, batch_size, seq_len, queue);
    }
    
    return output;
}

cl_mem ResidualBlock::step(
    cl_mem input,
    int batch_size,
    LayerState* state,
    cl_command_queue queue
) {
    cl_mem residual = input;  // Save for residual connection
    
    // Pre-norm
    if (!norm_point_.empty() && norm_point_ == "pre") {
        input = applyNormStep(input, batch_size, queue);
    }
    
    // Apply layer
    cl_mem output;
    if (stateful_) {
        output = layer_->step(input, batch_size, state, queue);
    } else {
        LayerState dummy_state = LayerState::null();
        output = layer_->step(input, batch_size, &dummy_state, queue);
    }
    
    // Pre-resid norm
    if (!norm_point_.empty() && norm_point_ == "pre_resid") {
        output = applyNormStep(output, batch_size, queue);
    }
    
    // Residual connection
    size_t output_size = batch_size * d_model_ * sizeof(float);
    
    // CPU fallback for residual addition (TODO: OpenCL kernel)
    std::vector<float> output_cpu(batch_size * d_model_);
    std::vector<float> residual_cpu(batch_size * d_model_);
    
    clEnqueueReadBuffer(queue, output, CL_TRUE, 0, output_size, output_cpu.data(), 0, nullptr, nullptr);
    clEnqueueReadBuffer(queue, residual, CL_TRUE, 0, output_size, residual_cpu.data(), 0, nullptr, nullptr);
    
    for (size_t i = 0; i < output_cpu.size(); ++i) {
        output_cpu[i] += residual_cpu[i];
    }
    
    clEnqueueWriteBuffer(queue, output, CL_TRUE, 0, output_size, output_cpu.data(), 0, nullptr, nullptr);
    
    // Post-norm
    if (!norm_point_.empty() && norm_point_ == "post") {
        output = applyNormStep(output, batch_size, queue);
    }
    
    return output;
}

cl_mem ResidualBlock::applyNorm(cl_mem input, int batch_size, int seq_len, cl_command_queue queue) {
    if (!norm_layer_) {
        return input;  // No normalization
    }
    return norm_layer_->forward(input, batch_size, seq_len, queue);
}

cl_mem ResidualBlock::applyNormStep(cl_mem input, int batch_size, cl_command_queue queue) {
    if (!norm_layer_) {
        return input;  // No normalization
    }
    return norm_layer_->step(input, batch_size, queue);
}

} // namespace cartesia_opencl

