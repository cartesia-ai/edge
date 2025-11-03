#include "sequence_model.h"
#include "opencl_context.h"
#include "layers/rms_norm_layer.h"
#include <stdexcept>
#include <CL/cl.h>

namespace cartesia_opencl {

SequenceModel::SequenceModel(OpenCLContextManager* ctx, int d_model, int n_layer_repeats, bool post_norm)
    : ctx_(ctx)
    , d_model_(d_model)
    , post_norm_(post_norm)
    , norm_layer_(nullptr)
    , use_post_norm_(post_norm)
{
    if (!ctx_) {
        throw std::runtime_error("OpenCLContext is null");
    }
}

SequenceModel::~SequenceModel() {
    layers_.clear();  // This will destroy all layer objects
    // norm_layer_ will clean itself up
}

void SequenceModel::addLayer(std::unique_ptr<ResidualBlock> layer) {
    layers_.push_back(std::move(layer));
}

cl_mem SequenceModel::forward(
    cl_mem input,
    int batch_size,
    int seq_len,
    std::vector<LayerState>* state,
    cl_command_queue queue
) {
    // TODO: Implement full forward pass through all layers
    // For now, just return input (identity)
    
    if (state) {
        state->clear();
        state->resize(layers_.size(), LayerState::null());
    }
    
    cl_mem current = input;
    
    // Process through each layer
    for (size_t i = 0; i < layers_.size(); ++i) {
        LayerState* layer_state = state ? &((*state)[i]) : nullptr;
        
        if (layers_[i]->isStateful()) {
            // Stateful layer returns (output, state)
            current = layers_[i]->forward(current, batch_size, seq_len, layer_state, queue);
        } else {
            // Stateless layer returns just output
            LayerState dummy_state = LayerState::null();
            current = layers_[i]->forward(current, batch_size, seq_len, &dummy_state, queue);
        }
    }
    
    // Apply post-norm if needed
    if (use_post_norm_ && post_norm_) {
        if (!norm_layer_) {
            // Initialize norm layer if not already done
            norm_layer_ = std::make_unique<RMSNormLayer>(ctx_, d_model_);
            std::vector<float> norm_weights(d_model_, 1.0f);
            norm_layer_->initializeWeights(norm_weights);
        }
        current = norm_layer_->forward(current, batch_size, seq_len, queue);
    }
    
    return current;
}

cl_mem SequenceModel::step(
    cl_mem input,
    int batch_size,
    std::vector<LayerState>* state,
    cl_command_queue queue
) {
    // TODO: Implement full step function through all layers
    // For now, just return input (identity)
    
    if (!state || state->size() != layers_.size()) {
        throw std::runtime_error("Invalid state vector size");
    }
    
    cl_mem current = input;
    
    // Process through each layer
    for (size_t i = 0; i < layers_.size(); ++i) {
        LayerState* layer_state = &((*state)[i]);
        
        if (layers_[i]->isStateful()) {
            current = layers_[i]->step(current, batch_size, layer_state, queue);
        } else {
            LayerState dummy_state = LayerState::null();
            current = layers_[i]->step(current, batch_size, &dummy_state, queue);
        }
    }
    
    // Apply post-norm if needed
    if (use_post_norm_ && post_norm_) {
        if (!norm_layer_) {
            // Initialize norm layer if not already done
            norm_layer_ = std::make_unique<RMSNormLayer>(ctx_, d_model_);
            std::vector<float> norm_weights(d_model_, 1.0f);
            norm_layer_->initializeWeights(norm_weights);
        }
        current = norm_layer_->step(current, batch_size, queue);
    }
    
    return current;
}

// Post-norm is now handled inline in forward() and step()

} // namespace cartesia_opencl

