#include "sequence_model.h"
#include "opencl_context.h"
#include "layers/rms_norm_layer.h"
#include <stdexcept>
#include <CL/cl.h>
#include <iostream>

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
        try {
            std::cout << "    [SeqModel] step layer " << i << "..." << std::flush;
            
            // Validate input buffer before layer
            if (!current) {
                throw std::runtime_error("Layer " + std::to_string(i) + " received null input buffer");
            }
            
            LayerState* layer_state = &((*state)[i]);
            
            // Validate state buffers for stateful layers
            if (layers_[i]->isStateful() && layer_state && !layer_state->is_null()) {
                if (layer_state->state1) {
                    // Quick validation - check if buffer is valid by trying to get info (non-destructive)
                    cl_int info_err;
                    size_t buf_size = 0;
                    info_err = clGetMemObjectInfo(layer_state->state1, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
                    if (info_err != CL_SUCCESS) {
                        throw std::runtime_error("Layer " + std::to_string(i) + " has invalid state1 buffer (err=" + std::to_string(info_err) + ")");
                    }
                }
            }
            
            if (layers_[i]->isStateful()) {
                current = layers_[i]->step(current, batch_size, layer_state, queue);
                if (!current) {
                    throw std::runtime_error("Layer " + std::to_string(i) + " returned null buffer");
                }
            } else {
                LayerState dummy_state = LayerState::null();
                current = layers_[i]->step(current, batch_size, &dummy_state, queue);
                if (!current) {
                    throw std::runtime_error("Layer " + std::to_string(i) + " returned null buffer");
                }
            }
            
            // Validate output buffer
            cl_int info_err;
            size_t buf_size = 0;
            info_err = clGetMemObjectInfo(current, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
            if (info_err != CL_SUCCESS) {
                throw std::runtime_error("Layer " + std::to_string(i) + " returned invalid buffer (err=" + std::to_string(info_err) + ")");
            }
            
            std::cout << " ✓" << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "\n    [SeqModel] ERROR in layer " << i << ": " << e.what() << std::endl;
            throw;
        } catch (...) {
            std::cerr << "\n    [SeqModel] FATAL: Unknown exception in layer " << i << std::endl;
            throw;
        }
    }
    
    std::cout << "    [SeqModel] All layers complete, checking post-norm..." << std::flush;
    
    // Apply post-norm if needed
    if (use_post_norm_ && post_norm_) {
        try {
            if (!norm_layer_) {
                std::cout << "\n      [SeqModel] Initializing post-norm layer..." << std::flush;
                // Initialize norm layer if not already done
                norm_layer_ = std::make_unique<RMSNormLayer>(ctx_, d_model_);
                std::vector<float> norm_weights(d_model_, 1.0f);
                norm_layer_->initializeWeights(norm_weights);
                std::cout << " ✓" << std::flush;
            }
            std::cout << "\n      [SeqModel] Running post-norm step..." << std::flush;
            cl_mem norm_output = norm_layer_->step(current, batch_size, queue);
            if (!norm_output) {
                throw std::runtime_error("Post-norm step returned null buffer");
            }
            current = norm_output;
            std::cout << " ✓" << std::flush;
        } catch (const std::exception& e) {
            std::cerr << "\n    [SeqModel] ERROR in post-norm: " << e.what() << std::endl;
            throw;
        }
    }
    
    std::cout << "\n    [SeqModel] step complete, returning buffer" << std::endl;
    
    if (!current) {
        throw std::runtime_error("SequenceModel::step returning null buffer");
    }
    
    return current;
}

// Post-norm is now handled inline in forward() and step()

} // namespace cartesia_opencl

