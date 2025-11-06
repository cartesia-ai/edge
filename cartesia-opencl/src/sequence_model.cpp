#include "sequence_model.h"
#include "opencl_context.h"
#include "layers/rms_norm_layer.h"
#include <stdexcept>
#include <CL/cl.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>

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

void SequenceModel::setPostNormWeights(const std::vector<float>& weights) {
    if (weights.size() != static_cast<size_t>(d_model_)) {
        throw std::runtime_error("Invalid post-norm weights size: expected " +
                               std::to_string(d_model_) + ", got " +
                               std::to_string(weights.size()));
    }
    post_norm_weights_ = weights;
}

cl_mem SequenceModel::forward(
    cl_mem input,
    int batch_size,
    int seq_len,
    std::vector<LayerState>* state,
    cl_command_queue queue,
    const std::string& output_prefix
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
        // Debug: Check input to this layer (first few layers only)
        if (i < 3) {
            size_t input_buf_size = 0;
            cl_int info_err = clGetMemObjectInfo(current, CL_MEM_SIZE, sizeof(size_t), &input_buf_size, nullptr);
            if (info_err == CL_SUCCESS && input_buf_size > 0) {
                std::vector<float> input_check(input_buf_size / sizeof(float));
                cl_int read_err = clEnqueueReadBuffer(queue, current, CL_TRUE, 0, input_buf_size, input_check.data(), 0, nullptr, nullptr);
                if (read_err == CL_SUCCESS && input_check.size() > 0) {
                    float input_min = input_check[0], input_max = input_check[0], input_sum = 0.0f;
                    for (float val : input_check) {
                        input_min = std::min(input_min, val);
                        input_max = std::max(input_max, val);
                        input_sum += val;
                    }
                    float input_mean = input_sum / input_check.size();
                    std::cout << "  [SeqModel] Layer " << i << " input stats: min=" << input_min 
                              << ", max=" << input_max << ", mean=" << input_mean << std::endl;
                }
            }
        }
        
        LayerState* layer_state = state ? &((*state)[i]) : nullptr;
        
        // Check input to layer 6 (first attention layer) for NaN
        static bool checked_layer6_input = false;
        if (!checked_layer6_input && i == 6 && current) {
            size_t buf_size = 0;
            cl_int info_err = clGetMemObjectInfo(current, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
            if (info_err == CL_SUCCESS && buf_size > 0) {
                std::vector<float> layer6_input(buf_size / sizeof(float));
                cl_int read_err = clEnqueueReadBuffer(queue, current, CL_TRUE, 0, buf_size, layer6_input.data(), 0, nullptr, nullptr);
                if (read_err == CL_SUCCESS) {
                    int nan_count = 0;
                    for (float val : layer6_input) {
                        if (std::isnan(val)) { nan_count++; }
                    }
                    std::cout << "\n  [Layer 6 Input Check] Before attention forward(): " << nan_count 
                              << " NaNs out of " << layer6_input.size() << " values" << std::endl;
                }
            }
            checked_layer6_input = true;
        }
        
        if (layers_[i]->isStateful()) {
            // Stateful layer returns (output, state)
            current = layers_[i]->forward(current, batch_size, seq_len, layer_state, queue);
        } else {
            // Stateless layer returns just output
            LayerState dummy_state = LayerState::null();
            current = layers_[i]->forward(current, batch_size, seq_len, &dummy_state, queue);
        }
        
        // Always dump layer outputs for comparison with MLX (if output_prefix is provided)
        if (!output_prefix.empty() && current) {
            size_t buf_size = 0;
            cl_int info_err = clGetMemObjectInfo(current, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
            if (info_err == CL_SUCCESS && buf_size > 0) {
                std::vector<float> layer_output(buf_size / sizeof(float));
                cl_int read_err = clEnqueueReadBuffer(queue, current, CL_TRUE, 0, buf_size, layer_output.data(), 0, nullptr, nullptr);
                if (read_err == CL_SUCCESS) {
                    // Calculate stats
                    float min_val = layer_output[0], max_val = layer_output[0], sum_val = 0.0f;
                    for (float val : layer_output) {
                        min_val = std::min(min_val, val);
                        max_val = std::max(max_val, val);
                        sum_val += val;
                    }
                    float mean_val = sum_val / layer_output.size();
                    
                    // Match MLX naming: {output_prefix}_layer_{i}_output_opencl.bin
                    std::stringstream ss;
                    ss << output_prefix << "_layer_" << i << "_output_opencl.bin";
                    std::ofstream out(ss.str(), std::ios::binary);
                    if (out.is_open()) {
                        out.write(reinterpret_cast<const char*>(layer_output.data()), layer_output.size() * sizeof(float));
                        out.close();
                        int d_model = buf_size / sizeof(float) / batch_size / seq_len;
                        std::cout << "\n  [OpenCL Debug] Dumped layer " << i << " output to " << ss.str() 
                                  << " (shape: (" << batch_size << ", " << seq_len << ", " << d_model << "))" << std::endl;
                    }
                }
            }
        }
        
        // NaN check for prefill (check first 6 layers to find where NaN originates)
        static bool checked_prefill = false;
        if (!checked_prefill && current && i < 6) {  // Only check layers 0-5
            size_t buf_size = 0;
            cl_int info_err = clGetMemObjectInfo(current, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
            if (info_err == CL_SUCCESS && buf_size > 0) {
                std::vector<float> layer_output(buf_size / sizeof(float));
                cl_int read_err = clEnqueueReadBuffer(queue, current, CL_TRUE, 0, buf_size, layer_output.data(), 0, nullptr, nullptr);
                if (read_err == CL_SUCCESS) {
                    int nan_count = 0;
                    for (float val : layer_output) {
                        if (std::isnan(val)) { nan_count++; }
                    }
                    // Always print for layers 0-5, even if no NaN (helps debugging)
                    std::cout << "\n  [Prefill NaN Check] Layer " << i << " output: " << nan_count 
                              << " NaNs out of " << layer_output.size() << " values" << std::endl;
                    if (i == 5) {
                        checked_prefill = true;  // Only set flag after checking layer 5
                        // Also check the buffer again right after layer 5 (before layer 6)
                        clFinish(queue);  // Ensure all writes are complete
                        std::vector<float> layer5_final(buf_size / sizeof(float));
                        cl_int final_read = clEnqueueReadBuffer(queue, current, CL_TRUE, 0, buf_size, layer5_final.data(), 0, nullptr, nullptr);
                        if (final_read == CL_SUCCESS) {
                            int nan_count_final = 0;
                            for (float val : layer5_final) {
                                if (std::isnan(val)) { nan_count_final++; }
                            }
                            std::cout << "  [After Layer 5 Check] Buffer after layer 5 complete: " << nan_count_final 
                                      << " NaNs out of " << layer5_final.size() << " values" << std::endl;
                        }
                    }
                }
            }
        }
    }
    
    // Check output before post-norm
    if (!output_prefix.empty()) {
        size_t buf_size = 0;
        cl_int info_err = clGetMemObjectInfo(current, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
        if (info_err == CL_SUCCESS && buf_size > 0) {
            std::vector<float> before_postnorm(buf_size / sizeof(float));
            cl_int read_err = clEnqueueReadBuffer(queue, current, CL_TRUE, 0, buf_size, before_postnorm.data(), 0, nullptr, nullptr);
            if (read_err == CL_SUCCESS) {
                int nan_count = 0;
                for (float val : before_postnorm) {
                    if (std::isnan(val)) { nan_count++; }
                }
                std::cout << "\n  [Before Post-Norm] NaN count: " << nan_count << " out of " << before_postnorm.size() << " values" << std::endl;
            }
        }
    }
    
    // Apply post-norm if needed
    if (use_post_norm_ && post_norm_) {
        std::cout << "\n  [Post-Norm] Applying post-norm..." << std::flush;
        if (!norm_layer_) {
            // Initialize norm layer if not already done
            norm_layer_ = std::make_unique<RMSNormLayer>(ctx_, d_model_);
            // Use actual post-norm weights if provided, otherwise default to all ones
            std::vector<float> norm_weights = post_norm_weights_.empty() 
                ? std::vector<float>(d_model_, 1.0f) 
                : post_norm_weights_;
            norm_layer_->initializeWeights(norm_weights);
        }
        current = norm_layer_->forward(current, batch_size, seq_len, queue);
        std::cout << " ✓" << std::endl;
        
        // Check output after post-norm
        size_t buf_size = 0;
        cl_int info_err = clGetMemObjectInfo(current, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
        if (info_err == CL_SUCCESS && buf_size > 0) {
            std::vector<float> after_postnorm(buf_size / sizeof(float));
            cl_int read_err = clEnqueueReadBuffer(queue, current, CL_TRUE, 0, buf_size, after_postnorm.data(), 0, nullptr, nullptr);
            if (read_err == CL_SUCCESS) {
                int nan_count = 0;
                for (float val : after_postnorm) {
                    if (std::isnan(val)) { nan_count++; }
                }
                std::cout << "  [After Post-Norm] NaN count: " << nan_count << " out of " << after_postnorm.size() << " values" << std::endl;
            }
        }
    } else {
        std::cout << "\n  [Post-Norm] Post-norm is disabled (use_post_norm_=" << use_post_norm_ << ", post_norm_=" << (post_norm_ ? "true" : "false") << ")" << std::endl;
    }
    
    return current;
}

cl_mem SequenceModel::step(
    cl_mem input,
    int batch_size,
    std::vector<LayerState>* state,
    cl_command_queue queue,
    const std::string& output_prefix
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
            
            // Add more detailed logging for layer 11 (last layer)
            if (i == 11) {
                std::cout << " [calling step]..." << std::flush;
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
            
            // Add more detailed logging for layer 11
            if (i == 11) {
                std::cout << " [validating output]..." << std::flush;
            }
            
            // Validate output buffer
            cl_int info_err;
            size_t buf_size = 0;
            info_err = clGetMemObjectInfo(current, CL_MEM_SIZE, sizeof(size_t), &buf_size, nullptr);
            if (info_err != CL_SUCCESS) {
                throw std::runtime_error("Layer " + std::to_string(i) + " returned invalid buffer (err=" + std::to_string(info_err) + ")");
            }
            
            // Dump layer outputs for comparison with MLX (if output_prefix is provided)
            if (!output_prefix.empty() && current) {
                std::vector<float> layer_output(buf_size / sizeof(float));
                cl_int read_err = clEnqueueReadBuffer(queue, current, CL_TRUE, 0, buf_size, layer_output.data(), 0, nullptr, nullptr);
                if (read_err == CL_SUCCESS) {
                    // Calculate stats
                    float min_val = layer_output[0], max_val = layer_output[0], sum_val = 0.0f;
                    for (float val : layer_output) {
                        min_val = std::min(min_val, val);
                        max_val = std::max(max_val, val);
                        sum_val += val;
                    }
                    float mean_val = sum_val / layer_output.size();
                    
                    // Match MLX naming: {output_prefix}_gen_step_0_layer_{i}_output_opencl.bin
                    std::stringstream ss;
                    ss << output_prefix << "_gen_step_0_layer_" << i << "_output_opencl.bin";
                    std::ofstream out(ss.str(), std::ios::binary);
                    if (out.is_open()) {
                        out.write(reinterpret_cast<const char*>(layer_output.data()), layer_output.size() * sizeof(float));
                        out.close();
                        int d_model = buf_size / sizeof(float) / batch_size;
                        std::cout << "\n  [OpenCL Debug] Dumped gen step 0 layer " << i << " output to " << ss.str()
                                  << " (shape: (" << batch_size << ", " << d_model << "))" << std::endl;
                    }
                }
            }
            
            // NaN/Inf check for first generation step only
            static bool nan_check_done = false;
            if (!nan_check_done && buf_size > 0) {
                std::vector<float> layer_output(buf_size / sizeof(float));
                cl_int read_err = clEnqueueReadBuffer(queue, current, CL_TRUE, 0, buf_size, layer_output.data(), 0, nullptr, nullptr);
                if (read_err == CL_SUCCESS) {
                    int nan_count = 0;
                    int inf_count = 0;
                    for (float val : layer_output) {
                        if (std::isnan(val)) { nan_count++; }
                        if (std::isinf(val)) { inf_count++; }
                        if (nan_count > 0 && inf_count > 0) break;  // Early exit if both found
                    }
                    if (nan_count > 0 || inf_count > 0) {
                        std::cout << " [Layer " << i << " output: " << nan_count << " NaNs, " << inf_count << " Infs!]";
                        nan_check_done = true;  // Stop checking after first NaN/Inf found
                    }
                }
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
                // Use actual post-norm weights if provided, otherwise default to all ones
                std::vector<float> norm_weights = post_norm_weights_.empty() 
                    ? std::vector<float>(d_model_, 1.0f) 
                    : post_norm_weights_;
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

