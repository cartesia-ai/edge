// RMS (Root Mean Square) Normalization kernel
// Normalizes input along the last dimension

__kernel void rms_norm(
    __global const float* input,      // [batch_size, seq_len, d_model] or [batch_size, d_model]
    __global const float* weight,     // [d_model] - scale weights
    __global float* output,           // Same shape as input
    const int d_model,               // Hidden dimension
    const int total_elements,        // Total elements in input (batch_size * seq_len * d_model or batch_size * d_model)
    const float eps                  // Epsilon for numerical stability
) {
    const int idx = get_global_id(0);  // Index of element
    
    if (idx >= total_elements) return;
    
    // Calculate which sequence element this is
    const int seq_idx = idx / d_model;
    const int feat_idx = idx % d_model;
    
    // Calculate mean square within this sequence element
    float mean_square = 0.0f;
    for (int i = 0; i < d_model; ++i) {
        float val = input[seq_idx * d_model + i];
        mean_square += val * val;
    }
    mean_square /= d_model;
    
    // RMS = sqrt(mean_square + eps)
    float rms = sqrt(mean_square + eps);
    
    // Normalize: output = (input / rms) * weight
    output[idx] = (input[idx] / rms) * weight[feat_idx];
}

