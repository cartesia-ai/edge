// Swish (SiLU) activation function: x * sigmoid(x)
__kernel void swish(
    __global const float* input,
    __global float* output,
    const int size
) {
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    float x = input[idx];
    float sigmoid_x = 1.0f / (1.0f + exp(-x));
    output[idx] = x * sigmoid_x;
}

// SwiGLU: (Swish(gate) * up) 
// Note: This is typically called after separate linear projections
__kernel void swiglu_combine(
    __global const float* gate,      // [size] - already Swish activated
    __global const float* up,         // [size]
    __global float* output,           // [size]
    const int size
) {
    const int idx = get_global_id(0);
    if (idx >= size) return;
    
    output[idx] = gate[idx] * up[idx];
}

