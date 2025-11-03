// Element-wise operations

// Element-wise addition: output = a + b
__kernel void add(
    __global const float* a,
    __global const float* b,
    __global float* output,
    const int size
) {
    const int idx = get_global_id(0);
    if (idx >= size) return;
    output[idx] = a[idx] + b[idx];
}

