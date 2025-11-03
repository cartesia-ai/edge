// Matrix multiplication: C = A * B
// A: [m, k], B: [k, n], C: [m, n]
// For batched: A: [batch, m, k], B: [k, n], C: [batch, m, n]

__kernel void matmul(
    __global const float* A,          // Input matrix [m, k] or [batch, m, k]
    __global const float* B,          // Weight matrix [k, n]
    __global float* C,                // Output matrix [m, n] or [batch, m, n]
    const int m,                      // Rows in A / output
    const int k,                      // Columns in A, rows in B
    const int n,                      // Columns in B / output
    const int batch_size              // Batch size (1 if not batched)
) {
    const int batch_idx = get_global_id(0) / m;
    const int row = get_global_id(0) % m;
    const int col = get_global_id(1);
    
    if (batch_idx >= batch_size || row >= m || col >= n) return;
    
    float sum = 0.0f;
    for (int i = 0; i < k; ++i) {
        int a_idx = batch_idx * m * k + row * k + i;
        int b_idx = i * n + col;
        sum += A[a_idx] * B[b_idx];
    }
    
    int c_idx = batch_idx * m * n + row * n + col;
    C[c_idx] = sum;
}

// Matrix-vector multiplication: y = A * x (for step function)
// A: [m, n], x: [n], y: [m]
__kernel void matvec(
    __global const float* A,          // Weight matrix [m, n]
    __global const float* x,          // Input vector [n]
    __global float* y,                // Output vector [m]
    const int m,                      // Rows in A
    const int n                       // Columns in A, size of x
) {
    const int row = get_global_id(0);
    if (row >= m) return;
    
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) {
        sum += A[row * n + i] * x[i];
    }
    
    y[row] = sum;
}

