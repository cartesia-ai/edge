// Attention computation kernels

// Split QKV output into queries, keys, values
// Input: qkv [batch, seq_len, d_proj] where d_proj = (n_heads + 2*kv_heads) * d_head
// Output: queries [batch, seq_len, n_heads * d_head]
//         keys [batch, seq_len, kv_heads * d_head]
//         values [batch, seq_len, kv_heads * d_head]
__kernel void split_qkv(
    __global const float* qkv,
    __global float* queries,
    __global float* keys,
    __global float* values,
    const int batch_size,
    const int seq_len,
    const int n_heads,
    const int kv_heads,
    const int d_head
) {
    const int batch_idx = get_global_id(0);
    const int seq_idx = get_global_id(1);
    
    if (batch_idx >= batch_size || seq_idx >= seq_len) return;
    
    int q_dim = n_heads * d_head;
    int kv_dim = kv_heads * d_head;
    int d_proj = q_dim + 2 * kv_dim;
    
    int qkv_base = (batch_idx * seq_len + seq_idx) * d_proj;
    int q_base = (batch_idx * seq_len + seq_idx) * q_dim;
    int k_base = (batch_idx * seq_len + seq_idx) * kv_dim;
    int v_base = (batch_idx * seq_len + seq_idx) * kv_dim;
    
    // Copy queries
    for (int i = 0; i < q_dim; ++i) {
        queries[q_base + i] = qkv[qkv_base + i];
    }
    
    // Copy keys
    for (int i = 0; i < kv_dim; ++i) {
        keys[k_base + i] = qkv[qkv_base + q_dim + i];
    }
    
    // Copy values
    for (int i = 0; i < kv_dim; ++i) {
        values[v_base + i] = qkv[qkv_base + q_dim + kv_dim + i];
    }
}

// Reshape and transpose for attention
// Input: flat [batch, seq_len, n_heads * d_head]
// Output: reshaped [batch, n_heads, seq_len, d_head]
__kernel void reshape_for_attention(
    __global const float* input,
    __global float* output,
    const int batch_size,
    const int seq_len,
    const int n_heads,
    const int d_head
) {
    const int batch_idx = get_global_id(0);
    const int head_idx = get_global_id(1);
    const int seq_idx = get_global_id(2);
    const int head_dim_idx = get_global_id(3);
    
    if (batch_idx >= batch_size || head_idx >= n_heads || 
        seq_idx >= seq_len || head_dim_idx >= d_head) return;
    
    // Input layout: [batch, seq_len, n_heads * d_head]
    int input_idx = batch_idx * seq_len * n_heads * d_head +
                    seq_idx * n_heads * d_head +
                    head_idx * d_head +
                    head_dim_idx;
    
    // Output layout: [batch, n_heads, seq_len, d_head]
    int output_idx = batch_idx * n_heads * seq_len * d_head +
                     head_idx * seq_len * d_head +
                     seq_idx * d_head +
                     head_dim_idx;
    
    output[output_idx] = input[input_idx];
}

// Scaled dot-product attention
// Q: [batch, n_heads, seq_len_q, d_head]
// K: [batch, kv_heads, seq_len_kv, d_head]
// V: [batch, kv_heads, seq_len_kv, d_head]
// Output: [batch, n_heads, seq_len_q, d_head]
__kernel void scaled_dot_product_attention(
    __global const float* Q,
    __global const float* K,
    __global const float* V,
    __global float* output,
    const float scale,
    const int batch_size,
    const int n_heads,
    const int kv_heads,
    const int seq_len_q,
    const int seq_len_kv,
    const int d_head,
    const int causal  // 1 for causal masking, 0 otherwise
) {
    const int batch_idx = get_global_id(0);
    const int head_idx = get_global_id(1);
    const int seq_q_idx = get_global_id(2);
    
    if (batch_idx >= batch_size || head_idx >= n_heads || seq_q_idx >= seq_len_q) return;
    
    // Compute attention scores for this query position
    // For each key position
    float scores[128];  // Maximum seq_len_kv (will need to handle larger sequences)
    float max_score = -INFINITY;
    
    int kv_head_idx = head_idx % kv_heads;
    
    // Compute Q @ K^T
    for (int seq_kv_idx = 0; seq_kv_idx < seq_len_kv && seq_kv_idx < 128; ++seq_kv_idx) {
        // Check causal mask
        if (causal && seq_kv_idx > seq_q_idx) {
            scores[seq_kv_idx] = -INFINITY;
            continue;
        }
        
        float score = 0.0f;
        for (int d = 0; d < d_head; ++d) {
            int q_idx = batch_idx * n_heads * seq_len_q * d_head +
                       head_idx * seq_len_q * d_head +
                       seq_q_idx * d_head +
                       d;
            int k_idx = batch_idx * kv_heads * seq_len_kv * d_head +
                       kv_head_idx * seq_len_kv * d_head +
                       seq_kv_idx * d_head +
                       d;
            score += Q[q_idx] * K[k_idx];
        }
        scores[seq_kv_idx] = score * scale;
        if (scores[seq_kv_idx] > max_score) {
            max_score = scores[seq_kv_idx];
        }
    }
    
    // Softmax
    float exp_sum = 0.0f;
    for (int seq_kv_idx = 0; seq_kv_idx < seq_len_kv && seq_kv_idx < 128; ++seq_kv_idx) {
        float exp_val = exp(scores[seq_kv_idx] - max_score);
        scores[seq_kv_idx] = exp_val;
        exp_sum += exp_val;
    }
    
    // Normalize and compute output
    for (int d = 0; d < d_head; ++d) {
        float val = 0.0f;
        for (int seq_kv_idx = 0; seq_kv_idx < seq_len_kv && seq_kv_idx < 128; ++seq_kv_idx) {
            int v_idx = batch_idx * kv_heads * seq_len_kv * d_head +
                       kv_head_idx * seq_len_kv * d_head +
                       seq_kv_idx * d_head +
                       d;
            val += (scores[seq_kv_idx] / exp_sum) * V[v_idx];
        }
        
        int out_idx = batch_idx * n_heads * seq_len_q * d_head +
                     head_idx * seq_len_q * d_head +
                     seq_q_idx * d_head +
                     d;
        output[out_idx] = val;
    }
}

// Reshape back from attention format
// Input: [batch, n_heads, seq_len, d_head]
// Output: [batch, seq_len, n_heads * d_head]
__kernel void reshape_from_attention(
    __global const float* input,
    __global float* output,
    const int batch_size,
    const int seq_len,
    const int n_heads,
    const int d_head
) {
    const int batch_idx = get_global_id(0);
    const int seq_idx = get_global_id(1);
    const int head_idx = get_global_id(2);
    const int head_dim_idx = get_global_id(3);
    
    if (batch_idx >= batch_size || seq_idx >= seq_len || 
        head_idx >= n_heads || head_dim_idx >= d_head) return;
    
    // Input layout: [batch, n_heads, seq_len, d_head]
    int input_idx = batch_idx * n_heads * seq_len * d_head +
                    head_idx * seq_len * d_head +
                    seq_idx * d_head +
                    head_dim_idx;
    
    // Output layout: [batch, seq_len, n_heads * d_head]
    int output_idx = batch_idx * seq_len * n_heads * d_head +
                    seq_idx * n_heads * d_head +
                    head_idx * d_head +
                    head_dim_idx;
    
    output[output_idx] = input[input_idx];
}

// Concatenate cached keys/values with new ones
// cached: [batch, kv_heads, cached_len, d_head]
// new: [batch, kv_heads, new_len, d_head]
// output: [batch, kv_heads, cached_len + new_len, d_head]
__kernel void concatenate_kv(
    __global const float* cached,
    __global const float* new_kv,
    __global float* output,
    const int batch_size,
    const int kv_heads,
    const int cached_len,
    const int new_len,
    const int d_head
) {
    const int batch_idx = get_global_id(0);
    const int head_idx = get_global_id(1);
    const int seq_idx = get_global_id(2);
    const int d_idx = get_global_id(3);
    
    if (batch_idx >= batch_size || head_idx >= kv_heads || 
        seq_idx >= cached_len + new_len || d_idx >= d_head) return;
    
    int total_len = cached_len + new_len;
    int idx = batch_idx * kv_heads * total_len * d_head +
              head_idx * total_len * d_head +
              seq_idx * d_head +
              d_idx;
    
    if (seq_idx < cached_len) {
        // Copy from cached
        int cached_idx = batch_idx * kv_heads * cached_len * d_head +
                        head_idx * cached_len * d_head +
                        seq_idx * d_head +
                        d_idx;
        output[idx] = cached[cached_idx];
    } else {
        // Copy from new
        int new_idx = batch_idx * kv_heads * new_len * d_head +
                     head_idx * new_len * d_head +
                     (seq_idx - cached_len) * d_head +
                     d_idx;
        output[idx] = new_kv[new_idx];
    }
}
