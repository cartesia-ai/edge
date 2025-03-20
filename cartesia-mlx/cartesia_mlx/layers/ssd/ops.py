from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


@mx.compile
def chunk_scan_ref(
    B: mx.array,
    C: mx.array,
    x: mx.array,
    states: mx.array,
    dt: mx.array,
    dtA: mx.array,
    dtA_cumsum: mx.array,
) -> mx.array:
    """
    Computes the chunked output for the forward pass using surrogate attention and initial states.

    Args:
        B: Input mixing matrix of shape (batch_size, num_chunks, chunk_length, num_groups, n).
        C: Output mixing matrix of shape (batch_size, num_chunks, chunk_length, num_groups, n).
        x: Input tensor of shape (batch_size, num_chunks, chunk_length, num_heads, head_dim).
        states: Initial states for each chunk of shape (batch_size, num_chunks, num_heads, head_dim, n).
        dt: Time deltas of shape (batch_size, num_chunks, num_heads, chunk_length).
        dtA: Scaled time deltas multiplied by state transition matrix, of shape (batch_size, num_heads, num_chunks, chunk_length).
        dtA_cumsum: Cumulative sum of `dtA` along the chunk dimension, of shape (batch_size, num_heads, num_chunks, chunk_length).

    Returns:
        y: Output tensor of shape (batch_size, num_chunks, chunk_length, num_heads, head_dim).
    """
    b, c, cl, h, dh = x.shape
    _, _, _, g, n = B.shape

    C_repeat = mx.repeat(C, repeats=h // g, axis=3)  # (b, c, cl, h, n)
    B = B.swapaxes(2, -1).swapaxes(2, -2)  # (b, c, g, n, cl)
    C = C.swapaxes(2, -2)  # (b, c, g, cl, n)
    CB = C @ B  # (b, c, g, cl, cl)
    CB = mx.repeat(CB, repeats=h // g, axis=2)  # (b, c, h, cl, cl)
    # dtA  (b, h, c, cl)
    decay = mx.exp(segsum(dtA)).swapaxes(1, 2)  # (b, c, h, cl, cl)
    surrogate_attention_matrix = mx.tril(CB * decay, 0)  # (b, c, h, cl, cl)
    surrogate_attention_values = dt.reshape(b, c, h, cl, 1) * x.swapaxes(2, 3)  # (b, c, h, cl, dh)
    y = surrogate_attention_matrix @ surrogate_attention_values  # (b, c, h, cl, dh)
    y = mx.swapaxes(y, -2, -3)  # (b, c, cl, h, dh)
    C_repeat = C_repeat.reshape(b, c, cl, h, n, 1)
    states = states.reshape(b, c, 1, h, dh, n)
    y_prev = (states @ C_repeat).reshape(b, c, cl, h, dh)
    dtA_cumsum = dtA_cumsum.swapaxes(1, 2).swapaxes(2, 3).reshape(b, c, cl, h, 1)
    y += y_prev * dtA_cumsum
    y = y.reshape(b, c, cl, h, dh)
    return y


def chunk_state_ref(dtA_cumsum: mx.array, dt: mx.array, B: mx.array, x: mx.array) -> mx.array:
    """
    Computes the final state for each chunk.

    Args:
        dtA_cumsum: Cumulative sum of scaled time deltas (dt * A) for each chunk, of shape (batch_size, num_heads, num_chunks, chunk_length).
        dt: Time deltas for each chunk, of shape (batch_size, num_chunks, num_heads, chunk_length).
        B: Input mixing matrix for each chunk, of shape (batch_size, num_chunks, chunk_length, num_groups, n).
        x: Input tensor for each chunk, of shape (batch_size, num_chunks, chunk_length, num_heads, head_dim).

    Returns:
        states: Final states for each chunk after applying decays and transformations, of shape (batch_size, num_chunks, num_heads, head_dim * n).
    """
    b, c, cl, h, dh = x.shape
    _, _, _, g, n = B.shape
    _, _, h, _ = dt.shape

    decay_states = mx.exp((dtA_cumsum[:, :, :, -1:] - dtA_cumsum))  # (b, h, c, cl)
    decay_states = decay_states.swapaxes(1, 2)  # (b, c, h, cl)
    decay_dt = decay_states * dt  # (b, c, h, cl)
    decay_dt = decay_dt.swapaxes(-1, -2).reshape(b, c, cl, h, 1)  # (b, c, cl, h, 1)

    x_decay = x * decay_dt  # (b, c, cl, h, dh)
    x_decay = x_decay.swapaxes(2, -2).swapaxes(-1, -2)  # (b, c, h, dh, cl)

    B_repeat = mx.repeat(B, repeats=h // g, axis=-2)  # (b, c, cl, h, n)
    B_repeat = B_repeat.swapaxes(2, 3)  # (b, c, h, cl, n)

    states = x_decay @ B_repeat  # (b, c, h, dh, n)
    states = states.reshape(b, c, h, dh * n)
    return states


def state_passing_ref(states: mx.array, dtA_cumsum: mx.array, dh: int, n: int) -> mx.array:
    """
    Passes states through chunks in a recurrent fashion.

    Args:
        states: The input states for each chunk, of shape (batch_size, num_chunks, num_heads, head_dim, n).
        dtA_cumsum: Cumulative sum of scaled time deltas (dt * A) for each chunk, of shape (batch_size, num_heads, num_chunks, chunk_length).
        state: Initial state for passing between chunks, of shape (batch_size, num_heads, head_dim * n).
        dh: Dimension of the head in the input tensor.
        n: Size of the state for each head.

    Returns:
        Updated states passed through the chunks in recurrent fashion, of shape (batch_size, num_chunks, num_heads, head_dim, n).
    """
    b, h, c, cl = dtA_cumsum.shape

    if states.dtype not in [mx.float32]:
        states = states.astype(mx.float32)  # we want high prec for this

    state = mx.zeros([b, h, dh * n])  # initial state

    passing_gates = dtA_cumsum[..., -1].swapaxes(1, 2).reshape(b, c, h, 1)

    current_state = state
    passed_states = mx.zeros([b, c, h, dh * n], dtype=dtA_cumsum.dtype)
    passed_states[:, 0] = current_state
    for chunk_idx in range(1, c):
        current_state = (
            current_state * mx.exp(passing_gates[:, chunk_idx]) + states[:, chunk_idx]
        )  # (b, h, dh*n)
        passed_states[:, chunk_idx] = current_state

    states = passed_states.reshape(b, c, h, dh, n).astype(dtA_cumsum.dtype)
    return states


@mx.compile
def ssd_forward_chunk_ref(
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt_bias: mx.array,
    dt_min: float,
    dt_max: float,
    softplus: bool = True,
    chunk_size: int = 128,
) -> Tuple[mx.array, mx.array]:
    """
    SSD-SSM forward pass with chunking.

    Args:
        x: Input tensor of shape (batch_size, length, num_heads, head_dim).
        dt: Time deltas of shape (batch_size, length, num_heads).
        A: State transition matrix of shape (num_heads,).
        B: Input mixing matrix of shape (batch_size, length, num_groups, n).
        C: Output mixing matrix of shape (batch_size, length, num_groups, n).
        D: Residual connection matrix of shape (num_heads,).
        dt_bias: Bias for time deltas of shape (num_heads,).
        dt_min: Minimum clipping value for time deltas.
        dt_max: Maximum clipping value for time deltas.
        softplus: Apply softplus to time deltas.
        chunk_size: The size of chunks to process in the forward pass.

    Returns:
        y: Output tensor of shape (batch_size, length, num_heads * head_dim).
        final_state: Final state of shape (batch_size, num_heads, head_dim, n).
    """
    raise NotImplementedError  # TODO: fix bug
    b, l, h, dh = x.shape
    _, _, g, n = B.shape
    cl = chunk_size

    if dt_bias is not None:
        dt = dt + dt_bias.reshape(1, 1, -1)

    if softplus is True:
        dt = nn.softplus(dt)

    dt = mx.clip(dt, a_min=dt_min, a_max=dt_max).astype(x.dtype)

    n_chunks = (l + cl - 1) // cl
    c = n_chunks
    n = B.shape[-1]
    if l % cl != 0:
        dt = mx.pad(dt, [(0, 0), (0, cl - (l % cl)), (0, 0)])
        x = mx.pad(x, [(0, 0), (0, cl - (l % cl)), (0, 0), (0, 0)])
        B = mx.pad(B, [(0, 0), (0, cl - (l % cl)), (0, 0), (0, 0)])
        C = mx.pad(C, [(0, 0), (0, cl - (l % cl)), (0, 0), (0, 0)])

    dt = mx.reshape(dt, (b, h, c, cl))

    dtA = dt * A.reshape(1, -1, 1, 1)
    dtA_cumsum = mx.cumsum(dtA, axis=-1)  # (b, h, c, cl)
    assert dtA_cumsum.shape == (b, h, c, cl)

    dt = dt.swapaxes(1, 2)  # (b, c, h, cl)

    B = mx.reshape(B, (b, c, cl, g, n))
    C = mx.reshape(C, (b, c, cl, g, n))
    x = mx.reshape(x, (b, c, cl, h, dh))

    states = chunk_state_ref(dtA_cumsum, dt, B, x)
    states = state_passing_ref(states, dtA_cumsum, dh, n)  # (b, c, h, dh, n)
    y = chunk_scan_ref(B, C, x, states, dt, dtA, dtA_cumsum)

    #  x (b, c, cl, h, dh)

    if D is not None:
        y += x * D.reshape(1, 1, h, 1)

    y = y.reshape(b, c * cl, h * dh)

    if l < c * cl:
        y = y[:, :l]

    final_state = states[:, -1]  # (b, h, dh, n)
    return y, final_state


@mx.compile
def ssd_forward_attn(
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt_bias: mx.array,
    dt_min: float,
    dt_max: float,
    softplus: bool = True,
) -> Tuple[mx.array, mx.array]:
    """SSD-SSM forward pass.

    Args:
        x: Input of shape (batch_size, num_heads, head_dim).
        dt: Time deltas of shape (num_heads,).
        A: State transition of shape (num_heads,).
        B: Input mixing of shape (batch_size, num_groups, n).
        C: Output mixing of shape (batch_size, num_groups, n).
        D: Residual connection.
        dt_bias: Bias for time deltas of shape (num_heads,).
        dt_min: Minimum value for time deltas after clipping.
        dt_max: Maximum value for time deltas after clipping.
    """
    b, l, h, dh = x.shape
    _, _, g, _ = B.shape

    dt = dt + dt_bias.reshape(1, 1, -1)

    if softplus is True:
        dt = nn.softplus(dt)

    dt = mx.clip(dt, a_min=dt_min, a_max=dt_max).astype(x.dtype)

    B = mx.swapaxes(mx.swapaxes(B, 1, 3), 1, 2)
    C = mx.swapaxes(C, 1, 2)

    CB = C @ B
    CB = mx.repeat(CB, repeats=h // g, axis=1)

    dtA = dt * A.reshape(1, 1, -1)
    dtA = mx.swapaxes(dtA, 1, 2)

    decay = mx.exp(segsum(dtA))

    surrogate_attention_matrix = mx.tril(CB * decay, 0)

    dtx = dt.reshape(b, l, h, 1) * x
    y = surrogate_attention_matrix @ dtx.swapaxes(1, 2)
    y = mx.swapaxes(y, 1, 2)

    decay = decay[:, :, -1, :].reshape(b, h, l).swapaxes(1, 2).reshape(b, l, h, 1)
    B = mx.repeat(B, h // g, axis=1).swapaxes(2, 3)
    dtxdecay = dtx * decay
    dtxdecay = dtxdecay.swapaxes(1, 2).swapaxes(2, 3)
    next_state = dtxdecay @ B

    y += x * D.reshape(1, 1, h, 1)
    y = y.reshape(b, l, h * dh)

    return y, next_state


def ssd_forward(
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt_bias: mx.array,
    dt_min: float,
    dt_max: float,
    softplus: bool = True,
    chunk_size: int = 128,
    chunk_min_len: int = 256,
    enable_chunked: bool = False,
) -> Tuple[mx.array, mx.array]:
    """
    SSD-SSM forward pass that switches between chunking, attention, and update modes.

    Args:
        x: Input tensor of shape (batch_size, length, num_heads, head_dim).
        dt: Time deltas of shape (batch_size, length, num_heads).
        A: State transition matrix of shape (num_heads,).
        B: Input mixing matrix of shape (batch_size, length, num_groups, n).
        C: Output mixing matrix of shape (batch_size, length, num_groups, n).
        D: Residual connection matrix of shape (num_heads,).
        dt_bias: Bias for time deltas of shape (num_heads,).
        dt_min: Minimum clipping value for time deltas.
        dt_max: Maximum clipping value for time deltas.
        softplus: Apply softplus to time deltas.
        chunk_size: The size of chunks for chunk-based forward pass.
        chunk_min_len: Minimum sequence length to apply chunking.

    Returns:
        y: Output tensor of shape (batch_size, length, num_heads * head_dim).
        state: Final state of the forward pass.
    """
    if enable_chunked is False or x.shape[1] < chunk_min_len:
        return ssd_forward_attn(x, dt, A, B, C, D, dt_bias, dt_min, dt_max, softplus)
    raise NotImplementedError  # TODO: fix bug in ssd_forward_chunk_ref
    return ssd_forward_chunk_ref(x, dt, A, B, C, D, dt_bias, dt_min, dt_max, softplus, chunk_size)


@mx.compile
def segsum(x):
    """Compute the segmented cumulative sum of a tensor.

    Args:
        x: Input tensor.

    Returns:
        Segmented cumulative sum.
    """
    l = x.shape[-1]
    x = mx.repeat(x[..., None], l, axis=-1)
    x = mx.tril(x, -1)
    x_segsum = mx.cumsum(x, axis=-2)
    return x_segsum
