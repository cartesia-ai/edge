from typing import Tuple

import mlx.core as mx
import mlx.nn as nn


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
        state: Previous state for recurrent connections.
            Shape (batch_size, n_heads, d_head, d_state).

    Returns:
        Output and next state.
    """
    b, l, h, dh = x.shape
    _, _, g, _ = B.shape

    if dt_bias is not None:
        dt = dt + dt_bias.reshape(1, 1, -1)

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

    if D is not None:
        y += x * D.reshape(1, 1, h, 1)

    y = y.reshape(b, l, h * dh)

    return y, next_state


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
