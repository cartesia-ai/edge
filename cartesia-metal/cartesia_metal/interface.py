from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from ._ext import (
    conv1d_forward_,
    conv1d_swish_forward_,
    conv1d_swish_update_,
    conv1d_update_,
    ssd_update_,
    ssd_update_no_z_,
)


@mx.compile
def conv1d_update(
    x: mx.array,
    w: mx.array,
    b: mx.array,
    state: mx.array,
    swish: bool = False,
) -> Tuple[mx.array, mx.array]:
    """Performs a 1D depthwise convolution update.

    Args:
        x: Input tensor of shape (b, d).
        w: Convolution weights of shape (d, k).
        b: Convolution bias of shape (d,).
        state: Optional tensor of shape (b, d, k-1) to maintain state across inputs.

    Returns:
        Output tensor of shape (b, d) and the updated state tensor.
    """
    fn = conv1d_swish_update_ if swish is True else conv1d_update_
    result = fn(x, w, b, state)
    return result[0], result[1]


@mx.compile
def conv1d_forward(
    x: mx.array,
    w: mx.array,
    b: mx.array,
    state: mx.array,
    swish: bool = False,
) -> Tuple[mx.array, mx.array]:
    """Performs a 1D depthwise convolution forward pass.

    Args:
        x: Input tensor of shape (b, l, d).
        w: Convolution weights of shape (d, k).
        b: Convolution bias of shape (d,).
        state: Optional tensor of shape (b, d, k-1) to maintain state across inputs.

    Returns:
        Output tensor of shape (b, l, d) and the updated state tensor.
    """
    kernel_size = w.shape[1]
    x = x.swapaxes(1, 2)
    x = mx.concatenate([state, x], axis=-1)
    next_state = x[:, :, -kernel_size + 1 :]
    next_state = mx.as_strided(next_state)
    fn = conv1d_swish_forward_ if swish is True else conv1d_forward_
    y = fn(x, w, b)[0]
    y = y[..., : -kernel_size + 1]
    y = y.swapaxes(1, 2)
    return y, next_state


def ssd_update(
    x: mx.array,
    dt: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    dt_bias: mx.array,
    dt_min: float,
    dt_max: float,
    state: mx.array,
    softplus: bool = True,
    z: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Updates the state space model with the given inputs and parameters.

    Args:
        x: Input tensor of shape (b, h, dh).
        dt: Time deltas tensor of shape (b, h).
        A: State transition tensor of shape (h).
        B: Input mixing tensor of shape (b, g, n).
        C: Output mixing tensor of shape (b, g, n).
        D: Residual connection tensor of shape (h).
        dt_bias: Bias for the time deltas of shape (h).
        dt_min: Minimum value for time deltas after clipping.
        dt_max: Maximum value for time deltas after clipping.
        state: State tensor of shape (b, h, dh, n).
        z: Optional gating tensor of shape (b, h*dh).

    Returns:
        tuple: Output tensor reshaped to (b, h*dh) and the updated state tensor.
    """
    b, h, dh = x.shape

    if dt_bias is not None:
        dt = dt + dt_bias.reshape(1, -1)

    if softplus is True:
        dt = nn.softplus(dt)

    dt = mx.clip(dt, a_min=dt_min, a_max=dt_max).astype(dt.dtype)

    if A is not None:
        decay = mx.exp(dt * A.reshape(1, -1))  # (b, h)
    else:
        decay = mx.exp(dt)

    if z is not None:
        x, state = ssd_update_(x, dt, decay, B, C, D, z, state)
    else:
        x, state = ssd_update_no_z_(x, dt, decay, B, C, D, state)
    x = x.reshape(b, h * dh)
    return x, state
