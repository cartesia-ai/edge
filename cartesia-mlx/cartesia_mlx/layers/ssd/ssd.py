from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from cartesia_metal import conv1d_forward, conv1d_update, ssd_update
from cartesia_mlx.layers.ssd.ops import ssd_forward
from cartesia_mlx.utils.configure import Inherit, set_cfg

uniform_init = nn.init.uniform()


SSDLayerState = Tuple[mx.array, mx.array]  # (conv_state, ssm_state)


class SSD(nn.Module):
    """State Space Duality (Mamba 2) layer.

    Reference:
        Dao* & Gu* Transformers are SSMs: Generalized Models and Efficient Algorithms
        Through Structured State Space Duality. ArXiv 2024.
        https://arxiv.org/abs/2405.21060.
    """

    base_cfg = dict(
        _class_="layers.ssd.ssd.SSD",
        quantization_kwargs=Inherit(default=None),
        d_model=Inherit(default=1024),
        expand=2,
        kernel_size=4,
        d_state=64,
        d_head=64,
        n_groups=1,
        A_init_range=(1, 16),
        dt_init_range=(0.0001, 0.1),
        dt_limit=(0.0, float("inf")),
        norm_before_gate=False,
        bias=False,
        conv_bias=False,
    )

    def __init__(self, cfg=None, parent=None):
        super().__init__()
        set_cfg(self, cfg, parent)

        self.d_inner = self.d_model * self.expand
        assert self.d_inner % self.d_head == 0
        self.n_heads = self.d_inner // self.d_head

        in_proj_dim = 2 * self.d_inner + 2 * self.d_state * self.n_groups + self.n_heads
        Linear = (
            partial(nn.QuantizedLinear, **self.quantization_kwargs)
            if self.quantization_kwargs
            else nn.Linear
        )
        self.in_proj = Linear(self.d_model, in_proj_dim, bias=self.bias)
        self.out_proj = Linear(self.d_inner, self.d_model, bias=self.bias)

        self.conv_dim = self.d_inner + 2 * self.d_state * self.n_groups
        self.conv_weight = mx.zeros([self.conv_dim, self.kernel_size])
        self.conv_bias = mx.zeros([self.conv_dim])

        self.A = (
            -uniform_init(mx.zeros([self.n_heads])) * (self.A_init_range[1] - self.A_init_range[0])
            + self.A_init_range[0]
        )

        self.dt_bias = (
            uniform_init(mx.zeros([self.n_heads])) * (self.dt_init_range[1] - self.dt_init_range[0])
            + self.dt_init_range[0]
        )
        self.dt_bias = softplus_inverse(self.dt_bias)

        self.D = mx.ones([self.n_heads])

        self.rms_norm = nn.RMSNorm(self.d_inner)

    def __call__(
        self, x: mx.array, state: Optional[SSDLayerState] = None, **kwargs
    ) -> Tuple[mx.array, SSDLayerState]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model).
            state: Tuple containing previous convolution and SSD states.
                See :meth:`get_default_state` for the shape information.

        Returns:
            A tuple of the output and updated state.
        """
        b, l, _ = x.shape

        conv_state, ssm_state = (
            state if state is not None else self.get_default_state(batch_size=b, dtype=x.dtype)
        )
        assert ssm_state is None, "SSD state passing is not supported."

        zxBCdt = self.in_proj(x)

        z, xBC, dt = mx.split(
            zxBCdt, [self.d_inner, 2 * self.d_inner + 2 * self.n_groups * self.d_state], axis=-1
        )

        xBC, conv_state = conv1d_forward(
            xBC, self.conv_weight, self.conv_bias, conv_state, swish=True
        )

        x, B, C = mx.split(
            xBC, [self.d_inner, self.d_inner + self.d_state * self.n_groups], axis=-1
        )

        x, B, C = mx.split(
            xBC, [self.d_inner, self.d_inner + self.d_state * self.n_groups], axis=-1
        )

        x, ssm_state = ssd_forward(
            x.reshape(b, l, self.n_heads, self.d_head),
            dt,
            self.A,
            B.reshape(b, l, self.n_groups, -1),
            C.reshape(b, l, self.n_groups, -1),
            self.D,
            self.dt_bias,
            dt_min=self.dt_limit[0],
            dt_max=self.dt_limit[1],
        )

        if self.norm_before_gate is True:
            x = self.rms_norm(x)
            x = x * nn.silu(z)
        else:
            x = x * nn.silu(z)
            x = self.rms_norm(x)

        x = self.out_proj(x)
        return x, (conv_state, ssm_state)

    def step(
        self, x: mx.array, state: SSDLayerState, *args, **kwargs
    ) -> Tuple[mx.array, SSDLayerState]:
        """
        Args:
            x: An input tensor of shape (batch_size, d_model).
            state: A tuple containing previous convolution and SSD states.
                See :meth:`get_default_state` for the shape information.

        Returns:
            A tuple of the the following:
            - output: The output tensor of shape (batch_size, seq_len, d_model).
            - updated_state: The updated state.
                See :meth:`get_default_state` for the shape information.
        """
        b, _ = x.shape
        conv_state, ssm_state = state

        zxBCdt = self.in_proj(x)

        z, xBC, dt = mx.split(
            zxBCdt, [self.d_inner, 2 * self.d_inner + 2 * self.n_groups * self.d_state], axis=-1
        )

        xBC, conv_state = conv1d_update(
            xBC,
            self.conv_weight,
            self.conv_bias,
            conv_state,
            swish=True,
        )

        x, B, C = mx.split(
            xBC, [self.d_inner, self.d_inner + self.d_state * self.n_groups], axis=-1
        )

        B = B.reshape(b, self.n_groups, -1)  # (b, g, n)
        C = C.reshape(b, self.n_groups, -1)  # (b, g, n)
        x = x.reshape(b, self.n_heads, self.d_head)  # (b, h, d_head)

        x, ssm_state = ssd_update(
            x,
            dt,
            self.A,
            B,
            C,
            self.D,
            self.dt_bias,
            dt_min=self.dt_limit[0],
            dt_max=self.dt_limit[1],
            state=ssm_state,
            z=None if self.norm_before_gate is True else z,
        )
        x = self.rms_norm(x)

        if self.norm_before_gate is True:
            x = x * nn.silu(z)

        x = self.out_proj(x)

        return x, (conv_state, ssm_state)

    def get_default_state(self, batch_size=1, dtype=mx.float16):
        """Get the default state for the layer.

        Args:
            batch_size: The batch size.
            dtype: The data type.

        Returns:
            A tuple representing the state with components:
            - conv_state: The convolution state. Shape (batch_size, conv_dim, kernel_size - 1).
            - ssm_state: The SSD state. Shape (batch_size, n_heads, d_head, d_state).
        """
        conv_state = mx.zeros([batch_size, self.conv_dim, self.kernel_size - 1], dtype=dtype)
        ssm_state = None
        state = (conv_state, ssm_state)
        return state


@mx.compile
def softplus_inverse(x: mx.array):
    """Computes the inverse softplus.

    :math:`x = softplus_inverse(softplus(x))`

    Args:
        x: The input tensor.
    """
    return x + mx.log(-mx.exp(-x) + 1)
