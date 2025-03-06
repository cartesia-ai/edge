from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from cartesia_metal import conv1d_forward, conv1d_update, ssd_update
from cartesia_mlx.layers.ssd.ops import ssd_forward
from cartesia_mlx.layers.ssd.ssd import SSD
from cartesia_mlx.utils.configure import Inherit, set_cfg

uniform_init = nn.init.uniform()


SSDLayerState = Tuple[mx.array, mx.array]  # (conv_state, ssm_state)


class LlambaSSD(SSD):
    """Mohawk State Space Duality (Mamba 2) layer.

    Reference:
        Aviv Bick, Kevin Y. Li, Eric P. Xing, J. Zico Kolter, Albert Gu
        Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models. ArXiv 2024.
        https://arxiv.org/abs/2408.10189.
    """

    base_cfg = dict(
        _class_="layers.ssd.llamba_ssd.LlambaSSD",
        quantization_kwargs=Inherit(default=None),
        d_model=Inherit(default=1024),
        expand=2,
        kernel_size=4,
        d_state=64,
        n_heads=32,
        A_init_range=(1, 16),
        dt_init_range=(0.0001, 0.1),
        dt_limit=(float("-inf"), float("inf")),
        bias=False,
        conv_bias=False,
        conv_swish=False,
        quantize_in_proj=True,
    )

    def __init__(self, cfg=None, parent=None):
        nn.Module.__init__(self)
        set_cfg(self, cfg, parent)

        self.d_inner = self.d_model * self.expand
        self.d_head = self.d_inner // self.n_heads

        in_proj_dim = 2 * self.d_inner + self.n_heads * self.d_state * 2 + self.n_heads

        Linear = (
            partial(nn.QuantizedLinear, **self.quantization_kwargs)
            if self.quantization_kwargs
            else nn.Linear
        )
        self.in_proj = Linear(self.d_model, in_proj_dim, bias=self.bias)
        self.out_proj = Linear(self.d_inner, self.d_model, bias=self.bias)

        self.conv_dim = self.d_inner + self.n_heads * self.d_state * 2
        self.conv_weight = mx.zeros([self.conv_dim, self.kernel_size])
        self.conv_bias = mx.zeros([self.conv_dim])

        self.z_bias = mx.zeros([self.d_inner])
        self.D = mx.ones([self.n_heads])

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

        xBCzA_log = self.in_proj(x)

        xBC, z, A_log = mx.split(
            xBCzA_log,
            [
                self.d_inner + self.n_heads * self.d_state * 2,
                2 * (self.d_inner + self.n_heads * self.d_state),
            ],
            axis=-1,
        )
        z = z + self.z_bias

        xBC, conv_state = conv1d_forward(
            xBC, self.conv_weight, self.conv_bias, conv_state, swish=self.conv_swish
        )

        x, B, C = mx.split(xBC, [self.d_inner, self.d_inner + self.d_state * self.n_heads], axis=-1)

        A = -mx.ones([self.n_heads])
        D = mx.zeros([self.n_heads])
        dt_bias = mx.zeros([self.n_heads])

        x_ = x.reshape(b, l, self.n_heads, self.d_head)

        x, ssm_state = ssd_forward(
            x_ / nn.softplus(A_log).reshape(b, l, self.n_heads, 1),
            A_log,
            A,
            B.reshape(b, l, self.n_heads, -1),
            C.reshape(b, l, self.n_heads, -1),
            D,
            dt_bias,
            dt_min=0,
            dt_max=self.dt_limit[1],
            softplus=True,
        )
        x = x.reshape(b, l, self.n_heads, self.d_head)
        x += x_ * self.D.reshape(1, 1, self.n_heads, 1)
        x = x.reshape(b, l, -1)
        x = x * nn.silu(z)
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

        xBCzA_log = self.in_proj(x)

        xBC, z, A_log = mx.split(
            xBCzA_log,
            [
                self.d_inner + self.n_heads * self.d_state * 2,
                2 * (self.d_inner + self.n_heads * self.d_state),
            ],
            axis=-1,
        )
        z = z + self.z_bias

        xBC, conv_state = conv1d_update(
            xBC,
            self.conv_weight,
            self.conv_bias,
            conv_state,
            swish=self.conv_swish,
        )

        x, B, C = mx.split(xBC, [self.d_inner, self.d_inner + self.d_state * self.n_heads], axis=-1)

        A = None
        D = mx.zeros([self.n_heads])
        dt_bias = None

        B = B.reshape(b, self.n_heads, -1)  # (b, g, n)
        C = C.reshape(b, self.n_heads, -1)  # (b, g, n)
        x_ = x.reshape(b, self.n_heads, self.d_head)  # (b, h, d_head)

        x, ssm_state = ssd_update(
            x_ / nn.softplus(A_log).reshape(b, self.n_heads, 1),
            A_log,
            A,
            B,
            C,
            D,
            dt_bias,
            dt_min=0,
            dt_max=self.dt_limit[1],
            state=ssm_state,
            softplus=True,
            flip_A_sign=True,
        )
        x = x.reshape(b, self.n_heads, self.d_head)
        x += x_ * self.D.reshape(1, self.n_heads, 1)
        x = x.reshape(b, -1)

        x = x * nn.silu(z)
        x = self.out_proj(x)
        return x, (conv_state, ssm_state)
