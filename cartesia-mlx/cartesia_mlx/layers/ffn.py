from functools import partial

import mlx.core as mx
import mlx.nn as nn

from cartesia_mlx.utils.configure import Inherit, set_cfg
from cartesia_mlx.utils.registry import ACTIVATIONS, NORMS


class FFN(nn.Module):
    """Standard feed-forward network."""

    base_cfg = dict(
        _class_="layers.ffn.FFN",
        quantization_kwargs=Inherit(default=None),
        quantize_gate=True,
        d_model=Inherit(default=1024),
        expand=4,
        activation="swish",  # or 'gelu'
        norm=None,
        glu=False,
        bias=False,
        bugged=False,  # skilled cherry is trained on a bugged model where both the gate and input are activated.
    )

    def __init__(self, cfg=None, parent=None):
        super().__init__()
        set_cfg(self, cfg, parent)
        Linear = (
            partial(nn.QuantizedLinear, **self.quantization_kwargs)
            if self.quantization_kwargs
            else nn.Linear
        )
        self.d_inner = int(round(self.expand * self.d_model))
        self.in_proj = Linear(self.d_model, self.d_inner, bias=self.bias)
        self.out_proj = Linear(self.d_inner, self.d_model, bias=self.bias)
        if self.glu:
            if self.quantization_kwargs and self.quantize_gate:
                Linear = partial(nn.QuantizedLinear, **self.quantization_kwargs)
            else:
                Linear = nn.Linear
            self.gate_proj = Linear(self.d_model, self.d_inner, bias=self.bias)
        if self.norm:
            self.norm = NORMS[self.norm](self.d_inner)
        self.activation = ACTIVATIONS[self.activation]

    def __call__(self, x: mx.array, *args, **kwargs) -> mx.array:
        """Forward pass.

        Args:
            x: The input tensor.

        Returns:
            The output of the feed-forward network.
                Shape (batch_size, seq_len, d_model).
        """
        y = self.in_proj(x)
        y = self.activation(y)
        if self.glu:
            g = self.gate_proj(x)
            y = y * g
            if self.norm:
                y = self.norm(y)
        x = self.out_proj(y)
        return x

    def step(self, x: mx.array, *args, **kwargs) -> mx.array:
        """See :meth:`__call__`."""
        x = self(x, *args, **kwargs)
        return x


class SwiGLU(FFN):
    """A feed-forward network with Swish-GLU.

    Reference:
        Shazeer et al. GLU Variants Improve Transformer. ArXiv 2020.
        `https://arxiv.org/pdf/2002.05202v1`.
    """

    base_cfg = dict(
        _class_="layers.ffn.SwiGLU",
        quantization_kwargs=Inherit(default=None),
        d_model=Inherit(default=1024),
        expand=2,
        glu=True,
        bias=False,
        norm=True,
    )
