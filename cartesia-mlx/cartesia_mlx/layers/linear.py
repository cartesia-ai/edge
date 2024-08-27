from functools import partial

import mlx.core as mx
import mlx.nn as nn

from cartesia_mlx.utils.configure import Inherit, set_cfg


class Linear(nn.Module):
    """A linear layer."""

    base_cfg = dict(
        _class_="layers.linear.Linear",
        quantization_kwargs=Inherit(default=None),
        input_dims=None,
        output_dims=None,
        bias=True,
    )

    def __init__(self, cfg=None, parent=None):
        super().__init__()
        set_cfg(self, cfg, parent)
        Linear = (
            partial(nn.QuantizedLinear, **self.quantization_kwargs)
            if self.quantization_kwargs
            else nn.Linear
        )
        self.linear = Linear(self.input_dims, self.output_dims, bias=self.bias)

    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor.

        Returns:
            The linear transformation of the input tensor.
        """
        return self.linear(x)
