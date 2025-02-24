from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from cartesia_mlx.layers.sa import SelfAttention
from cartesia_mlx.utils.configure import Inherit, instantiate, set_cfg


class ResidualBlock(nn.Module):
    """A residual block composed of a single layer.

    A residual block is a layer that applies a layer to the input tensor
    and adds the input tensor to the output tensor. The layer does not
    need to have a state, but it can have a state if needed.
    """

    base_cfg = dict(
        _class_="layers.residual_block.ResidualBlock",
        quantization_kwargs=Inherit(default=None),
        d_model=Inherit(default=1024),
        layer=SelfAttention.base_cfg,
        norm_point="pre",  # None, pre, pre_resid, post
        stateful=True,
    )

    def __init__(self, cfg=None, parent=None):
        super().__init__()
        set_cfg(self, cfg, parent)
        if self.norm_point:
            self.norm = nn.RMSNorm(self.d_model)
        self.layer = instantiate(self.layer, parent=self)

    def __call__(
        self, x: mx.array, *args, state: Optional[Union[mx.array, Tuple[mx.array]]] = None, **kwargs
    ):
        """Forward pass for the residual block.

        Args:
            x: The input tensor.
            *args: Additional arguments to pass to the layers.
            state: The state of the layer.
            **kwargs: Additional keyword arguments.

        Returns:
            The output tensor and the next state if the layer is stateful.
            Otherwise, just the output tensor.
        """
        r = x
        if self.norm_point == "pre":
            x = self.norm(x)
        z = self.layer(x, *args, state=state, **kwargs)
        if self.stateful:
            x, state = z
        else:
            x, state = z, None
        if self.norm_point == "pre_resid":
            x = self.norm(x)
        x = x + r
        if self.norm_point == "post":
            x = self.norm(x)
        if self.stateful:
            return x, state
        return x

    def step(
        self, x: mx.array, *args, state: Optional[Union[mx.array, Tuple[mx.array]]] = None, **kwargs
    ):
        """Step function for the residual block.

        Args:
            x: The input tensor.
            *args: Additional arguments to pass to the layers.
            state: The state of the layer.
            **kwargs: Additional keyword arguments.
        """
        r = x
        if self.norm_point == "pre":
            x = self.norm(x)
        z = self.layer.step(x, *args, state=state, **kwargs)
        if self.stateful:
            x, state = z
        else:
            x, state = z, None
        if self.norm_point == "pre_resid":
            x = self.norm(x)
        x = x + r
        if self.norm_point == "post":
            x = self.norm(x)
        if self.stateful:
            return x, state
        return x
