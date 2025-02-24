from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from cartesia_mlx.layers.ffn import FFN
from cartesia_mlx.layers.ssd.mohawk_ssd import MohawkSSD
from cartesia_mlx.utils.configure import Inherit, instantiate, set_cfg


class PhiBlock(nn.Module):
    """A PhiBlock consisting of a channel mixing and a time mixing module.

    Reference:
        1.  Marah Abdin et. al.
            Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone
            https://arxiv.org/abs/2404.14219

        2.  Aviv Bick, Kevin Y. Li, Eric P. Xing, J. Zico Kolter, Albert Gu
            Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models. ArXiv 2024.
            https://arxiv.org/abs/2408.10189.
    """

    base_cfg = dict(
        _class_="layers.phi_block.PhiBlock",
        d_model=Inherit(default=1024),
        quantization_kwargs=Inherit(default=None),
        channel_mixing=FFN.base_cfg,
        time_mixing=MohawkSSD.base_cfg,
        stateful=True,
    )

    def __init__(self, cfg=None, parent=None):
        super().__init__()
        set_cfg(self, cfg, parent)
        self.norm = nn.LayerNorm(self.d_model, eps=1e-5)
        self.channel_mixing = instantiate(self.channel_mixing, parent=self)
        self.time_mixing = instantiate(self.time_mixing, parent=self)

    def __call__(
        self, x: mx.array, *args, state: Optional[Union[mx.array, Tuple[mx.array]]] = None, **kwargs
    ):
        """Forward pass for the phi block.

        Args:
            x: The input tensor.
            *args: Additional arguments to pass to the layers.
            state: The state of the layer.
            **kwargs: Additional keyword arguments.

        Returns:
            The output tensor and the next state.
            Otherwise, just the output tensor.
        """
        r = x
        # x = self.norm(x)
        if len(x.shape) == 3 and x.shape[1] == 1:
            b, l, d = x.shape
            x = self.norm(x.reshape(b, d)).reshape(b, l, d)
        else:
            x = self.norm(x)
        x_1, state = self.time_mixing(x, *args, state=state, **kwargs)
        x_2 = self.channel_mixing(x, *args, **kwargs)
        x = x_1 + x_2 + r
        return x, state

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
        x = self.norm(x)
        x_1, state = self.time_mixing.step(x, *args, state=state, **kwargs)
        x_2 = self.channel_mixing(x, *args, **kwargs)
        x = x_1 + x_2 + r
        return x, state
