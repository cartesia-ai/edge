import copy
from functools import partial
from typing import Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from cartesia_mlx.layers.residual_block import ResidualBlock
from cartesia_mlx.layers.sa import SelfAttention
from cartesia_mlx.layers.ssd.ssd import SSD
from cartesia_mlx.utils.configure import Inherit, instantiate, set_cfg, sub_cfg
from cartesia_mlx.utils.registry import NORMS

State = Union[mx.array, Tuple[mx.array]]


class SequenceModel(nn.Module):
    """A base class for a sequence model.

    A sequence model is composed of a sequence of stateful layers (e.g. ssd, attention, etc.).
    These layers can be configured in the config displayed below.
    """

    base_cfg = dict(
        _class_="layers.sequence_model.SequenceModel",
        quantization_kwargs=Inherit(default=None),
        d_model=Inherit(default=512),
        n_layer_repeats=1,
        unique_layers=[
            sub_cfg(ResidualBlock.base_cfg, layer=SelfAttention.base_cfg),
            sub_cfg(ResidualBlock.base_cfg, layer=SSD.base_cfg),
        ],
        post_norm=True,
        pre_norm=False,
        final_proj=False,
        norm_type="rms",
    )

    def __init__(self, cfg=None, parent=None):
        super().__init__()
        set_cfg(self, cfg, parent)
        if self.post_norm or self.pre_norm:
            self.norm = NORMS[self.norm_type](self.d_model)
        layers = [self.unique_layers] * self.n_layer_repeats
        layers = flatten([copy.deepcopy(layer) for layer in layers])
        self.layers = [instantiate(layer, parent=self) for layer in layers]

        if self.final_proj:
            Linear = (
                partial(nn.QuantizedLinear, **self.quantization_kwargs)
                if self.quantization_kwargs
                else nn.Linear
            )
            self.out_proj = Linear(self.d_model, self.d_model)

    def __call__(
        self, x: mx.array, *args, state: Optional[State] = None, **kwargs
    ) -> Tuple[mx.array, State]:
        """Forward pass on the sequence model.

        Args:
            x: The input tensor. Shape (batch_size, seq_len, ...).
            *args: Additional arguments to pass to the layers.
            state: The state of the model.
            **kwargs: Additional keyword arguments to pass to the layers.

        Returns:
            The output tensor and the next state.
        """
        if self.pre_norm:
            x = self.norm(x)
        next_state = []
        for i, layer in enumerate(self.layers):
            state_i = state[i] if state else None
            z = layer(x, *args, state=state_i, **kwargs)
            if layer.stateful is True:
                x, state_i = z
            else:
                x, state_i = z, None
            next_state.append(state_i)
        if self.post_norm:
            x = self.norm(x)
        if self.final_proj:
            x = self.out_proj(x)
        return x, next_state

    def step(self, x: mx.array, *args, state=None, **kwargs) -> Tuple[mx.array, State]:
        """A single step on the sequence model.

        Args:
            x: Input tensor.
            state: State of the model.

        Returns:
            The output and the next state.
        """
        if self.pre_norm:
            x = self.norm(x)
        next_state = []
        for i, layer in enumerate(self.layers):
            state_i = state[i] if state else None
            z = layer.step(x, *args, state=state_i, **kwargs)
            if layer.stateful is True:
                x, state_i = z
            else:
                x, state_i = z, None
            next_state.append(state_i)
        if self.post_norm:
            x = self.norm(x)
        if self.final_proj:
            x = self.out_proj(x)
        return x, next_state


def flatten(x) -> list:
    """Flattens a list of list-likes (e.g. list, array, tensor).

    Args:
        x: A list-like object that can be selected like, x[0], x[1], ...

    Returns:
        List: A flattened list of all elements in x.
    """
    y = []
    for xi in x:
        y.extend(xi)
    return y
