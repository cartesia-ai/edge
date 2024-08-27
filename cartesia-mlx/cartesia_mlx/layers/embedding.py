import mlx.core as mx
import mlx.nn as nn

from cartesia_mlx.utils.configure import Inherit, set_cfg


class Embedding(nn.Module):
    """An embedding lookup layer."""

    base_cfg = dict(
        _class_="layers.embedding.Embedding",
        d_model=Inherit(default=1024),
        n_tokens=50288,
    )

    def __init__(self, cfg=None, parent=None):
        super().__init__()
        set_cfg(self, cfg, parent)
        self.encoder = nn.Embedding(self.n_tokens, self.d_model)

    def encode(self, x: mx.array) -> mx.array:
        """Encode the input tensor.

        Args:
            x: The input tensor. Shape (...).

        Returns:
            The encoded tensor. Shape (..., d_model).
        """
        return self.encoder(x)

    def __call__(self, x: mx.array) -> mx.array:
        """See :meth:`encode`."""
        return self.encode(x)
