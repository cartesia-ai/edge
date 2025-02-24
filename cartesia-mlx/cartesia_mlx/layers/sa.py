from functools import partial
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from cartesia_mlx.utils.configure import Inherit, set_cfg

SelfAttentionLayerState = Tuple[mx.array, mx.array, mx.array]  # (keys, values, mask)


class SelfAttention(nn.Module):
    """Implements Self-Attention https://arxiv.org/abs/1706.03762 and Multi-Query-Attention https://arxiv.org/abs/1911.02150."""

    base_cfg = dict(
        _class_="layers.sa.SelfAttention",
        quantization_kwargs=Inherit(default=None),
        d_model=Inherit(default=1024),
        d_head=64,
        kv_heads=1,
        expand=None,
        bias=False,
        causal=True,
        max_context_len=4096,
    )

    def __init__(self, cfg=None, parent=None):
        super().__init__()
        set_cfg(self, cfg, parent)
        if self.expand is None:
            self.d_inner = self.d_model
        else:
            self.d_inner = int(round(self.expand * self.d_model))
        self.n_heads = self.d_inner // self.d_head
        self.kv_heads = self.n_heads if self.kv_heads is None else self.kv_heads
        self.softmax_scale = 1 / (self.d_head**0.5)
        self.d_proj = (self.n_heads + 2 * self.kv_heads) * self.d_head
        Linear = (
            partial(nn.QuantizedLinear, **self.quantization_kwargs)
            if self.quantization_kwargs
            else nn.Linear
        )
        self.qkv_proj = Linear(self.d_model, self.d_proj, bias=self.bias)
        self.out_proj = Linear(self.n_heads * self.d_head, self.d_model, bias=self.bias)

    def __call__(
        self,
        x: mx.array,
        *args,
        state: Optional[SelfAttentionLayerState] = None,
        mask: Optional[mx.array] = None,
        **kwargs,
    ) -> Tuple[mx.array, SelfAttentionLayerState]:
        """
        Args:
            x (mx.array): Input tensor of shape (batch_size, seq_len, d_model).
            state: Tuple containing previous keys, values, and mask state (optional).
            mask: Attention mask (optional).

        Returns:
            tuple: Output tensor and updated state.
        """
        b, l, _ = x.shape
        qkv = self.qkv_proj(x)
        queries, keys, values = mx.split(
            qkv,
            indices_or_sections=[
                self.n_heads * self.d_head,
                (self.n_heads + self.kv_heads) * self.d_head,
            ],
            axis=-1,
        )

        queries = queries.reshape(b, l, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(b, l, self.kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(b, l, self.kv_heads, -1).transpose(0, 2, 1, 3)

        if state is not None:
            key_state, value_state, mask_state = state
            keys = mx.concatenate([key_state, keys], axis=2)
            values = mx.concatenate([value_state, values], axis=2)
            if mask is None and mask_state is not None:
                mask_state = mx.concatenate(
                    [mask_state, mx.ones([b, l], dtype=x.dtype)], axis=1
                )  # (b, l+l_state)
            if mask is not None and mask_state is None:
                l_state = key_state.shape[2]
                mask_state = mx.concatenate([mx.ones([b, l_state], dtype=x.dtype), mask], axis=1)
            elif mask is not None and mask_state is not None:
                mask_state = mx.concatenate(mask_state, mask, axis=1)
        else:
            mask_state = mask

        l_q, l_kv = l, keys.shape[2]
        mask_lq_x_lkv = _construct_mask(
            b, mask, mask_state, l_q, l_kv, x.dtype, self.causal, self.max_context_len
        )

        dtype = queries.dtype

        x = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.softmax_scale, mask=mask_lq_x_lkv
        )

        if x.dtype != dtype:
            x = x.astype(dtype)

        x = x.transpose(0, 2, 1, 3).reshape(b, l, -1)

        x = self.out_proj(x)
        state = (keys, values, mask_state)
        return x, state

    def step(
        self, x: mx.array, *args, state: Optional[SelfAttentionLayerState] = None, **kwargs
    ) -> Tuple[mx.array, SelfAttentionLayerState]:
        """
        Args:
            x (mx.array): Input tensor of shape (batch_size, d_model).
            state: Tuple containing previous keys, values, and mask state (optional).

        Returns:
            tuple: Output tensor and updated state.
        """
        b, d = x.shape
        x = x.reshape(b, 1, d)

        # Truncate state at max_context_len
        if self.max_context_len is not None and state[1].shape[2] > self.max_context_len:
            keys = state[0][:, :, -self.max_context_len - 1 :]
            values = state[1][:, :, -self.max_context_len - 1 :]
            mask_state = None if state[2] is None else state[2][: -self.max_context_len - 1 :]
            state = (keys, values, mask_state)

        x, state = self.__call__(x, *args, state=state, **kwargs)
        return x[:, 0, :], state


def _construct_mask(
    b, mask_q, mask_kv, l_q, l_kv, dtype, causal, max_context_len, max_context_len_forces_caual=True
):
    if mask_q is None and mask_kv is None and not causal and not max_context_len:
        return None
    if mask_q is None:
        mask_q = mx.ones([b, l_q], dtype=dtype)
    if mask_kv is None:
        mask_kv = mx.ones([b, l_kv], dtype=dtype)
    mask_lq_x_lkv = mask_q.reshape(b, l_q, 1).astype(dtype) @ mask_kv.reshape(b, 1, l_kv).astype(
        dtype
    )  # (b, l_q, l_kv)
    mask_lq_x_lkv = mx.where(mask_lq_x_lkv == 0, float("-inf"), 0)
    diag_offset = l_kv - l_q
    if causal or max_context_len and max_context_len_forces_caual:
        causal_mask = mx.tri(l_q, l_kv, k=diag_offset, dtype=dtype)
        causal_mask = mx.where(causal_mask == 0, float("-inf"), 0)
        mask_lq_x_lkv += causal_mask.reshape(1, l_q, l_kv)
    if max_context_len:
        context_mask = mx.tri(l_q, l_kv, k=-max_context_len + diag_offset, dtype=dtype)
        context_mask = mx.where(context_mask == 1, float("-inf"), 0)
        if not causal:
            context_mask_upper = mx.tri(l_q, l_kv, k=max_context_len + diag_offset - 1)
            context_mask_upper = mx.where(context_mask_upper == 0, float("-inf"), 0)
            context_mask += context_mask_upper
        mask_lq_x_lkv += context_mask.reshape(1, l_q, l_kv)
    mask_lq_x_lkv = mask_lq_x_lkv.reshape(b, 1, l_q, l_kv)
    return mask_lq_x_lkv
