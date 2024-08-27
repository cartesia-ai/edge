from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from flash_attn import flash_attn_with_kvcache
from mamba_ssm.models.mixer_seq_simple import _init_weights
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.modules.mha import _update_kv_cache
from mamba_ssm.utils.generation import GenerationMixin as MambaGenerationMixin
from transformers.modeling_outputs import CausalLMOutput
from transformers.modeling_utils import PreTrainedModel

from .configuration_rene import ReneConfig


class ReneMLP(nn.Module):
    """One-hidden-layer network with GELU activation.

    Args:
      d_input: Block input dimension.
      d_output: Block output dimension.
      expand: Block expansion factor.
      bias: Use biases in linear layers.
    """

    def __init__(self, d_input, d_output=None, expand=3, bias=True, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.d_input = d_input
        self.d_output = d_input if d_output is None else d_output
        self.d_inner = int(round(expand * d_input))
        self.in_proj = nn.Linear(self.d_input, self.d_inner, bias=bias, **factory_kwargs)
        self.activation = nn.GELU()
        self.out_proj = nn.Linear(self.d_inner, self.d_input, bias=bias, **factory_kwargs)

    def forward(self, x, inference_params=None):
        """Forward pass through the MLP module."""
        y = self.in_proj(x)
        y = self.activation(y)
        y = self.out_proj(y)
        return y

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache for ReneMLP. (There is nothing to cache for this module)."""
        return None


class ReneMHA(nn.Module):
    """Multi-head self-attention. Adapted from mamba_ssm MHA class."""

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_heads_kv=None,
        head_dim=None,  # If None, use embed_dim // num_heads
        qkv_proj_bias=True,
        out_proj_bias=True,
        softmax_scale=None,
        causal=True,
        sliding_window_length=None,  # If None, infinite context
        layer_idx=None,
        device=None,
        dtype=None,
    ) -> None:
        """
        num_heads_kv: can be used to toggle MQA / GQA. If None, use num_heads.
        return_residual: whether to return the input x along with the output. This is for
            performance reason: for post-norm architecture, returning the input allows us
            to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.embed_dim = embed_dim
        self.layer_idx = layer_idx
        self.softmax_scale = softmax_scale
        self.causal = causal
        assert self.causal, "Rene does not yet support non-causal modeling"

        self.num_heads = num_heads
        self.num_heads_kv = num_heads_kv if num_heads_kv is not None else num_heads
        assert (
            self.num_heads % self.num_heads_kv == 0
        ), "num_heads must be divisible by num_heads_kv"
        if head_dim is None:
            assert self.embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.head_dim = head_dim if head_dim is not None else self.embed_dim // num_heads
        qkv_dim = self.head_dim * (self.num_heads + 2 * self.num_heads_kv)
        out_dim = self.head_dim * self.num_heads

        self.sliding_window_length = sliding_window_length
        if self.sliding_window_length is None:
            self.window_size = (-1, -1)
        else:
            self.window_size = (self.sliding_window_length - 1, 0)  # for flash_attn

        self.in_proj = nn.Linear(embed_dim, qkv_dim, bias=qkv_proj_bias, **factory_kwargs)
        self.out_proj = nn.Linear(out_dim, embed_dim, bias=out_proj_bias, **factory_kwargs)

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None):
        """Allocate inference cache for the multi-head self-attention module."""
        dtype = self.out_proj.weight.dtype if dtype is None else dtype
        device = self.out_proj.weight.device
        kv_cache = torch.empty(
            batch_size,
            max_seqlen,
            2,
            self.num_heads_kv,
            self.head_dim,
            dtype=dtype,
            device=device,
        )
        return kv_cache, None

    def _pytorch_attn(self, q, kv):
        k, v = kv.unbind(dim=-3)
        k = torch.repeat_interleave(k, dim=2, repeats=self.num_heads // self.num_heads_kv)
        v = torch.repeat_interleave(v, dim=2, repeats=self.num_heads // self.num_heads_kv)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        L, S = q.size(-2), k.size(-2)
        if S > self.sliding_window_length:
            attn_mask = (
                torch.ones(L, S, dtype=torch.bool)
                .tril(diagonal=0)
                .triu(-self.window_size[0])
                .to(device=q.device)
            )
            # Since we pass in an attn_mask explicitly, we need to pass is_causal=False to
            # `scaled_dot_product_attention` (even though the attn_mask itself is in fact causal).
            is_causal_arg = False
        else:
            # The previous branch would also handle this case correctly, but it is more efficient
            # to omit the attn_mask when we don't need it.
            attn_mask = None
            is_causal_arg = True
        return F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, is_causal=is_causal_arg, scale=self.softmax_scale
        ).transpose(1, 2)

    def _update_kv_cache(self, kv, inference_params):
        """kv: (batch_size, seqlen, 2, nheads, head_dim) or (batch_size, 1, 2, nheads, head_dim)."""
        assert self.layer_idx is not None, "Generation requires layer_idx in the constructor"
        return _update_kv_cache(kv, inference_params, self.layer_idx)

    def _update_kvcache_attention(self, q, kv, inference_params):
        """Write kv to inference_params, then compute attention."""
        if inference_params.seqlen_offset == 0 or flash_attn_with_kvcache is None:
            # TODO: this only uses seqlen_offset and not lengths_per_sample.
            kv = self._update_kv_cache(kv, inference_params)
            return self._pytorch_attn(q, kv)
        else:
            batch = q.shape[0]
            kv_cache, _ = inference_params.key_value_memory_dict[self.layer_idx]
            kv_cache = kv_cache[:batch]
            cache_seqlens = (
                inference_params.lengths_per_sample[:batch]
                if inference_params.lengths_per_sample is not None
                else inference_params.seqlen_offset
            )
            return flash_attn_with_kvcache(
                q,
                kv_cache[:, :, 0],
                kv_cache[:, :, 1],
                kv[:, :, 0],
                kv[:, :, 1],
                cache_seqlens=cache_seqlens,
                softmax_scale=self.softmax_scale,
                causal=self.causal,
                window_size=self.window_size,
            )

    def forward(self, x, inference_params=None):
        """Forward pass through the multi-head self-attention module."""
        if (
            inference_params is not None
            and self.layer_idx not in inference_params.key_value_memory_dict
        ):
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
                x.shape[0], inference_params.max_seqlen, dtype=x.dtype
            )
        qkv = self.in_proj(x)
        q, kv = qkv.split(
            [self.num_heads * self.head_dim, self.num_heads_kv * 2 * self.head_dim], dim=-1
        )
        q = rearrange(q, "... (h d) -> ... h d", d=self.head_dim)
        kv = rearrange(kv, "... (two hkv d) -> ... two hkv d", two=2, d=self.head_dim)
        if inference_params is None:
            context = self._pytorch_attn(q, kv)
        else:
            context = self._update_kvcache_attention(q, kv, inference_params)
        context = rearrange(context, "... h d -> ... (h d)")
        out = self.out_proj(context)
        return out


class Block(nn.Module):
    """Simple residual block with normalization that wraps an inner "mixer" module."""

    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, residual_in_fp32=False):
        """
        dim: The dimension of the input data.
        mixer_cls: The class of the mixer module.
        norm_cls: The class of the normalization module.
        residual_in_fp32: Whether to keep residuals in fp32.
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.norm = norm_cls(dim)
        self.mixer = mixer_cls(dim)

    def forward(self, x, inference_params=None, **mixer_kwargs):
        """Forward pass through the block."""
        y = self.norm(x.to(dtype=self.norm.weight.dtype))
        y = self.mixer(y, inference_params=inference_params, **mixer_kwargs)

        residual = x
        if self.residual_in_fp32:
            residual = residual.to(torch.float32)
        y = y + residual
        y = y.to(dtype=x.dtype)

        return y

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache for the mixer module."""
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def _create_block(
    d_model,
    norm_cls,
    ssm_cfg=None,
    attn_layer_idx=None,
    attn_cfg=None,
    mlp_layer_idx=None,
    mlp_cfg=None,
    residual_in_fp32=False,
    layer_idx=None,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    if attn_layer_idx is None:
        attn_layer_idx = []
    if attn_cfg is None:
        attn_cfg = {}
    if mlp_layer_idx is None:
        mlp_layer_idx = []
    if mlp_cfg is None:
        mlp_cfg = {}
    if layer_idx in attn_layer_idx:
        mixer_cls = partial(ReneMHA, layer_idx=layer_idx, **attn_cfg, **factory_kwargs)
    elif layer_idx in mlp_layer_idx:
        mixer_cls = partial(ReneMLP, **mlp_cfg, **factory_kwargs)
    else:
        mixer_cls = partial(Mamba2, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)
    return Block(d_model, mixer_cls, norm_cls=norm_cls, residual_in_fp32=residual_in_fp32)


class MixerModel(nn.Module):
    """Adapted from mamba_ssm.models.mixer_seq_simple.MixerModel."""

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        vocab_size: int,
        ssm_cfg=None,
        attn_layer_idx=None,
        attn_cfg=None,
        mlp_layer_idx=None,
        mlp_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        residual_in_fp32=False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.residual_in_fp32 = residual_in_fp32

        if rms_norm:
            from mamba_ssm.ops.triton.layer_norm import RMSNorm as norm_cls_base
        else:
            norm_cls_base = nn.LayerNorm
        norm_cls = partial(norm_cls_base, eps=norm_epsilon, **factory_kwargs)

        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)

        self.layers = nn.ModuleList(
            [
                _create_block(
                    d_model,
                    norm_cls=norm_cls,
                    ssm_cfg=ssm_cfg,
                    attn_layer_idx=attn_layer_idx,
                    attn_cfg=attn_cfg,
                    mlp_layer_idx=mlp_layer_idx,
                    mlp_cfg=mlp_cfg,
                    residual_in_fp32=residual_in_fp32,
                    layer_idx=i,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.norm_f = norm_cls(d_model)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
                n_residuals_per_layer=1,
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache for all layers."""
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, input_ids, inference_params=None, **mixer_kwargs):
        """Forward pass through the model."""
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, inference_params=inference_params, **mixer_kwargs)
        hidden_states = self.norm_f(hidden_states.to(dtype=self.norm_f.weight.dtype))
        return hidden_states


class ReneLMHeadModel(PreTrainedModel, MambaGenerationMixin):
    """
    Rene language model architecture.
    Based on mamba_ssm.models.mixer_seq_simple.MambaLMHeadModel, with several adaptations.
    """

    config_class = ReneConfig
    base_model_prefix = "backbone"
    _no_split_modules = ["Block", "Mamba2"]
    supports_gradient_checkpointing = True
    _is_stateful = True
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(
        self,
        config: ReneConfig,
        initializer_cfg=None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(config)
        d_model = config.d_model
        n_layer = config.n_layer
        vocab_size = config.vocab_size
        ssm_cfg = config.ssm_cfg
        attn_layer_idx = config.attn_layer_idx
        attn_cfg = config.attn_cfg
        mlp_layer_idx = config.mlp_layer_idx
        mlp_cfg = config.mlp_cfg
        rms_norm = config.rms_norm
        residual_in_fp32 = config.residual_in_fp32
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        factory_kwargs = {"device": device, "dtype": dtype}

        if set(attn_layer_idx).intersection(mlp_layer_idx):
            raise ValueError(f"Conflicting {attn_layer_idx=} and {mlp_layer_idx=}")

        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)

        self.backbone = MixerModel(
            d_model=d_model,
            n_layer=n_layer,
            vocab_size=vocab_size,
            ssm_cfg=ssm_cfg,
            attn_layer_idx=attn_layer_idx,
            attn_cfg=attn_cfg,
            mlp_layer_idx=mlp_layer_idx,
            mlp_cfg=mlp_cfg,
            rms_norm=rms_norm,
            initializer_cfg=initializer_cfg,
            residual_in_fp32=residual_in_fp32,
            **factory_kwargs,
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        # Initialize weights
        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.tie_weights()

    def tie_weights(self):
        """Tie embeddings and softmax layer weights if specified by config."""
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache."""
        return self.backbone.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

    def forward(
        self, input_ids, position_ids=None, inference_params=None, num_last_tokens=0, **mixer_kwargs
    ):
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens.
        """
        hidden_states = self.backbone(input_ids, inference_params=inference_params, **mixer_kwargs)
        if num_last_tokens > 0:
            hidden_states = hidden_states[:, -num_last_tokens:]
        lm_logits = self.lm_head(hidden_states)

        return CausalLMOutput(logits=lm_logits)

    def generate(self, *args, **kwargs):
        """
        Calls the custom `generate` method from `mamba_ssm.utils.generation.GenerationMixin`.
        Refer to that method for argument names and defaults.
        """
        return MambaGenerationMixin.generate(self, *args, **kwargs)
