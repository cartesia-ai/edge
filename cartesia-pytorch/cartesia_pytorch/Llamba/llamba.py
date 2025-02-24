# Copyright (c) 2024, Kevin Li, Aviv Bick.

import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from mamba_ssm.utils.generation import GenerationMixin
from torch import Tensor
from transformers.utils import ModelOutput

from .configuration_llamba import LlambaConfig
from .mixers.discrete_mamba2 import DiscreteMamba2
from .modeling_llama import LlamaMLP, LlamaRMSNorm


@dataclass
class CustomMambaCausalLMOutput(ModelOutput):
    """Custom output class for MambaLMHeadModel."""

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class LlambaLMHeadModel(nn.Module, GenerationMixin, PyTorchModelHubMixin):
    """MambaLM model with a language modeling head on top (linear layer)."""

    def __init__(self, config, initializer_cfg=None, device=None, dtype=None, **kwargs) -> None:
        super().__init__()

        # Load config
        if not isinstance(config, LlambaConfig):
            config = LlambaConfig(**config)
        self.config = config

        # Factory kwargs
        factory_kwargs = {"device": device, "dtype": dtype}

        # Pad vocab size to be a multiple of pad_vocab_size_multiple
        vocab_size = config.vocab_size
        pad_vocab_size_multiple = config.pad_vocab_size_multiple
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
        self.config.vocab_size = vocab_size

        # Mixer model
        self.backbone = MixerModel(
            input_size=vocab_size,
            config=self.config,
            initializer_cfg=initializer_cfg,
            **factory_kwargs,
        )

        # LM head
        if not self.config.tie_embeddings:
            self.lm_head = nn.Linear(
                in_features=self.config.d_model,
                out_features=self.config.vocab_size,
                bias=self.config.lm_head_bias,
                **factory_kwargs,
            )
        else:
            self.lm_head = lambda x: x @ self.backbone.embedding.weight.t()

    def allocate_inference_cache(self, *args, **kwargs):
        """Allocate inference cache for the model."""
        return self.backbone.allocate_inference_cache(*args, **kwargs)

    def forward(
        self,
        input_ids,
        position_ids=None,
        return_hidden_states=False,
        return_logits=True,
        inference_params=None,
        num_last_tokens=0,
    ):
        """
        Args:
            input_ids: torch.Tensor of shape (batch_size, seq_len),
            position_ids: torch.Tensor of shape (batch_size, seq_len), optional, not used (just for compatibility),
            return_hidden_states: bool, optional,
            return_logits: bool, optional, whether to compute the logits with the LM head,
            inference_params: dict, optional, the model's inference cache,
            num_last_tokens: int, optional. If > 0, only return the logits for the last n tokens.

        Returns:
            CustomMambaCausalLMOutput.

        """
        outputs = self.backbone(
            input_ids,
            return_hidden_states=return_hidden_states,
            inference_params=inference_params,
            position_ids=position_ids,
        )

        if outputs["last_hidden_state"] is not None and return_logits:
            logits = self.lm_head(outputs["last_hidden_state"]).float()
            outputs["logits"] = logits if num_last_tokens == 0 else logits[:, -num_last_tokens:]
        else:
            outputs["logits"] = None

        return CustomMambaCausalLMOutput(
            loss=None,
            logits=outputs["logits"],
            all_hidden_states=outputs["all_hidden_states"],
            last_hidden_state=outputs["last_hidden_state"],
        )

    def save_pretrained(self, save_directory):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        # Ensure save_directory exists
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # Save the model's state_dict
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration of the model
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config.to_dict(), f)


class MixerModel(nn.Module):
    """Mixer model with a stack of Mixer layers."""

    def __init__(self, input_size, config=None, device=None, dtype=None, **kwargs) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(input_size, self.config.d_model, **factory_kwargs)

        self.layers = nn.ModuleList(
            [
                Block(
                    config=config,
                    factory_kwargs=factory_kwargs,
                    layer_idx=i,
                ).to(device)
                for i in range(self.config.n_layer)
            ]
        )

        self.final_layernorm = LlamaRMSNorm(
            hidden_size=self.config.d_model,
            eps=self.config.norm_epsilon,
            factory_kwargs=factory_kwargs,
        )

        return

    def allocate_inference_cache(self, *args, **kwargs):
        """Allocate inference cache for the model."""
        return {
            i: layer.allocate_inference_cache(*args, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(
        self,
        input_ids,
        return_hidden_states=False,
        inference_params=None,
        position_ids=None,
    ):
        """Run the model."""
        # Start running the layers
        hidden_states = self.embedding(input_ids)

        # Initialize outputs
        outputs = {
            "last_hidden_state": None,
            "all_hidden_states": (hidden_states,) if return_hidden_states else (),
        }

        # Run the layers
        for layer in self.layers:
            layer_outputs = layer(
                hidden_states,
                inference_params=inference_params,
                position_ids=position_ids,
            )
            # Record outputs
            hidden_states = layer_outputs["hidden_states"]
            if return_hidden_states:
                outputs["all_hidden_states"] += (hidden_states,)

        # Last layer, apply layer norm
        outputs["last_hidden_state"] = self.final_layernorm(hidden_states)
        return outputs


class Block(nn.Module):
    """
    Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection.

    This Block has a slightly different structure compared to a regular
    prenorm Transformer block.
    The standard block is: LN -> MHA/MLP -> Add.
    [Ref: https://arxiv.org/abs/2002.04745]
    Here we have: Add -> LN -> Mixer, returning both
    the hidden_states (output of the mixer) and the residual.
    This is purely for performance reasons, as we can fuse add and LayerNorm.
    The residual needs to be provided (except for the very first block).
    """

    def __init__(self, config, factory_kwargs, layer_idx, **kwargs):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        # Mixer
        self.mixer = DiscreteMamba2(
            d_model=self.config.d_model,
            layer_idx=layer_idx,
            **config.ssm_cfg,
            **factory_kwargs,
        )

        # Other components
        self.input_layernorm = LlamaRMSNorm(
            hidden_size=self.config.d_model, eps=1e-5, factory_kwargs=factory_kwargs
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            hidden_size=self.config.d_model, eps=1e-5, factory_kwargs=factory_kwargs
        )
        self.mlp = LlamaMLP(
            hidden_size=self.config.d_model,
            **config.mlp_cfg,
            factory_kwargs=factory_kwargs,
        )

    def forward(
        self,
        hidden_states: Tensor,
        inference_params=None,
        **kwargs,
    ):
        """
        Pass the input through the encoder layer.

        Args:
            hidden_states: torch.Tensor of shape (batch_size, seq_len, hidden_size),
            inference_params: dict, optional,

        Returns:
            dict with keys:
                hidden_states: torch.Tensor of shape (batch_size, seq_len, hidden_size),
                mamba_hidden_states: torch.Tensor of shape (batch_size, seq_len, hidden_size),
                transfer_matrix: torch.Tensor of shape (batch_size, seq_len, seq_len).
        """
        outputs = {}

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Apply Mixer
        mixer_outputs = self.mixer(
            hidden_states,
            inference_params=inference_params,
        )

        hidden_states = mixer_outputs["hidden_states"].to(residual.dtype) + residual

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs["hidden_states"] = hidden_states

        return outputs

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate inference cache for the model."""
        if getattr(self.mixer, "allocate_inference_cache", None) is None:
            return
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
