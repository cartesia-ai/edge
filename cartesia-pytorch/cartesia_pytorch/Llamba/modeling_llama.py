# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
from torch import nn
from transformers.activations import ACT2FN


class LlamaRMSNorm(nn.Module):
    """LlamaRMSNorm (taken from transformers.models.llama.modeling_llama.LlamaRMSNorm)."""

    def __init__(self, hidden_size, eps=1e-6, factory_kwargs=None):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size, **factory_kwargs))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        """
        Args:
            hidden_states: torch.Tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor of shape (batch_size, seq_len, hidden_size).
        """
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        """Set the extra representation of the module."""
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class LlamaMLP(nn.Module):
    """LlamaMLP (taken from transformers.models.llama.modeling_llama.LlamaMLP)."""

    def __init__(self, hidden_size, intermediate_size, bias, act_fn, factory_kwargs=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=bias, **factory_kwargs
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=bias, **factory_kwargs
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=bias, **factory_kwargs
        )
        self.act_fn = ACT2FN[act_fn]

    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape (batch_size, seq_len, hidden_size).

        Returns:
            torch.Tensor of shape (batch_size, seq_len, hidden_size).
        """
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
