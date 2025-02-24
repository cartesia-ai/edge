import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None
from einops import repeat


class DiscreteMamba2(nn.Module):
    """DiscreteMamba2 (taken github.com/goombalab/phi-mamba.git)."""

    def __init__(
        self,
        d_model,
        d_state=64,
        n_qk_heads=32,
        n_v_heads=32,
        d_conv=4,
        expand=1,
        activation="identity",
        bias=False,
        conv_bias=True,
        chunk_size=128,
        layer_idx=None,
        device=None,
        dtype=None,
        **kwargs,
    ):
        """
        See the class .kernel.SSKernel for the kernel constructor which accepts kernel_args.
        Relevant options that are worth considering and tuning include "mode" + "measure", "dt_min", "dt_max", "lr".

        Other options are all experimental and should not need to be configured.
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.d_model
        self.n_qk_heads = n_qk_heads
        self.n_v_heads = n_v_heads
        self.headdim = self.d_inner // self.n_v_heads
        assert self.n_v_heads == self.d_inner // self.headdim
        assert self.d_inner % self.headdim == 0
        assert self.n_v_heads % self.n_qk_heads == 0
        self.activation = activation
        self.chunk_size = chunk_size
        self.layer_idx = layer_idx
        self.bias = bias
        self.kwargs = kwargs

        # Projections
        self.in_proj = nn.Linear(
            self.d_model,
            2 * self.d_inner + 2 * self.n_qk_heads * self.d_state + self.n_v_heads,
            bias=bias,
            **factory_kwargs,
        )
        self.z_bias = (
            nn.Parameter(torch.zeros(self.d_inner, **factory_kwargs)) if not bias else 0
        )  # make sure z_bias always exists

        # Convolutional layer
        conv_dim = self.d_inner + 2 * self.n_qk_heads * self.d_state
        self.conv_bias = conv_bias
        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        # Activation after conv
        if self.activation == "identity":
            self.act = nn.Identity()
        elif self.activation in ["silu", "swish"]:
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unknown activation {self.activation}")

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.n_v_heads, **factory_kwargs))

        # out_proj
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    @property
    def d_output(self):
        """Returns the output dimension of the model."""
        return self.d_model

    @property
    def state_to_tensor(self):
        """Returns the state of the model as a tensor."""
        return self.layer.state_to_tensor

    def forward(self, u, inference_params=None, **kwargs):
        """
        Args:
            u: (B, L, D),
            inference_params: dict.

        Returns:
            outputs: dict.
            outputs["hidden_states"]: (B, L, D).
            outputs["state"]: inference cache.
        """
        outputs = {}
        # assert state is None
        batch, seqlen, dim = u.shape

        state = None
        if inference_params is not None:
            state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # States are updated inplace
                u = u.squeeze(1) if len(u.shape) == 3 else u
                out, _ = self.step(u, state)
                out = out.unsqueeze(1) if len(u.shape) == 2 else out
                return {"hidden_states": out}

        # Hacky way to initialize state during inference
        chunk_size = self.chunk_size if state is None else seqlen

        # Pad input to nearest multiple of chunklen
        padded_len = (1 + (seqlen - 1) // chunk_size) * chunk_size
        u = F.pad(u, (0, 0, 0, padded_len - seqlen))

        # Project input
        xBCzA_log = self.in_proj(u)

        xBC, z, A_log = torch.split(
            xBCzA_log,
            [
                self.d_inner + 2 * self.n_qk_heads * self.d_state,
                self.d_inner,
                self.n_v_heads,
            ],
            dim=-1,
        )

        if state is not None:
            # If we just take xBC[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
            # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
            xBC_t = rearrange(xBC[:, :seqlen, :], "b l d -> b d l")
            state["conv"].copy_(
                F.pad(xBC_t, (self.d_conv - xBC_t.shape[-1], 0))
            )  # Update state (B D W)

        # Convolutional layer
        xBC = self.convolutional_forward(xBC, padded_len)

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_qk_heads * self.d_state,
                self.n_qk_heads * self.d_state,
            ],
            dim=-1,
        )

        x = rearrange(x, "b l (h n) -> b l h n", h=self.n_v_heads)
        B = rearrange(B, "b l (h n) -> b l h n", h=self.n_qk_heads)
        C = rearrange(C, "b l (h n) -> b l h n", h=self.n_qk_heads)

        # SSM forward
        result = mamba_chunk_scan_combined(
            x=x / F.softplus(A_log).to(x.dtype).unsqueeze(-1),
            dt=A_log,
            dt_softplus=True,
            A=-torch.ones(self.n_v_heads, device=A_log.device),
            B=B,
            C=C,
            chunk_size=chunk_size,
            # initial_states=(state["ssm"] if state is not None else None), # currently not supported by mamba_ssm.utils.generation
            return_final_states=(state is not None),
        )

        if state is not None:
            y, ssm_state = result
            state["ssm"].copy_(ssm_state)
        else:
            y = result

        Du = torch.einsum("h,blhp->blhp", self.D, x)
        y = rearrange(y + Du, "b l h p -> b l (h p)")

        # Norm and gate
        out = self.out_proj(y * F.silu(z + self.z_bias))
        outputs["hidden_states"] = out[:, :seqlen, :]

        return outputs

    def step(self, u, state, **kwargs):
        """
        Args:
            u: (B, D),
            state: dict.

        Returns:
            out: (B, D),
            state: dict.

        """
        # Project input
        xBCzA_log = self.in_proj(u)
        xBC, z, A_log = torch.split(
            xBCzA_log,
            [
                self.d_inner + 2 * self.n_qk_heads * self.d_state,
                self.d_inner,
                self.n_v_heads,
            ],
            dim=-1,
        )

        xBC, conv_state = self.convolutional_step(xBC, state["conv"])
        state["conv"].copy_(conv_state)  # update state in place

        x, B, C = torch.split(
            xBC,
            [
                self.d_inner,
                self.n_qk_heads * self.d_state,
                self.n_qk_heads * self.d_state,
            ],
            dim=-1,
        )

        x = rearrange(x, "b (h s) -> b h s", h=self.n_v_heads)
        B = rearrange(B, "b (h s) -> b h s", h=self.n_qk_heads)
        C = rearrange(C, "b (h s) -> b h s", h=self.n_qk_heads)

        state["ssm"] = state["ssm"].to(x.dtype)
        zeros = torch.zeros((self.n_v_heads, self.headdim), device=A_log.device).to(dtype=x.dtype)
        ones = torch.ones((self.n_v_heads, self.headdim, self.d_state), device=A_log.device).to(
            dtype=x.dtype
        )
        y = selective_state_update(
            x=x / F.softplus(A_log).to(x.dtype).unsqueeze(-1),
            dt=repeat(A_log, "b h -> b h p", p=self.headdim),
            dt_softplus=True,
            A=-ones,
            B=B,
            C=C,
            state=state["ssm"],  # will be updated in place
            dt_bias=zeros,
            D=zeros,
        )

        y = y + self.D[:, None] * x
        y = rearrange(y, "b h p -> b (h p)")

        # Norm and gate
        out = self.out_proj(y * F.silu(z + self.z_bias))

        return out, state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        """Allocate memory for inference cache."""
        device = self.in_proj.weight.device
        # conv_state:
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size,
            self.d_conv,
            self.conv1d.weight.shape[0],
            device=device,
            dtype=conv_dtype,
        ).transpose(1, 2)
        # ssm_state:
        ssm_dtype = self.in_proj.weight.dtype if dtype is None else dtype
        ssm_state = torch.zeros(
            batch_size,
            self.n_v_heads,
            self.headdim,
            self.d_state,
            device=device,
            dtype=ssm_dtype,
        )
        return {"conv": conv_state, "ssm": ssm_state}

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        """
        Get states from cache.

        conv_state: (batch, d_conv, conv1d.weight.shape[0])
        ssm_state: (batch, n_qk_heads, headdim, d_state)
        """
        assert self.layer_idx is not None
        # Allocate memory if not exists
        if self.layer_idx not in inference_params.key_value_memory_dict:
            inference_params.key_value_memory_dict[self.layer_idx] = self.allocate_inference_cache(
                batch_size, inference_params.max_seqlen, dtype=torch.float32
            )
        # Get states
        states = inference_params.key_value_memory_dict[self.layer_idx]
        if initialize_states:
            states["conv"].zero_()
            states["ssm"].zero_()
        return states

    def convolutional_forward(self, xBC, padded_len):
        """Convolutional layer forward pass for the full sequence."""
        if causal_conv1d_fn is None or self.activation not in [
            "silu",
            "swish",
            "identity",
        ]:
            xBC = self.act(self.conv1d(xBC.transpose(1, 2))[..., :padded_len].transpose(1, 2))
        else:
            xBC = causal_conv1d_fn(
                xBC.transpose(1, 2),
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                activation=None if self.activation == "identity" else self.activation,
            ).transpose(1, 2)
        return xBC

    def convolutional_step(self, xBC, conv_state):
        """Convolutional layer forward pass for a single step."""
        conv_state = conv_state.to(xBC.dtype)
        if causal_conv1d_update:
            xBC = causal_conv1d_update(
                xBC,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation if self.activation != "identity" else None,
            )
        else:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = xBC
            xBC = torch.sum(
                conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1
            )  # (B D)
            if self.conv_bias:
                xBC = xBC + self.conv1d.bias
            xBC = self.act(xBC).to(xBC.dtype)  # Some activations change dtype

        return xBC, conv_state
