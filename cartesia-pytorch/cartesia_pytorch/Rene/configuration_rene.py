from typing import Dict, List, Optional

from transformers.configuration_utils import PretrainedConfig


class ReneConfig(PretrainedConfig):
    r"""Configuration class for the Rene model.

    This is the configuration class to store the configuration of a [`ReneLMHeadModel`].
    It is used to instantiate a Rene model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield
    a similar configuration to that of the Rene-v0.1-1.3b-pytorch model.
    [cartesia-ai/Rene-v0.1-1.3b-pytorch](https://huggingface.co/cartesia-ai/Rene-v0.1-1.3b-pytorch)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        d_model (`int`, *optional*, defaults to 2048):
            Dimension of the hidden representations.
        n_layer (`int`, *optional*, defaults to 48):
            Number of architecture blocks.
        vocab_size (`int`, *optional*, defaults to 50280):
            Vocabulary size of the Rene model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`ReneModel`].
        ssm_cfg (`dict`, *optional*):
            Configuration parameters for the SSM layers.
        attn_layer_idx (`List[int]`, *optional*):
            Indices of the architecture blocks that should have attention layers.
        attn_cfg (`dict`, *optional*):
            Configuration parameters for the attention layers.
        mlp_layer_idx (`List[int]`, *optional*):
            Indices of the architecture blocks that should have MLP layers.
        mlp_cfg (`dict`, *optional*):
            Configuration parameters for the MLP layers.
        rms_norm (`bool`, *optional*, defaults to `True`):
            Whether to use RMSNorm (instead of LayerNorm).
        residual_in_fp32 (`bool`, *optional*, defaults to `True`):
            Whether to keep residual values in fp32.
        pad_vocab_size_multiple (`int`, *optional*, defaults to 16):
            Pad the vocabulary size up to the next multiple of this value.
        tie_word_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied. Note that this is only relevant if the
            model has a output word embedding layer.
        pad_token_id (`int`, *optional*, defaults to 1):
            The id of the padding token.
        bos_token_id (`int`, *optional*):
            The id of the "beginning-of-sequence" token.
        eos_token_id (`int`, *optional*, defaults to 50279):
            The id of the "end-of-sequence" token.
    """

    model_type = "rene"

    def __init__(
        self,
        d_model: int = 2048,
        n_layer: int = 48,
        vocab_size: int = 50280,
        ssm_cfg: Optional[Dict] = None,
        attn_layer_idx: Optional[List] = None,
        attn_cfg: Optional[Dict] = None,
        mlp_layer_idx: Optional[List] = None,
        mlp_cfg: Optional[Dict] = None,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        pad_vocab_size_multiple: int = 16,
        tie_word_embeddings: bool = True,
        pad_token_id=1,
        bos_token_id=None,
        eos_token_id=50279,
        **kwargs,
    ):
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

        self.d_model = d_model
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.ssm_cfg = ssm_cfg
        self.attn_layer_idx = attn_layer_idx
        self.attn_cfg = attn_cfg
        self.mlp_layer_idx = mlp_layer_idx
        self.mlp_cfg = mlp_cfg
        self.rms_norm = rms_norm
        self.residual_in_fp32 = residual_in_fp32
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.tie_word_embeddings = tie_word_embeddings
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
