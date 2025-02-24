from transformers.configuration_utils import PretrainedConfig


class LlambaConfig(PretrainedConfig):
    r"""Configuration class for the CustomMamba model.

    This configuration is used to instantiate the CustomMamba model according to the specified arguments,
    defining the model architecture.

    Args:
        vocab_size (`int`, *optional*, defaults to 128256):
            Vocabulary size of the model.
        tie_embeddings (`bool`, *optional*, defaults to `False`):
            Whether the model's input and output word embeddings should be tied.
        pad_vocab_size_multiple (`int`, *optional*, defaults to 8):
            Pad the vocabulary size up to the next multiple of this value.
        lm_head_bias (`bool`, *optional*, defaults to `False`):
            Whether the LM head includes a bias term.
        d_model (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        lm_head_prenorm (`str`, *optional*, defaults to "rms"):
            Normalization type for LM head.
        n_layer (`int`, *optional*, defaults to 32):
            Number of layers in the model.
        resid_dropout (`float`, *optional*, defaults to 0.0):
            Dropout rate for residual connections.
        norm_epsilon (`float`, *optional*, defaults to 1e-5):
            Epsilon value used for normalization layers.
        mlp_cfg (`dict`, *optional*):
            Configuration for the MLP (Multi-Layer Perceptron) layer, including intermediate size, activation function, and whether to use bias.
        ssm_cfg (`dict`, *optional*):
            Configuration for the SSM (State Space Model) layer, including d_state, number of heads, expansion, and other parameters.

    """

    model_type = "llamba"

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        tie_embeddings: bool = False,
        pad_vocab_size_multiple: int = 8,
        lm_head_bias: bool = False,
        n_layer: int = 32,
        resid_dropout: float = 0.0,
        norm_epsilon: float = 1e-5,
        mlp_cfg: dict = None,
        ssm_cfg: dict = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings
        self.pad_vocab_size_multiple = pad_vocab_size_multiple
        self.lm_head_bias = lm_head_bias
        self.d_model = d_model
        self.n_layer = n_layer
        self.resid_dropout = resid_dropout
        self.norm_epsilon = norm_epsilon

        # MLP (Multi-Layer Perceptron) Config
        self.mlp_cfg = mlp_cfg or {
            "intermediate_size": 14336,
            "bias": False,
            "act_fn": "silu",
        }

        # SSM (State Space Model) Config
        self.ssm_cfg = ssm_cfg or {
            "d_state": 64,
            "n_v_heads": 32,
            "n_qk_heads": 32,
            "expand": 1,
            "chunk_size": 128,
            "activation": "identity",
            "bias": False,
        }
